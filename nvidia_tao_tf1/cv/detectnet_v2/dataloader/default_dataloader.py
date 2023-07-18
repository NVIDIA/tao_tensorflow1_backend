# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Default dataloader for DetectNet V2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import logging

from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import \
    augment_marker_labels
import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.augment import (
    apply_all_transformations_to_image,
    apply_spatial_transformations_to_polygons,
    get_all_transformations_matrices,
    get_transformation_ops
)
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.base_dataloader import BaseDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import (
    extract_tfrecords_features,
    get_num_samples,
    get_tfrecords_iterator,
    process_image_for_dnn_input,
    read_image
)
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import apply_label_filters
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_crop_label_filter import BboxCropLabelFilter
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_dimensions_label_filter import (
    BboxDimensionsLabelFilter
)

import six
from six.moves import zip
import tensorflow as tf

# Constants used for reading required features from tfrecords.
BW_POLY_COEFF1_60FC = 0.000545421498827636
FRAME_ID_KEY = 'frame/id'
HEIGHT_KEY = 'frame/height'
WIDTH_KEY = 'frame/width'

UNKNOWN_CLASS = '-1'
UNKNOWN_ORIENTATION = 0.0
UNKNOWN_DISTANCE = 0.0

logger = logging.getLogger(__name__)


class DefaultDataloader(BaseDataloader):
    """Default dataloader for object detection datasets such as KITTI and Cyclops.

    Implements a data loader that reads labels and frame id from .tfrecords files and compiles
    image and ground truth tensors used in training and validation.
    """

    def __init__(self,
                 training_data_source_list,
                 image_file_encoding,
                 augmentation_config,
                 validation_fold=None,
                 validation_data_source_list=None,
                 target_class_mapping=None):
        """Instantiate the dataloader.

        Args:
            training_data_source_list (list): List of tuples (tfrecord_file_pattern,
                image_directory_path) to use for training.
            target_class_mapping (dict): maps from source class to target class (both str).
            image_file_encoding (str): How the images to be produced by the dataset are encoded.
                Can be e.g. "jpg", "fp16", "png".
            augmentation_config (nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation_config.
                AugmentationConfig): Holds the parameters for augmentation and preprocessing.
            validation_fold (int): Validation fold number (0-based). Indicates which fold from the
                training data to use as validation. Can be None.
            validation_data_source_list (list): List of tuples (tfrecord_file_pattern,
                image_directory_path) to use for validation. Can be None.
        """
        self.target_class_mapping = target_class_mapping
        self.image_file_encoding = image_file_encoding
        self.augmentation_config = augmentation_config
        self.validation_fold = validation_fold

        # Get training data sources.
        self.training_data_sources = training_data_source_list
        # Now, potentially, get the validation data sources.
        self.validation_data_sources = validation_data_source_list
        if augmentation_config is None:
            self.num_input_channels = 3
        else:
            self.num_input_channels = self.augmentation_config.preprocessing.output_image_channel

        assert(self.num_input_channels in [1, 3]), "Set the output_image_channel param to 1 " \
                                                   "or 3 in the augmentation config."
        # TODO(@williamz): Why do some tests supply None as augmentation_config? Should we make
        #  a kwarg?
        if augmentation_config is None:
            bbox_dimensions_label_filter_kwargs = dict()
            bbox_crop_label_filter_kwargs = dict()
        else:
            bbox_dimensions_label_filter_kwargs = {
                'min_width': self.augmentation_config.preprocessing.min_bbox_width,
                'min_height': self.augmentation_config.preprocessing.min_bbox_height}
            bbox_crop_label_filter_kwargs = {
                'crop_left': 0,
                'crop_right': self.augmentation_config.preprocessing.output_image_width,
                'crop_top': 0,
                'crop_bottom': self.augmentation_config.preprocessing.output_image_height}
        self._bbox_dimensions_label_filter = \
            BboxDimensionsLabelFilter(**bbox_dimensions_label_filter_kwargs)
        self._bbox_crop_label_filter = \
            BboxCropLabelFilter(**bbox_crop_label_filter_kwargs)

        self.target_class_to_source_classes_mapping = defaultdict(list)
        if self.target_class_mapping is not None:
            for source_class_name, target_class_name in \
                    six.iteritems(self.target_class_mapping):
                self.target_class_to_source_classes_mapping[
                    target_class_name].append(source_class_name.lower())

        self.target_class_to_source_classes_mapping = \
            dict(self.target_class_to_source_classes_mapping)

        # Get the transformation ops.
        self._stm_op, self._ctm_op = get_transformation_ops()

        # For parsing TF records.
        self._extracted_features = dict()

    def get_data_tensor_shape(self):
        """Interface for querying data tensor shape.

        Returns:
            Data tensor shape as a (C,H,W) tuple without the batch dimension.
        """
        return (self.num_input_channels,
                self.augmentation_config.preprocessing.output_image_height,
                self.augmentation_config.preprocessing.output_image_width)

    def get_num_samples(self, training):
        """Get number of dataset samples.

        Args:
            training (bool): Get number of samples in the training (true) or
                validation (false) set.

        Returns:
            Number of samples in the chosen set.
        """
        data_sources = self.training_data_sources if training else self.validation_data_sources

        # In case file list is empty, don't load anything.
        if not data_sources:
            return 0

        return get_num_samples(data_sources, training)

    def get_dataset_tensors(self, batch_size, training, enable_augmentation, repeat=True):
        """Get input images and ground truth labels as tensors for training and validation.

        Returns also the number of minibatches required to loop over the dataset once.

        Args:
            batch_size (int): Minibatch size.
            training (bool): Get samples from the training (True) or validation (False) set.
            enable_augmentation (bool): Whether to augment input images and labels.
            repeat (bool): Whether the dataset can be looped over multiple times or only once.

        Returns:
            images (Tensor of shape (batch, channels, height, width)): Input images with values
                in the [0, 1] range.
            ground_truth_labels (list of dicts of Tensors): Each element in this list corresponds
                to the augmented and filtered labels in a frame.
            num_samples (int): Total number of samples found in the dataset.
        """
        data_sources = self.training_data_sources if training else self.validation_data_sources

        # In case file list is empty, don't load anything
        if not data_sources:
            return None, None, 0

        # Get the location of .tfrecords files for each validation fold and the location of images
        tfrecords_iterator, num_samples = get_tfrecords_iterator(data_sources,
                                                                 batch_size,
                                                                 training=training,
                                                                 repeat=repeat)

        # Extract features from a sample tfrecords file. These features are then read from all
        # tfrecords files.
        tfrecords_file = None

        # Find the first tfrecord file.
        at_least_one_tfrecord = False
        for data_source_config in data_sources:
            if data_source_config.dataset_files:
                at_least_one_tfrecord = True
                # Assume all tfrecords in a single source will have the same schema.
                tfrecords_file = data_source_config.dataset_files[0]
                self._extracted_features.update(extract_tfrecords_features(tfrecords_file))

        assert at_least_one_tfrecord, "No valid tfrecords files found in %s" % data_sources

        # Generate augmented input images and ground truth labels.
        images, ground_truth_labels =\
            self._generate_images_and_ground_truth_labels(tfrecords_iterator,
                                                          enable_augmentation)
        return images, ground_truth_labels, num_samples

    def _generate_images_and_ground_truth_labels(self, tfrecords_iterator,
                                                 enable_augmentation=False):
        """Return generators for input image and output target tensors.

        Args:
            tfrecords_iterator (TFRecordsIterator): Iterator for dataset .tfrecords files.
            enable_augmentation (bool): Augment input images and ground truths.

        Returns:
            images (Tensor of shape (batch, channels, height, width)): Input images with values
                in the [0, 1] range.
            ground_truth_labels (list of dicts of Tensors): Each dict contains e.g. tensors for
                the augmented bbox coordinates, their class name, etc.
        """
        # Create the proto parser.
        parse_example_proto_layer = self._get_parse_example_proto()

        # We first yield our tfrecords, by calling the processor we created earlier.
        # This will return a tuple - list of individual samples, that all contain 1 record
        # and a list of image directory paths, 1 for each record.
        records, img_dirs, source_weights = tfrecords_iterator()

        # Loop over each record, and deserialize the example proto. This will yield the tensors.
        # Both the number of records and the loop's length are the same as the batch size.
        # We are repeating the same operation for each item in the batch. Our batch size is hence
        # fixed.
        images = []
        ground_truth_labels = []

        for record, img_dir, source_weight in zip(records, img_dirs, source_weights):
            # Deserialize the record. It will yield a dictionary of items in this proto.
            # Inside this (now deserialized) Example is the image, label, metadata, etc.
            example = parse_example_proto_layer(record)  # Returns a dict.
            # Map target classes in the datasource to target classes of the model.
            if self.target_class_mapping is not None:
                example = self._map_to_model_target_classes(
                    example, self.target_class_mapping)

            # Load network input image tensors.
            image = self._load_input_tensors(example, img_dir)

            labels, image = self._augment(example, image, enable_augmentation)
            labels["source_weight"] = source_weight
            images.append(image)
            ground_truth_labels.append(labels)

        # Zip together the results as extracted on a per-sample basis to one entire batch.
        # What happened beforehand, on a per-image basis, happened in parallel
        # for each sample individually. From this point on, we are working with batches.
        images = tf.stack(images, axis=0)

        return images, ground_truth_labels

    def _augment(self, example, input_tensors, enable_augmentation):
        """Apply augmentation operations to example.

        Args:
            example: tf.train.Example protobuf message.
            input_tensors (3-D Tensor, HWC): Input tensors.
            enable_augmentation (boolean): True if random augmentations are enabled.
        Returns:
            labels (dict): Augmented labels.
            input_tensors (3-D Tensor, CHW): Augmented input tensors.
        """
        # Retrieve the augmentation matrices.
        sm, cm = get_all_transformations_matrices(self.augmentation_config,
                                                  enable_augmentation)

        # Get additional labels.
        additional_labels = self._translate_additional_labels(example)

        # The old format for bbox labels in TFRecords was 'target/coordinates_x1',
        # 'target/coordinates_x2', 'target/coordinates_y1', 'target/coordinates_y2'.
        # The new format has 'target/coordinates/x', 'target/coordinates/y',
        # and 'target/coordinates/index' which more closely resembles how arbitrary polygons
        # might be specified. The following call ensures the old format gets translated to the new
        # one.
        example = self.translate_bbox_to_polygon(example)

        # Apply augmentations to input tensors.
        input_tensors, rmat = self._apply_augmentations_to_input_tensors(
            input_tensors, sm, cm, example)

        # Apply augmentations to ground truth labels.
        labels = self._apply_augmentations_to_ground_truth_labels(
            example, sm, rmat)

        # Apply augmentations to additional labels.
        additional_labels = self._apply_augmentations_to_additional_labels(
            additional_labels, sm)
        labels.update(additional_labels)

        # Do possible label filtering.
        labels = apply_label_filters(
            label_filters=[self._bbox_dimensions_label_filter, self._bbox_crop_label_filter],
            ground_truth_labels=labels,
            mode='and')

        return labels, input_tensors

    @staticmethod
    def translate_bbox_to_polygon(example):
        """Cast all bounding box coordinates to polygon coordinates, if they exist.

        Args:
            example (dict): Labels for one sample.

        Returns:
            new_example (dict): Where bounding box labels, if any, have been cast to polygon
            coordinates.
        """
        needed_bbox_features = {'target/coordinates_x1', 'target/coordinates_x2',
                                'target/coordinates_y1', 'target/coordinates_y2'}
        is_bbox_present = all(f in example for f in needed_bbox_features)
        needed_polygon_features = {'target/coordinates/x', 'target/coordinates/y',
                                   'target/coordinates/index'}
        is_polygon_present = all(f in example for f in needed_polygon_features)

        if is_polygon_present and not is_bbox_present:
            # No need to convert.
            return example
        if is_polygon_present and is_bbox_present:
            logger.warning('The data sources, combined, have both old-style bbox coordinates, '
                           'and new-style polygon vertices. Translating the old ones to new ones '
                           'where applicable')
        elif is_bbox_present:
            # The tfrecord is guaranteed to have bbox features. Convert them to polygon features.
            logger.info("Bounding box coordinates were detected in the input specification! Bboxes"
                        " will be automatically converted to polygon coordinates.")

        x1, x2, y1, y2 = (example['target/coordinates_x1'], example['target/coordinates_x2'],
                          example['target/coordinates_y1'], example['target/coordinates_y2'])

        # Create an index like 0,0,0,0, 1,1,1,1, ... n-1,n-1,n-1,n-1, where N is the number of
        #  bboxes.
        num_bboxes = tf.size(x1)
        coordinates_index = tf.cast(tf.floor(tf.range(num_bboxes, delta=0.25)), dtype=tf.int64)

        # Construct polygon bounding boxes with the coordinate ordering TL, TR, BR, BL.
        coordinates_x = tf.reshape(tf.stack([x1, x2, x2, x1], axis=1), shape=(-1,))
        coordinates_y = tf.reshape(tf.stack([y1, y1, y2, y2], axis=1), shape=(-1,))

        if is_polygon_present and is_bbox_present:
            # If these are empty, replace the coordinates with the polygon vertices.
            is_empty = tf.equal(num_bboxes, 0)
            coordinates_x = tf.cond(is_empty,
                                    lambda: example['target/coordinates/x'],
                                    lambda: coordinates_x)
            coordinates_y = tf.cond(is_empty,
                                    lambda: example['target/coordinates/y'],
                                    lambda: coordinates_y)
            coordinates_index = tf.cond(is_empty,
                                        lambda: example['target/coordinates/index'],
                                        lambda: coordinates_index)

        new_example = {
            k: v
            for k, v in six.iteritems(example)
            if k not in needed_bbox_features
        }
        new_example.update({
            'target/coordinates/x': coordinates_x,
            'target/coordinates/y': coordinates_y,
            'target/coordinates/index': coordinates_index
        })

        return new_example

    def _get_parse_example_proto(self):
        """Get the maglev example proto parser.

        Returns:
            nvidia_tao_tf1.core.processors.ParseExampleProto object to parse example(s).
        """
        return nvidia_tao_tf1.core.processors.ParseExampleProto(
                features=self._extracted_features, single=True)

    def _apply_augmentations_to_input_tensors(self, input_tensors, sm, cm, example):
        """
        Apply augmentations to input image tensors.

        Args:
            input_tensors (3-D Tensor, HWC): Input image tensors.
            sm (2-D Tensor): 3x3 spatial transformation/augmentation matrix.
            cm (2-D Tensor): 3x3 color augmentation matrix.
            example: tf.train.Example protobuf message. (Unused here but used in subclasses.)
        Returns:
            image (Tensor, CHW): Augmented input tensor. The values are scaled between [0, 1].
            rmat: Matrix that transforms from augmented space to the original image space.
        """
        # Apply cropping, zero padding, resizing, and color and spatial augmentations to images.
        image, rmat = \
            apply_all_transformations_to_image(
                self.augmentation_config.preprocessing.output_image_height,
                self.augmentation_config.preprocessing.output_image_width,
                self._stm_op, self._ctm_op, sm, cm, input_tensors, self.num_input_channels)

        # Apply cropping, zero padding, resizing, and color and spatial augmentations to images.
        # HWC -> CHW
        image = process_image_for_dnn_input(image)
        return image, rmat

    def _apply_augmentations_to_ground_truth_labels(self, example, sm, rmat):
        """
        Apply augmentations to ground truth labels.

        Args:
            example: tf.train.Example protobuf message.
            sm (2-D Tensor): 3x3 spatial transformation/augmentation matrix.
            rmat (Tensor): 3x3 matrix that transforms from augmented space to the original
                image space.
        Returns:
            augmented_labels (dict): Ground truth labels for the frame, after preprocessing and /
                or augmentation have been applied.
        """
        augmented_labels = dict()
        augmented_x, augmented_y = apply_spatial_transformations_to_polygons(
            sm, example['target/coordinates/x'], example['target/coordinates/y'])

        augmented_labels['target/coordinates/x'] = augmented_x
        augmented_labels['target/coordinates/y'] = augmented_y
        self._update_bbox_from_polygon_coords(example)

        # Used as a frame metadata in evaluation.
        image_height = self.augmentation_config.preprocessing.output_image_height
        image_width = self.augmentation_config.preprocessing.output_image_width
        image_dimensions = tf.constant([[image_width, image_height]])

        # Compile ground truth data to a list of dicts used in training and validation.
        augmented_labels['frame/augmented_to_input_matrices'] = rmat
        augmented_labels['frame/image_dimensions'] = image_dimensions

        if 'target/front' in example and 'target/back' in example:
            augmented_front_labels = \
                augment_marker_labels(example['target/front'], sm)
            augmented_labels['target/front'] = augmented_front_labels
            augmented_back_labels = \
                augment_marker_labels(example['target/back'], sm)
            augmented_labels['target/back'] = augmented_back_labels

        # For anything that is unaffected by augmentation or preprocessing, forward it through.
        for feature_name, feature_tensor in six.iteritems(example):
            if feature_name not in augmented_labels:
                augmented_labels[feature_name] = feature_tensor

        # Update bbox and truncation info in example.
        # Clip cropped coordinates to image boundary.
        augmented_labels = self._update_example_after_crop(crop_left=0, crop_right=image_width,
                                                           crop_top=0, crop_bottom=image_height,
                                                           example=augmented_labels)

        return augmented_labels

    def _load_input_tensors(self, example, file_dir):
        """
        Return a generator for the input image tensors.

        Args:
            example: tf.train.Example protobuf message.
            file_dir (string): Dataset input image directory.
        Returns:
            image (3-D Tensor, HWC): The image.
        """
        # Reshape image_path to have rank 0 as expected by TensorFlow's ReadFile.
        image_path = tf.string_join([file_dir, example[FRAME_ID_KEY]])
        image_path = tf.reshape(image_path, [])

        height, width = tf.reshape(example[HEIGHT_KEY], []), \
            tf.reshape(example[WIDTH_KEY], [])

        image_path = tf.string_join([image_path, '.' + self.image_file_encoding])
        image = read_image(image_path, self.image_file_encoding, self.num_input_channels,
                           width, height)

        return image

    @staticmethod
    def _update_bbox_from_polygon_coords(example):
        """Update the non-rotated bounding rectangle of a polygon from its coordinates.

        Args:
            example (dict): Labels for one sample.

        Returns:
            example (dict): A reference to the now modified example.
        """
        coord_x = example['target/coordinates/x']
        coord_y = example['target/coordinates/y']
        coord_idx = example['target/coordinates/index']

        xmin = tf.math.segment_min(coord_x, coord_idx)
        xmax = tf.math.segment_max(coord_x, coord_idx)

        ymin = tf.math.segment_min(coord_y, coord_idx)
        ymax = tf.math.segment_max(coord_y, coord_idx)

        example['target/bbox_coordinates'] = tf.stack([xmin, ymin, xmax, ymax], axis=1)
        return example

    @classmethod
    def _update_example_after_crop(cls, crop_left, crop_right, crop_top, crop_bottom, example):
        """Update bbox and truncation_type according to cropping preprocess.

        Args:
            crop_left/crop_right/crop_top/crop_bottom (int): crop rectangle coordinates.
            example (tf.train.Example): Labels for one sample.

        Returns
            example (tf.train.Example): Labels where bbox and truncation_type are updated according
            to crop preprocess.

        Raises:
            ValueError: if crop_left > crop_right, or crop_top > crop_bottom, raise error.
        """
        coord_x = example['target/coordinates/x']
        coord_y = example['target/coordinates/y']
        coord_idx = example['target/coordinates/index']

        # TODO(@drendleman) Use the Maglev ClipPolygon transformer here?
        if all(item == 0 for item in [crop_left, crop_right, crop_top, crop_bottom]):
            cls._update_bbox_from_polygon_coords(example)
            return example

        if crop_left > crop_right or crop_top > crop_bottom:
            raise ValueError("crop_right/crop_bottom should be larger than crop_left/crop_top.")

        crop_left = tf.cast(crop_left, tf.float32)
        crop_right = tf.cast(crop_right, tf.float32)
        crop_top = tf.cast(crop_top, tf.float32)
        crop_bottom = tf.cast(crop_bottom, tf.float32)

        # The coordinates have their origin as (0, 0) in the image.
        if 'target/truncation_type' in example:
            # Update Truncation Type of truncated objects.
            # Overlap: is any single vertex per each polygon inside the crop region?
            overlap = tf.ones_like(coord_idx, dtype=tf.bool)
            overlap = tf.logical_and(overlap, tf.less(coord_x, crop_right))
            overlap = tf.logical_and(overlap, tf.greater(coord_x, crop_left))
            overlap = tf.logical_and(overlap, tf.less(coord_y, crop_bottom))
            overlap = tf.logical_and(overlap, tf.greater(coord_y, crop_top))

            # Logical OR together all overlapped coordinate statuses for each polygon.
            overlap = tf.math.segment_max(tf.cast(overlap, dtype=tf.int32), coord_idx)

            # Truncated: is any single vertex per each polygon outside the crop region?
            truncated = tf.zeros_like(coord_idx, dtype=tf.bool)
            truncated = tf.logical_or(truncated, tf.less(coord_x, crop_left))
            truncated = tf.logical_or(truncated, tf.greater(coord_x, crop_right))
            truncated = tf.logical_or(truncated, tf.less(coord_y, crop_top))
            truncated = tf.logical_or(truncated, tf.greater(coord_y, crop_bottom))

            # Logical OR all truncated coordinate statuses for each polygon.
            truncated = tf.math.segment_max(tf.cast(truncated, dtype=tf.int32), coord_idx)

            # Ensure an object is still truncated if it was originally truncated.
            truncation_type = \
                tf.logical_and(tf.cast(truncated, dtype=tf.bool), tf.cast(overlap, dtype=tf.bool))
            truncation_type = \
                tf.logical_or(tf.cast(example['target/truncation_type'], dtype=tf.bool),
                              truncation_type)

            example['target/truncation_type'] = tf.cast(truncation_type, dtype=tf.int32)

        elif 'target/truncation' in example:
            logger.debug("target/truncation is not updated to match the crop area "
                         "if the dataset contains target/truncation.")

        # Update bbox coordinates.
        # TODO(@drendleman) We can't use clip_by_value here because of a tensorflow bug when both
        #  the tensor and the clip values are empty.
        truncated_x = tf.minimum(tf.maximum(example['target/coordinates/x'], crop_left), crop_right)
        truncated_y = tf.minimum(tf.maximum(example['target/coordinates/y'], crop_top), crop_bottom)

        example.update({'target/coordinates/x': truncated_x,
                        'target/coordinates/y': truncated_y})
        cls._update_bbox_from_polygon_coords(example)

        return example

    def _map_to_model_target_classes(self, example, target_class_mapping):
        """Map object classes as they are defined in the data source to the model target classes.

        Args:
            example (tf.train.Example): Labels for one sample.
            target_class_mapping: Protobuf map.

        Returns
            example (tf.train.Example): Labels where data source target classes are mapped to
                model target classes. If target_class_mapping is not defined, then example is
                unchanged.
        """
        datasource_target_classes = list(target_class_mapping.keys())

        if datasource_target_classes:
            mapped_target_classes = list(target_class_mapping.values())
            default_value = tf.constant(UNKNOWN_CLASS)

            lookup = nvidia_tao_tf1.core.processors.LookupTable(keys=datasource_target_classes,
                                                    values=mapped_target_classes,
                                                    default_value=default_value)

            # Retain source class.
            example['target/source_class'] = example['target/object_class']

            # Overwrite 'object_class' with mapped target class.
            new_target_classes = lookup(example['target/object_class'])
            example['target/object_class'] = new_target_classes

        return example

    def _translate_additional_labels(self, labels):
        """Translate additional labels if required.

        This private helper takes care of parsing labels on top of those needed for 'bare 2D'
        detection, and translating them to the domain expected by the model.
        E.g. This can translate (front, back) markers to an orientation value.

        Args:
            labels (dict): Keys are label feature names, values the corresponding tf.Tensor.

        Returns:
            additional_labels (dict): Keys are label feature names produced from the translation,
                the values the corresponding tf.Tensor.
        """
        additional_labels = dict()

        depth_name = 'target/world_bbox_z'

        if depth_name in labels:
            additional_labels[depth_name] = labels[depth_name]
            # Now adjust to camera if the information is present.
            if 'frame/bw_poly_coeff1' in labels:
                # Use the ratio of the first order backward polynomial coefficients as the scaling
                # factor. Default camera is 60FOV, and this is its first order bw-poly coeff.
                scale_factor = labels['frame/bw_poly_coeff1'] / \
                    BW_POLY_COEFF1_60FC
                additional_labels[depth_name] *= scale_factor
        else:
            additional_labels[depth_name] = \
                tf.ones(tf.shape(input=labels['target/object_class'])) * UNKNOWN_DISTANCE

        return additional_labels

    def _apply_augmentations_to_additional_labels(self, additional_labels, stm):
        """Apply augmentations to additional labels.

        This private helper applies augmentations (currently only spatial augmentations) to those
        labels produced by _translate_additional_labels().

        Args:
            additional_labels (dict): Keys are (additional) label feature names, values the
                corresponding tf.Tensor.
            stm (tf.Tensor): 3x3 Spatial transformation matrix.

        Returns:
            augmented_additional_labels (dict): Keys are the same as <additional_labels>, values the
                corresponding tf.Tensor with augmentation applied to them.
        """
        augmented_additional_labels = dict()
        augmented_additional_labels.update(additional_labels)

        if 'target/world_bbox_z' in additional_labels:
            # Zoom factor is the square root of the inverse of the determinant of the left-top 2x2
            # corner of the spatial transformation matrix.
            abs_determinant = tf.abs(tf.linalg.det(stm[:2, :2]))
            # Although in practice the spatial transaformation matrix should always be invertible,
            # add a runtime check here.
            with tf.control_dependencies([tf.compat.v1.assert_greater(abs_determinant, 0.001)]):
                scale_factor = 1. / tf.sqrt(abs_determinant)
                augmented_additional_labels['target/world_bbox_z'] *= scale_factor

        return augmented_additional_labels
