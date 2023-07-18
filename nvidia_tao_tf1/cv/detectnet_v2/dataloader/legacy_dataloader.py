# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Default dataloader for DetectNet V2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import glob
import logging

from keras import backend as K

import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.augment import (
    apply_all_transformations_to_image,
    apply_spatial_transformations_to_bboxes,
    get_all_transformations_matrices,
    get_transformation_ops
)
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.base_dataloader import BaseDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.process_markers import augment_orientation_labels
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.process_markers import INVALID_ORIENTATION
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.process_markers import map_markers_to_orientations
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import extract_tfrecords_features
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_absolute_data_path
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_num_samples
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_tfrecords_iterator
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import process_image_for_dnn_input
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import read_image
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import apply_label_filters
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_crop_label_filter import BboxCropLabelFilter
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_dimensions_label_filter import (
    BboxDimensionsLabelFilter
)

import six
import tensorflow as tf

# Constants used for reading required features from tfrecords.
FRAME_ID_KEY = 'frame/id'
HEIGHT_KEY = 'frame/height'
WIDTH_KEY = 'frame/width'

UNKNOWN_CLASS = '-1'

FOLD_STRING = "fold-{:03d}-of-"

logger = logging.getLogger(__name__)


class LegacyDataloader(BaseDataloader):
    """Legacy dataloader for object detection datasets such as KITTI and Cyclops.

    Implements a data loader that reads labels and frame id from .tfrecords files and compiles
    image and ground truth tensors used in training and validation.
    """

    def __init__(self,
                 training_data_source_list,
                 target_class_mapping,
                 image_file_encoding,
                 augmentation_config,
                 validation_fold=None,
                 validation_data_source_list=None):
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
        self.training_data_sources = \
            self.get_data_sources(data_source_list=training_data_source_list,
                                  validation_fold=self.validation_fold, training=True)
        # Now, potentially, get the validation data sources.
        self.validation_data_sources = []
        if self.validation_fold is not None:
            # Take one fold from training data as validation.
            self.validation_data_sources.extend(
                self.get_data_sources(data_source_list=training_data_source_list,
                                      validation_fold=self.validation_fold, training=False))
        if validation_data_source_list is not None:
            # TODO(@williamz): This "training=True" part is really confusing.
            self.validation_data_sources.extend(
                self.get_data_sources(data_source_list=validation_data_source_list,
                                      validation_fold=None, training=True))
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
        for source_class_name, target_class_name in \
                six.iteritems(self.target_class_mapping):
            # TODO: @vpraveen Update this from lower to sentence case for running with
            # JSON based records.
            self.target_class_to_source_classes_mapping[target_class_name].\
                append(source_class_name.lower())

        self.target_class_to_source_classes_mapping = \
            dict(self.target_class_to_source_classes_mapping)

        # Get the transformation ops.
        self._stm_op, self._ctm_op = get_transformation_ops()

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
        if len(data_sources) == 0:
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
        if len(data_sources) == 0:
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
        for tfrecords_file, _ in data_sources:
            if tfrecords_file:
                tfrecords_file = tfrecords_file[0]
                break

        assert tfrecords_file, "No valid tfrecords files found in %s" % data_sources

        self.extracted_features = extract_tfrecords_features(tfrecords_file)

        # Generate augmented input images and ground truth labels.
        images, ground_truth_labels =\
            self._generate_images_and_ground_truth_labels(tfrecords_iterator,
                                                          enable_augmentation)
        # DNN input data type has to match the computation precision.
        images = tf.cast(images, dtype=K.floatx())

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

            # Load network input image tensors.
            image = self._load_input_tensors(example, img_dir)

            # Map target classes in the datasource to target classes of the model.
            example = self._map_to_model_target_classes(
                example, self.target_class_mapping)

            # Now get additional labels.
            additional_labels = self._translate_additional_labels(example)

            # Retrieve the augmentation matrices.
            sm, cm = get_all_transformations_matrices(self.augmentation_config,
                                                      enable_augmentation)

            # Apply augmentations to input image tensors.
            image, rmat = self._apply_augmentations_to_input_tensors(
                image, sm, cm, example)

            # Apply augmentations to ground truth labels.
            labels = self._apply_augmentations_to_ground_truth_labels(
                example, sm, rmat, tf.shape(image))

            # Apply augmentations to additional labels.
            additional_labels = self._apply_augmentations_to_additional_labels(
                additional_labels, sm)
            labels.update(additional_labels)

            # Do possible label filtering.
            labels = apply_label_filters(
                label_filters=[self._bbox_dimensions_label_filter,
                               self._bbox_crop_label_filter],
                ground_truth_labels=labels, mode='and')

            labels["source_weight"] = source_weight

            images.append(image)
            ground_truth_labels.append(labels)

        # Zip together the results as extracted on a per-sample basis to one entire batch.
        # What happened beforehand, on a per-image basis, happened in parallel
        # for each sample individually. From this point on, we are working with batches.
        images = tf.stack(images, axis=0)

        return images, ground_truth_labels

    def _get_parse_example_proto(self):
        """Get the maglev example proto parser.

        Returns:
            nvidia_tao_tf1.core.processors.ParseExampleProto object to parse example(s).
        """
        return nvidia_tao_tf1.core.processors.ParseExampleProto(
            features=self.extracted_features, single=True)

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
        image, rmat = apply_all_transformations_to_image(self.augmentation_config.
                                                         preprocessing.output_image_height,
                                                         self.augmentation_config.
                                                         preprocessing.output_image_width,
                                                         self._stm_op, self._ctm_op,
                                                         sm, cm, input_tensors,
                                                         self.num_input_channels)

        # Apply cropping, zero padding, resizing, and color and spatial augmentations to images.
        # HWC -> CHW
        image = process_image_for_dnn_input(image)
        return image, rmat

    def _apply_augmentations_to_ground_truth_labels(self, example, sm, rmat, image_shape):
        """
        Apply augmentations to ground truth labels.

        Args:
            example: tf.train.Example protobuf message.
            sm (2-D Tensor): 3x3 spatial transformation/augmentation matrix.
            rmat (Tensor): 3x3 matrix that transforms from augmented space to the original
                image space.
            image_shape (Tensor): Image shape.
        Returns:
            augmented_labels (dict): Ground truth labels for the frame, after preprocessing and /
                or augmentation have been applied.
        """
        augmented_labels = dict()

        # if not self.augmentation_config.preprocessing.input_mono:
        xmin, ymin, xmax, ymax = \
            apply_spatial_transformations_to_bboxes(
                sm, example['target/coordinates_x1'], example['target/coordinates_y1'],
                example['target/coordinates_x2'], example['target/coordinates_y2'])
        augmented_labels['target/bbox_coordinates'] = tf.stack(
            [xmin, ymin, xmax, ymax], axis=1)
        # TODO(@williamz): Remove the need for this redundancy.
        augmented_labels['target/coordinates_x1'] = xmin
        augmented_labels['target/coordinates_y1'] = ymin
        augmented_labels['target/coordinates_x2'] = xmax
        augmented_labels['target/coordinates_y2'] = ymax

        # Used as a frame metadata in evaluation.
        image_dimensions = tf.stack([image_shape[1:][::-1]])

        # Compile ground truth data to a list of dicts used in training and validation.
        augmented_labels['frame/augmented_to_input_matrices'] = rmat
        augmented_labels['frame/image_dimensions'] = image_dimensions

        # For anything that is unaffected by augmentation or preprocessing, forward it through.
        for feature_name, feature_tensor in six.iteritems(example):
            if feature_name not in augmented_labels:
                augmented_labels[feature_name] = feature_tensor

        # Update bbox and truncation info in example.
        # Clip cropped coordinates to image boundary.
        image_height = self.augmentation_config.preprocessing.output_image_height
        image_width = self.augmentation_config.preprocessing.output_image_width

        # if not self.augmentation_config.preprocessing.input_mono:
        augmented_labels = self._update_example_after_crop(crop_left=0,
                                                           crop_right=image_width, crop_top=0,
                                                           crop_bottom=image_height,
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

        image_path = tf.string_join(
            [image_path, '.' + self.image_file_encoding])
        image = read_image(image_path, self.image_file_encoding, self.num_input_channels,
                           width, height)

        return image

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

        if len(datasource_target_classes) > 0:
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

    @staticmethod
    def _update_example_after_crop(crop_left, crop_right, crop_top, crop_bottom, example):
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
        if all(item == 0 for item in [crop_left, crop_right, crop_top, crop_bottom]):
            return example

        if crop_left > crop_right or crop_top > crop_bottom:
            raise ValueError(
                "crop_right/crop_bottom should be larger than crop_left/crop_top.")

        crop_left = tf.cast(crop_left, tf.float32)
        crop_right = tf.cast(crop_right, tf.float32)
        crop_top = tf.cast(crop_top, tf.float32)
        crop_bottom = tf.cast(crop_bottom, tf.float32)

        # The coordinates have their origin as (0, 0) in the image.
        x1, y1, x2, y2 = tf.unstack(example['target/bbox_coordinates'], axis=1)
        if 'target/truncation_type' in example:
            # Update Truncation Type of truncated objects.
            overlap = tf.ones_like(
                example['target/object_class'], dtype=tf.bool)
            overlap = tf.logical_and(overlap, tf.less(x1, crop_right))
            overlap = tf.logical_and(overlap, tf.greater(x2, crop_left))
            overlap = tf.logical_and(overlap, tf.less(y1, crop_bottom))
            overlap = tf.logical_and(overlap, tf.greater(y2, crop_top))

            truncated = tf.zeros_like(
                example['target/truncation_type'], dtype=tf.bool)
            truncated = tf.logical_or(truncated,
                                      tf.logical_and(overlap, tf.less(x1, crop_left)))
            truncated = tf.logical_or(truncated,
                                      tf.logical_and(overlap, tf.greater(x2, crop_right)))
            truncated = tf.logical_or(truncated,
                                      tf.logical_and(overlap, tf.less(y1, crop_top)))
            truncated = tf.logical_or(truncated,
                                      tf.logical_and(overlap, tf.greater(y2, crop_bottom)))
            truncation_type =\
                tf.logical_or(truncated, tf.cast(
                    example['target/truncation_type'], dtype=tf.bool))

            example['target/truncation_type'] = tf.cast(
                truncation_type, dtype=tf.int32)

        elif 'target/truncation' in example:
            logger.debug("target/truncation is not updated to match the crop area "
                         "if the dataset contains target/truncation.")

        # Update bbox coordinates.
        new_x1 = tf.maximum(x1, crop_left)
        new_x2 = tf.minimum(x2, crop_right)
        new_y1 = tf.maximum(y1, crop_top)
        new_y2 = tf.minimum(y2, crop_bottom)

        new_augmented_coordinates = tf.stack(
            [new_x1, new_y1, new_x2, new_y2], axis=1)
        example.update({'target/bbox_coordinates': new_augmented_coordinates,
                        'target/coordinates_x1': new_x1,
                        'target/coordinates_x2': new_x2,
                        'target/coordinates_y1': new_y1,
                        'target/coordinates_y2': new_y2})

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

        if 'target/orientation' not in labels:
            if 'target/front' in labels and 'target/back' in labels:
                orientation = \
                    map_markers_to_orientations(
                        front_markers=labels['target/front'],
                        back_markers=labels['target/back'])
                additional_labels['target/orientation'] = orientation
            else:
                additional_labels['target/orientation'] = \
                    tf.ones(
                        tf.shape(labels['target/object_class'])) * INVALID_ORIENTATION

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

        if 'target/orientation' in additional_labels:
            augmented_orientation_labels = \
                augment_orientation_labels(
                    additional_labels['target/orientation'], stm)
            augmented_additional_labels['target/orientation'] = augmented_orientation_labels

        return augmented_additional_labels

    @staticmethod
    def get_data_sources(data_source_list, validation_fold, training):
        """Get data sources.

        Args:
            data_source_list: (list) List of tuples (tfrecord_file_pattern, image_directory_path).
            validation_fold: (int) Validation fold number (0-based), can be None.
            training: (bool) Whether or not this call pertains to building a training set.

        Returns:
            data_sources: (list) List of tuples (list_of_tfrecord_files, image_directory_path).

        Raises:
            AssertionError: If specified data sources were not found.
        """
        # No validation fold specified and training False means no validation data.
        if validation_fold is None and not training:
            return [([], [])]

        data_sources = []
        for tfrecords_pattern, image_dir_path in data_source_list:
            # Convert both to absolute paths.
            abs_tfrecords_pattern = get_absolute_data_path(tfrecords_pattern)
            abs_image_dir_path = get_absolute_data_path(image_dir_path)

            # Convert pattern to list of files.
            tfrecords_paths = glob.glob(abs_tfrecords_pattern)

            assert len(tfrecords_paths) > 0, \
                "No tfrecord files match pattern {}.".format(
                    abs_tfrecords_pattern)

            # Filter out files based on validation fold only if validation fold specified.
            if validation_fold is not None:
                fold_identifier = FOLD_STRING.format(validation_fold)

                if training:
                    # Take all .tfrecords files expect the one matching to the validation fold.
                    tfrecords_paths = [filename for filename in tfrecords_paths
                                       if fold_identifier not in filename]
                else:
                    # Take only the file matching to the validation fold.
                    tfrecords_paths = [filename for filename in tfrecords_paths
                                       if fold_identifier in filename]
                    assert len(tfrecords_paths) != 0, "Cannot find val tfrecords for fold {}"\
                        "for tfrecord: {}. Please check the validation fold number and retry".\
                        format(validation_fold, tfrecords_pattern)

            data_sources.append((tfrecords_paths, abs_image_dir_path))

        return data_sources
