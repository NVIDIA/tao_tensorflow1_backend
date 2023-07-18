# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataloader for DriveNet based on dlav common dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from keras import backend as K

import nvidia_tao_tf1.core
from nvidia_tao_tf1.blocks.multi_source_loader import CHANNELS_FIRST
from nvidia_tao_tf1.blocks.multi_source_loader.data_loader import DataLoader
from nvidia_tao_tf1.blocks.multi_source_loader.processors import (
    BboxClipper,
    Crop,
    Pipeline,
    RandomBrightness,
    RandomContrast,
    RandomFlip,
    RandomHueSaturation,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Scale,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources import (
    TFRecordsDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Bbox2DLabel,
    Coordinates2D,
    FEATURE_CAMERA,
    filter_bbox_label_based_on_minimum_dims,
    Images2DReference,
    LABEL_OBJECT,
    SequenceExample,
    set_auto_resize,
    set_image_channels,
    set_max_side,
    set_min_side,
    sparsify_dense_coordinates,
    vector_and_counts_to_sparse_tensor,
)

from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import DefaultDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import FRAME_ID_KEY
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import HEIGHT_KEY
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import UNKNOWN_CLASS
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import WIDTH_KEY
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import extract_tfrecords_features
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import filter_labels

import tensorflow as tf

Canvas2D = nvidia_tao_tf1.core.types.Canvas2D
BW_POLY_COEFF1_60FC = 0.000545421498827636
logger = logging.getLogger()


class DriveNetTFRecordsParser(object):
    """Parse tf.train.Example protos into DriveNet Examples."""

    def __init__(
        self, tfrecord_path, image_dir, extension,
        channels, source_weight=1.0, auto_resize=False
    ):
        """Construct a parser for drivenet labels.

        Args:
            tfrecord_path (list): List of paths to tfrecords file.
            image_dir (str): Path to the directory where images are contained.
            extension (str): Extension for images that get loaded (
                ".fp16", ".png", ".jpg" or ".jpeg").
            channels (int): Number of channels in each image.
            auto_resize(bool): Flag to enable automatic image resize or not.

        Raises:
            ValueError: If the number of input channels is not unsupported (i.e. must be equal to 3)
        """
        if channels not in [1, 3]:
            raise ValueError("DriveNetTFRecordsParser: unsupported number of channels %d." %
                             channels)
        self._image_file_extension = extension
        # These will be set once all data sources have been instantiated and the common
        # maximum image size is known.
        self._output_height = None
        self._output_width = None
        self._output_min = None
        self._output_max = None
        self._num_input_channels = channels
        set_image_channels(self._num_input_channels)
        self._image_dir = image_dir
        if not self._image_dir.endswith('/'):
            self._image_dir += '/'

        self._tfrecord_path = tfrecord_path
        # Delay the actual definition to call time.
        self._parse_example = None

        # Set the source_weight.
        self.source_weight = source_weight
        # auto_resize
        self.auto_resize = auto_resize
        set_auto_resize(self.auto_resize)

    def _get_parse_example(self):
        if self._parse_example is None:
            extracted_features = extract_tfrecords_features(self._tfrecord_path[0])

            self._parse_example = nvidia_tao_tf1.core.processors.ParseExampleProto(
                features=extracted_features,
                single=True
            )
        return self._parse_example

    def set_target_size(self, height, width, min_side=None, max_side=None):
        """Set size for target image.

        Args:
            height (int): Target image height.
            width (int): Target image width.
            min_side(int): Target minimal side of the image(either width or height)
            max_side(int): The larger side of the image(the one other than min_side).
        """
        self._output_height = height
        self._output_width = width
        self._output_min = min_side
        self._output_max = max_side
        set_min_side(min_side)
        set_max_side(max_side)

    def __call__(self, tfrecord):
        """Parse a tfrecord.

        Args:
            tfrecord (tensor): a serialized example proto.

        Returns:
            (Example) Example compatible with Processors.
        """
        example = self._get_parse_example()(tfrecord)
        example = DefaultDataloader.translate_bbox_to_polygon(example)

        # Reshape to have rank 0.
        height = tf.cast(tf.reshape(example[HEIGHT_KEY], []), dtype=tf.int32)
        width = tf.cast(tf.reshape(example[WIDTH_KEY], []), dtype=tf.int32)

        example[HEIGHT_KEY] = height
        example[WIDTH_KEY] = width

        # Reshape image_path to have rank 0 as expected by TensorFlow's ReadFile.
        image_path = tf.strings.join([self._image_dir, example[FRAME_ID_KEY]])
        image_path = tf.reshape(image_path, [])

        extension = tf.convert_to_tensor(value=self._image_file_extension)
        image_path = tf.strings.join([image_path, extension])

        labels = self._extract_bbox_labels(example)

        # @vpraveen: This is the point where the image datastructure is populated. The loading
        # and decoding functions are defined as member variables in Images2DReference.
        return SequenceExample(
            instances={
                FEATURE_CAMERA: Images2DReference(
                    path=image_path,
                    extension=extension,
                    canvas_shape=Canvas2D(
                        height=tf.ones([self._output_height]),
                        width=tf.ones([self._output_width])),
                    input_height=height,
                    input_width=width
                ),
                # TODO(@williamz): This is where FEATURE_SESSION: Session() would be populated
                # if we ever went down that path.
            },
            labels={LABEL_OBJECT: labels}
        )

    def _extract_depth(self, example):
        """Extract depth label.

        Args:
            example (dict): Maps from feature name (str) to tf.Tensor.

        Returns:
            depth (tf.Tensor): depth values with possible scale adjustments.
        """
        depth = example['target/world_bbox_z']

        # Use the ratio of the first order backward polynomial coefficients as the scaling factor.
        # Default camera is 60 degree camera, and this is the first order bw-poly coeff of it.
        if 'frame/bw_poly_coeff1' in example:
            scale_factor = example['frame/bw_poly_coeff1'] / \
                BW_POLY_COEFF1_60FC
        else:
            scale_factor = 1.0

        depth *= scale_factor

        return depth

    def _extract_bbox_labels(self, example):
        """Extract relevant features from labels.

        Args:
            example (dict): Maps from feature name (str) to tf.Tensor.

        Returns:
            bbox_label (Bbox2DLabel): Named tuple containing all the feature in tf.SparseTensor
                form.
        """
        # Cast polygons to rectangles. For polygon support, use SQLite.
        coord_x = example['target/coordinates/x']
        coord_y = example['target/coordinates/y']
        coord_idx = example['target/coordinates/index']
        xmin = tf.math.segment_min(coord_x, coord_idx)
        xmax = tf.math.segment_max(coord_x, coord_idx)
        ymin = tf.math.segment_min(coord_y, coord_idx)
        ymax = tf.math.segment_max(coord_y, coord_idx)

        # Massage the above to get a [N, 2] tensor. N refers to the number of vertices, so 2
        # per bounding box, and always in (x, y) order.
        dense_coordinates = tf.reshape(
            tf.stack([xmin, ymin, xmax, ymax], axis=1),
            (-1, 2))
        # scale coordinates if resize and keep AR
        if self._output_min > 0:
            height = tf.cast(tf.reshape(example[HEIGHT_KEY], []), dtype=tf.int32)
            width = tf.cast(tf.reshape(example[WIDTH_KEY], []), dtype=tf.int32)
            dense_coordinates = self._scale_coordinates(
                dense_coordinates, height,
                width, self._output_min,
                self._output_max
            )
        else:
            # resize coordidnates to target size without keeping aspect ratio
            if self.auto_resize:
                height = tf.cast(tf.reshape(example[HEIGHT_KEY], []), dtype=tf.int32)
                width = tf.cast(tf.reshape(example[WIDTH_KEY], []), dtype=tf.int32)
                dense_coordinates = self._resize_coordinates(
                    dense_coordinates, height,
                    width, self._output_height,
                    self._output_width
                )
        counts = tf.ones_like(example['target/object_class'], dtype=tf.int64)
        # 2 vertices per bounding box (since we can infer the other 2 using just these).
        vertex_counts_per_polygon = 2 * counts

        sparse_coordinates = \
            sparsify_dense_coordinates(dense_coordinates, vertex_counts_per_polygon)

        # This will be used to instantiate the namedtuple Bbox2DLabel.
        bbox_2d_label_kwargs = dict()
        bbox_2d_label_kwargs['vertices'] = Coordinates2D(
            coordinates=sparse_coordinates,
            canvas_shape=Canvas2D(
                height=tf.ones([self._output_height]),
                width=tf.ones([self._output_width]))
            )
        bbox_2d_label_kwargs['frame_id'] = tf.reshape(example[FRAME_ID_KEY], [])
        # Take care of all other possible target features.
        for feature_name in Bbox2DLabel._fields:
            if feature_name in {'vertices', 'frame_id'}:
                continue
            if 'target/' + feature_name in example:
                if feature_name == 'world_bbox_z':
                    sparse_feature_tensor = vector_and_counts_to_sparse_tensor(
                        vector=self._extract_depth(example),
                        counts=counts)
                else:
                    sparse_feature_tensor = vector_and_counts_to_sparse_tensor(
                        vector=example['target/' + feature_name],
                        counts=counts)
            else:
                # TODO(@williamz): Is there a better way to handle optional labels?
                sparse_feature_tensor = []
            bbox_2d_label_kwargs[feature_name] = sparse_feature_tensor

        # Assign source_weight.
        bbox_2d_label_kwargs['source_weight'] = [tf.constant(self.source_weight, tf.float32)]

        bbox_label = Bbox2DLabel(**bbox_2d_label_kwargs)
        # Filter out labels whose dimensions are too small. NOTE: this is mostly for historical
        # reasons (the DefaultDataloader has such a mechanism by default), and due to the fact
        # that labels are actually not enforced to have x2 > x1 and y2 > y1.
        bbox_label = filter_bbox_label_based_on_minimum_dims(
            bbox_2d_label=bbox_label, min_height=1.0, min_width=1.0)

        return bbox_label

    def _scale_coordinates(
        self,
        dense_coordinates,
        height,
        width,
        min_side,
        max_side
    ):
        """Scale coordinates for resize and keep AR."""
        scaling = self._calculate_scale(height, width, min_side, max_side)
        return dense_coordinates * scaling

    def _calculate_scale(self, height, width, min_side, max_side):
        """Calculate the scaling factor for resize and keep aspect ratio."""
        scale_factor = tf.cond(
            tf.less_equal(height, width),
            true_fn=lambda: tf.cast(min_side / height, tf.float32),
            false_fn=lambda: tf.cast(min_side / width, tf.float32)
        )
        # if the scale factor resulting in image's larger side
        # exceed max_side, then calculate scale factor again
        # such that the larger side is scaled to max_side.
        scale_factor2 = tf.cond(
            tf.less_equal(height, width),
            true_fn=lambda: tf.cast(max_side / width, tf.float32),
            false_fn=lambda: tf.cast(max_side / height, tf.float32)
        )
        # take the smaller scale factor, which ensures the scaled image size is
        # no bigger than min_side x max_side
        scale_factor = tf.minimum(scale_factor, scale_factor2)
        return scale_factor

    def _resize_coordinates(
        self,
        dense_coordinates,
        height,
        width,
        target_height,
        target_width
    ):
        """Resize coordinates to target size, do not keep AR."""
        scale_x = tf.cast(target_width / width, tf.float32)
        scale_y = tf.cast(target_height / height, tf.float32)
        scale_xy = tf.reshape(tf.stack([scale_x, scale_y]), (-1, 2))
        return dense_coordinates * scale_xy


# TODO(@williamz): Should the `set_target_size` be upstreamed to TFRecordsDataSource?
class DriveNetTFRecordsDataSource(TFRecordsDataSource):
    """DataSource for reading examples from TFRecords files."""

    def __init__(self, tfrecord_path, image_dir, extension,
                 height, width, channels, subset_size,
                 preprocessing, sample_ratio=1.0,
                 source_weight=1.0, min_side=None,
                 max_side=None, auto_resize=False):
        """Construct a DriveNetTFRecordsDataSource.

        Args:
            tfrecord_path (str): Path, or a list of paths to tfrecords file(s).
            image_dir (str): Path to directory where images referenced by examples are stored.
            extension (str): Extension of image files.
            height (int): Output image height.
            width (int): Output image width.
            channels (int): Number of channels for images stored in this dataset.
            subset_size (int): Number of images from tfrecord_path to use.
            preprocessing (Pipeline): Preprocessing processors specific to this dataset.
            sample_ratio (float): probability at which a sample from this data source is picked
                for inclusion in a batch.
            source_weight (float): Value by which to weight the loss for samples
                coming from this DataSource.
            min_side(int): Minimal side of the image.
            max_side(int): Maximal side of the image.
            auto_resize(bool): Flag to enable automatic resize or not.
        """
        super(DriveNetTFRecordsDataSource, self).__init__(
            tfrecord_path=tfrecord_path,
            image_dir=image_dir,
            extension=extension,
            height=height,
            width=width,
            channels=channels,
            subset_size=subset_size,
            preprocessing=preprocessing,
            sample_ratio=sample_ratio
        )

        self._parser = None
        if self.tfrecord_path:
            self._parser = DriveNetTFRecordsParser(
                tfrecord_path=self.tfrecord_path,
                image_dir=image_dir,
                extension=extension,
                channels=channels,
                source_weight=source_weight,
                auto_resize=auto_resize)

        self.num_samples = sum([sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(filename))
                               for filename in self.tfrecord_path])
        self.min_side = min_side
        self.max_side = max_side
        self.max_image_width, self.max_image_height = self._get_max_image_size()
        # Set the target size for the parser.
        self.set_target_size(height=self.max_image_height,
                             width=self.max_image_width,
                             min_side=self.min_side,
                             max_side=self.max_side)

    @property
    def parse_example(self):
        """Parser for labels in TFRecords used by DriveNet."""
        return lambda dataset: dataset.map(self._parser)

    def set_target_size(self, height, width, min_side=None, max_side=None):
        """Set size for target image .

        Args:
            height (int): Target image height.
            width (int): Target image width.
            min_side(int): Minimal side of the image.
            max_side(int): Maximal side of the image.
        """
        if self._parser:
            self._parser.set_target_size(
                height=height,
                width=width,
                min_side=min_side,
                max_side=max_side
            )

    def _get_max_image_size(self):
        """Scan for the maximum image size of this data source.

        Returns:
            (int) max_image_width, max_image_height.
        """
        max_image_height = 0
        max_image_width = 0
        for path in self.tfrecord_path:
            for record in tf.compat.v1.python_io.tf_record_iterator(path):
                example = tf.train.Example()
                example.ParseFromString(record)
                height = int(str(example.features.feature[HEIGHT_KEY].int64_list.value[0]))
                width = int(str(example.features.feature[WIDTH_KEY].int64_list.value[0]))
                max_image_height = max(max_image_height, height)
                max_image_width = max(max_image_width, width)

        return max_image_width, max_image_height


class DriveNetDataloader(DefaultDataloader):
    """Dataloader for object detection datasets such as KITTI and Cyclops.

    Implements a data loader that reads labels and frame id from datasets and compiles
    image and ground truth tensors used in training and validation.
    """

    def __init__(self,
                 training_data_source_list,
                 image_file_encoding,
                 augmentation_config,
                 validation_data_source_list=None,
                 data_sequence_length_in_frames=None,
                 target_class_mapping=None,
                 auto_resize=False,
                 sampling_mode="user_defined"):
        """Instantiate the dataloader.

        Args:
            training_data_source_list (list): List of DataSourceConfigs specifying training set.
            image_file_encoding (str): How the images to be produced by the dataset are encoded.
                Can be e.g. "jpg", "fp16", "png".
            augmentation_config (dlav.drivenet.common.dataloader.augmentation_config.
                AugmentationConfig): Holds the parameters for augmentation and preprocessing.
            validation_data_source_list (list): List of DataSourceConfigs specifying validation
                set. Can be None.
            data_sequence_length_in_frames (int): Number of frames in each sequence. If not None,
                the output images will be 5D tensors with additional temporal dimension.
            target_class_mapping (dict): source to target class mapper from the ModelConfig proto.
            auto_resize(bool): Flag to enable automatic resize or not.
        """
        super(DriveNetDataloader, self).__init__(
            training_data_source_list=training_data_source_list,
            image_file_encoding=image_file_encoding,
            augmentation_config=augmentation_config,
            validation_data_source_list=validation_data_source_list,
            target_class_mapping=target_class_mapping)

        self._min_image_side = self.augmentation_config.preprocessing.output_image_min
        self._max_image_side = self.augmentation_config.preprocessing.output_image_max
        self._sequence_length_in_frames = data_sequence_length_in_frames
        self.auto_resize = auto_resize

        self.training_sources, self.num_training_samples =\
            self._construct_data_sources(self.training_data_sources)
        if validation_data_source_list is not None:
            self.validation_sources, self.num_validation_samples =\
                self._construct_data_sources(self.validation_data_sources)
        else:
            self.validation_sources = None
            self.num_validation_samples = 0

        # Set up a look up table for class mapping.
        self._target_class_lookup = None
        if self.target_class_mapping is not None:
            self._target_class_lookup = nvidia_tao_tf1.core.processors.LookupTable(
                keys=list(self.target_class_mapping.keys()),
                values=list(self.target_class_mapping.values()),
                default_value=tf.constant(UNKNOWN_CLASS)
            )
        if sampling_mode not in ["user_defined", "proportional", "uniform"]:
            raise NotImplementedError(
                f"Sampling mode: {sampling_mode} requested wasn't implemented."
            )
        self.sampling_mode = sampling_mode
        logger.info(
            "Sampling mode of the dataloader was set to {sample_mode}.".format(
                sample_mode=self.sampling_mode
            )
        )

    def _construct_data_sources(self, data_source_list):
        """Instantiate data sources.

        Args:
            data_source_list (list): List of DataSourceConfigs.

        Returns:
            data_sources (list): A list of DataSource instances.
            num_samples (int): Sum of the number of samples in the above data sources.

        Raises:
            ValueError: If an unknown dataset type was encountered.
        """
        data_sources = []
        for data_source_config in data_source_list:
            if data_source_config.dataset_type == 'tfrecord':
                data_source =\
                    DriveNetTFRecordsDataSource(
                        tfrecord_path=data_source_config.dataset_files,
                        image_dir=data_source_config.images_path,
                        extension='.' + self.image_file_encoding,
                        height=0,
                        width=0,
                        channels=self.num_input_channels,
                        subset_size=0,  # TODO(jrasanen) use this.
                        sample_ratio=1.0,  # TODO(jrasanen) use this.
                        preprocessing=[],
                        source_weight=data_source_config.source_weight,
                        min_side=self._min_image_side,
                        max_side=self._max_image_side,
                        auto_resize=self.auto_resize
                    )
            else:
                raise ValueError("Unknown dataset type \'%s\'" % data_source_config.dataset_type)

            data_sources.append(data_source)

        if self.auto_resize:
            # Use specified target image size in augmentation_config
            self._max_image_height = self.augmentation_config.preprocessing.output_image_height
            self._max_image_width = self.augmentation_config.preprocessing.output_image_width
        else:
            # Scan through all data sources and compute the maximum image size. Needed so that we
            # can pad all images to the same size for minibatching.
            max_image_height = 0
            max_image_width = 0
            for data_source in data_sources:
                max_image_height = max(data_source.max_image_height, max_image_height)
                max_image_width = max(data_source.max_image_width, max_image_width)

            max_image_height = max(
                max_image_height, self.augmentation_config.preprocessing.output_image_height)
            max_image_width = max(
                max_image_width, self.augmentation_config.preprocessing.output_image_width)

            self._max_image_height = max_image_height
            self._max_image_width = max_image_width

        num_samples = 0
        for data_source in data_sources:
            # TODO(@williamz): There should be some API at the DataSource ABC level to allow
            # these "batchability" mechanics.
            if isinstance(data_source, DriveNetTFRecordsDataSource):
                data_source.set_target_size(
                    height=self._max_image_height,
                    width=self._max_image_width,
                    min_side=self._min_image_side,
                    max_side=self._max_image_side
                )

            source_samples = len(data_source)
            num_samples += source_samples

            # This is to be consistent with the DefaultDataloader's concatenation behavior.
            # Note that it doesn't functionally reproduce concatenating multiple sources into one,
            # but statistically should lead to the samples being seen the same amount of times.
            data_source.sample_ratio = source_samples

        return data_sources, num_samples

    def get_num_samples(self, training):
        """Get number of dataset samples.

        Args:
            training (bool): Get number of samples in the training (true) or
                validation (false) set.

        Returns:
            Number of samples in the chosen set.
        """
        if training:
            return self.num_training_samples
        return self.num_validation_samples

    def get_dataset_tensors(self, batch_size, training, enable_augmentation, repeat=True):
        """Get input images and ground truth labels as tensors for training and validation.

        Args:
            batch_size (int): Minibatch size.
            training (bool): Get samples from the training (True) or validation (False) set.
            enable_augmentation (bool): Whether to augment input images and labels.
            repeat (bool): Whether the dataset can be looped over multiple times or only once.

        Returns:
            images (Tensor of shape (batch, channels, height, width)): Input images with values
                in the [0, 1] range.
            labels (Bbox2DLabel): Contains labels corresponding to ``images``.
            num_samples (int): Total number of samples found in the dataset.
        """
        # TODO(jrasanen) Need to support repeat in dlav/common data loader? Looks like we
        # currently have repeat=True everywhere, so could actually remove the arg.
        assert repeat is True
        data_sources = self.training_sources if training else self.validation_sources
        # Construct data source independent augmentation pipeline.
        if self._min_image_side == 0:
            augmentation_pipeline = _get_augmentation_pipeline(
                augmentation_config=self.augmentation_config,
                max_image_height=self._max_image_height,
                max_image_width=self._max_image_width,
                enable_augmentation=enable_augmentation,
            )

        preprocessors = []

        if training:
            num_gpus = distribution.get_distributor().size()
            local_gpu = distribution.get_distributor().rank()
        else:
            # We want data to be unsharded during evaluation because currently only single-GPU
            # evaluation is enabled.
            num_gpus = 1
            local_gpu = 0
        data_loader = DataLoader(data_sources=data_sources,
                                 augmentation_pipeline=[],
                                 batch_size=batch_size * num_gpus,
                                 shuffle=training,
                                 sampling=self.sampling_mode,
                                 preprocessing=preprocessors,
                                 pipeline_dtype=tf.float16)  # Use fp16 image processing.
        data_loader.set_shard(shard_count=num_gpus, shard_index=local_gpu)

        # Instantiate the data loader pipeline.
        sequence_example = data_loader()

        if self._min_image_side == 0:
            # Compute augmentation transform matrices.
            # TODO(@williamz/@jrasanen): Can this also be moved back up to the `DataLoader`?
            transformed_example = augmentation_pipeline(sequence_example)

            # Apply augmentations and cast to model dtype.
            sequence_example = transformed_example(output_image_dtype=K.floatx())

            # Since TransformedExample only lazily captures augmentations but does not apply them,
            # the BboxClipper processor has to be applied outside of the augmentation_pipeline,
            # as it expects to deal with transformed labels. Hence, it is not included in the above
            # ``augmentation_pipeline``.
            # DriveNet quirks: update truncation_type, throw out labels outside the crop,
            # and clip the coordinates of those that are partially outside.
            bbox_clipper = BboxClipper(
                crop_left=self.augmentation_config.preprocessing.crop_left,
                crop_top=self.augmentation_config.preprocessing.crop_top,
                crop_right=self.augmentation_config.preprocessing.crop_right,
                crop_bottom=self.augmentation_config.preprocessing.crop_bottom)

            sequence_example = bbox_clipper.process(sequence_example)

        images = sequence_example.instances[FEATURE_CAMERA].images
        if images.dtype != tf.float32:
            images = tf.cast(images, dtype=tf.float32)

        labels = sequence_example.labels[LABEL_OBJECT]

        if self._sequence_length_in_frames is None:
            images = images[:, 0, ...]

        if self.target_class_mapping is not None:
            labels = self._map_to_model_target_classes(labels)

        return images, labels, len(data_loader)

    def _map_to_model_target_classes(self, labels):
        """Map object classes in the data source to the target classes in the dataset_config.

        Args:
            labels(BBox2DLabel): Input data label.

        Returns:
            filterred_labels (Bbox2DLabel): Output labels with mapped class names.
        """
        source_classes = labels.object_class
        mapped_classes = tf.SparseTensor(
            values=self._target_class_lookup(source_classes.values),
            indices=source_classes.indices,
            dense_shape=source_classes.dense_shape)
        mapped_labels = labels._replace(object_class=mapped_classes)
        valid_indices = tf.not_equal(mapped_classes.values, UNKNOWN_CLASS)
        return filter_labels(mapped_labels, valid_indices)


def _get_augmentation_pipeline(
    augmentation_config,
    max_image_height,
    max_image_width,
    enable_augmentation=False,
):
    """Define an augmentation (+preprocessing) pipeline.

    Args:
        augmentation_config (AugmentationConfig)
        max_image_height (int)
        max_image_width (int)
        enable_augmentation (bool): Whether to enable augmentation or not.

    Returns:
        pipeline (Pipeline): Augmentation pipeline.
    """
    # Note: once / if we are able to move to a common spec / builder, this should be removed.
    processors = []
    # So our lines aren't too long.
    spatial_config = augmentation_config.spatial_augmentation
    color_config = augmentation_config.color_augmentation
    num_channels = augmentation_config.preprocessing.output_image_channel
    # Preprocessing: scaling and cropping.
    # First: scaling (e.g. downscale 0.5 for side camera models).
    scale_width = augmentation_config.preprocessing.scale_width
    scale_height = augmentation_config.preprocessing.scale_height
    if scale_width != 0. or scale_height != 0.:
        if scale_height != 0.:
            scaled_height = scale_height * max_image_height
        else:
            scaled_height = max_image_height
        if scale_width != 0.:
            scaled_width = scale_width * max_image_width
        else:
            scaled_width = max_image_width
        processors.append(Scale(height=scaled_height, width=scaled_width))
    # Then: cropping. Note that we're adding Crop unconditionally so that we're guaranteed to
    # end up with a non-empty pipeline.
    crop_left, crop_top, crop_right, crop_bottom = \
        augmentation_config.preprocessing.crop_left, \
        augmentation_config.preprocessing.crop_top, \
        augmentation_config.preprocessing.crop_right, \
        augmentation_config.preprocessing.crop_bottom
    if {crop_left, crop_top, crop_right, crop_bottom} == {0}:
        crop_left = 0
        crop_right = augmentation_config.preprocessing.output_image_width
        crop_top = 0
        crop_bottom = augmentation_config.preprocessing.output_image_height
    processors.append(Crop(
        left=crop_left,
        top=crop_top,
        right=crop_right,
        bottom=crop_bottom))

    # Spatial and color augmentation.
    if enable_augmentation:
        processors.append(RandomFlip(horizontal_probability=spatial_config.hflip_probability))
        processors.append(RandomTranslation(
            max_x=int(spatial_config.translate_max_x),
            max_y=int(spatial_config.translate_max_y)))
        processors.append(RandomZoom(
            ratio_min=spatial_config.zoom_min,
            ratio_max=spatial_config.zoom_max,
            probability=1.0))
        processors.append(RandomRotation(
            min_angle=-spatial_config.rotate_rad_max,
            max_angle=spatial_config.rotate_rad_max,
            probability=spatial_config.rotate_probability
        ))
        # Color augmentation.
        if num_channels == 3:
            processors.append(RandomBrightness(
                scale_max=color_config.color_shift_stddev * 2.0,
                uniform_across_channels=True))
            processors.append(RandomHueSaturation(
                hue_rotation_max=color_config.hue_rotation_max,
                saturation_shift_max=color_config.saturation_shift_max))
            processors.append(RandomContrast(
                scale_max=color_config.contrast_scale_max,
                center=color_config.contrast_center))

    augmentation_pipeline = Pipeline(
        processors=processors,
        input_data_format=CHANNELS_FIRST,
        output_data_format=CHANNELS_FIRST)

    return augmentation_pipeline
