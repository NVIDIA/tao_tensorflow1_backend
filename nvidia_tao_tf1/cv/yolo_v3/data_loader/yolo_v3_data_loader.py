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

"""YOLO v3 data loader."""

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_loader import DataLoaderYOLOv3
from nvidia_tao_tf1.blocks.multi_source_loader.processors import (
    TemporalBatcher,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources import (
    TFRecordsDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Bbox2DLabel,
    Coordinates2D,
    FEATURE_CAMERA,
    filter_bbox_label_based_on_minimum_dims,
    LabelledImages2DReference,
    LabelledImages2DReferenceVal,
    SequenceExample,
    set_augmentations,
    set_augmentations_val,
    set_h_tensor,
    set_h_tensor_val,
    set_image_channels,
    set_image_depth,
    set_max_side,
    set_min_side,
    set_w_tensor,
    set_w_tensor_val,
    sparsify_dense_coordinates,
    vector_and_counts_to_sparse_tensor,
)
import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import _pattern_to_files
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.data_source_config import DataSourceConfig
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import DefaultDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import FRAME_ID_KEY
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import HEIGHT_KEY
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import UNKNOWN_CLASS
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import WIDTH_KEY
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import (
    extract_tfrecords_features,
    get_absolute_data_path,
)
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import filter_labels

from nvidia_tao_tf1.cv.yolo_v3.data_loader.augmentation import (
    apply_letterbox_resize,
    inner_augmentations
)

Canvas2D = tao_core.types.Canvas2D
BW_POLY_COEFF1_60FC = 0.000545421498827636
FOLD_STRING = "fold-{:03d}-of-"


class YOLOv3TFRecordsParser(object):
    """Parse tf.train.Example protos into YOLO v3 Examples."""

    def __init__(
        self, tfrecord_path, image_dir, extension,
        channels, depth, source_weight, h_tensor, w_tensor,
        augmentations, target_class_mapping,
        class_to_idx_mapping, training
    ):
        """Construct a parser for YOLO v3 labels.

        Args:
            tfrecord_path (list): List of paths to tfrecords file.
            image_dir (str): Path to the directory where images are contained.
            extension (str): Extension for images that get loaded (
                ".fp16", ".png", ".jpg" or ".jpeg").
            channels (int): Number of channels in each image.
            depth(int): Depth of image(8 or 16).
            h_tensor(Tensor): Image height tensor.
            w_tensor(Tensor): Image width tensor.
            augmentations(List): List of augmentations.

        Raises:
            ValueError: If the number of input channels is not unsupported (i.e. must be equal to 3)
        """
        if channels not in [1, 3]:
            raise ValueError("YOLOv3TFRecordsParser: unsupported number of channels %d." %
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
        if depth == 16 and extension in ["JPG", "JPEG", "jpg", "jpeg"]:
            raise ValueError(
                f"Only PNG(png) images can support 16-bit depth, got {extension}"
            )
        self._image_depth = depth
        set_image_depth(self._image_depth)
        self.training = training
        self._h_tensor = h_tensor
        if self.training:
            set_h_tensor(h_tensor)
        else:
            set_h_tensor_val(h_tensor)
        self._w_tensor = w_tensor
        if self.training:
            set_w_tensor(w_tensor)
        else:
            set_w_tensor_val(w_tensor)
        self._augmentations = augmentations
        if self.training:
            set_augmentations(augmentations)
        else:
            set_augmentations_val(augmentations)
        self.target_class_mapping = target_class_mapping
        self.class_to_idx_mapping = class_to_idx_mapping

        self._image_dir = image_dir
        if not self._image_dir.endswith('/'):
            self._image_dir += '/'

        self._tfrecord_path = tfrecord_path
        # Delay the actual definition to call time.
        self._parse_example = None

        # Set the source_weight.
        self.source_weight = source_weight

    def _get_parse_example(self):
        if self._parse_example is None:
            extracted_features = extract_tfrecords_features(self._tfrecord_path[0])

            self._parse_example = \
                tao_core.processors.ParseExampleProto(features=extracted_features,
                                                     single=True)
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
        if self.training:
            x = SequenceExample(
                instances={
                    FEATURE_CAMERA: LabelledImages2DReference(
                        path=image_path,
                        extension=extension,
                        canvas_shape=Canvas2D(
                            height=tf.ones([self._output_height]),
                            width=tf.ones([self._output_width])),
                        input_height=height,
                        input_width=width,
                        labels=labels
                    ),
                },
                labels=[]
            )
        else:
            x = SequenceExample(
                instances={
                    FEATURE_CAMERA: LabelledImages2DReferenceVal(
                        path=image_path,
                        extension=extension,
                        canvas_shape=Canvas2D(
                            height=tf.ones([self._output_height]),
                            width=tf.ones([self._output_width])),
                        input_height=height,
                        input_width=width,
                        labels=labels
                    ),
                },
                labels=[]
            )
        return x

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

    def _resize_coordinates(
        self,
        dense_coordinates,
        height,
        width,
        target_height,
        target_width
    ):
        """Resize coordinates to target size."""
        scale_x = tf.cast(target_width / width, tf.float32)
        scale_y = tf.cast(target_height / height, tf.float32)
        scale_xy = tf.reshape(tf.stack([scale_x, scale_y]), (-1, 2))
        return dense_coordinates * scale_xy


class YOLOv3TFRecordsDataSource(TFRecordsDataSource):
    """DataSource for reading examples from TFRecords files."""

    def __init__(self, tfrecord_path, image_dir, extension,
                 height, width, channels, depth, subset_size,
                 preprocessing, sample_ratio=1.0,
                 source_weight=1.0, min_side=None,
                 max_side=None, h_tensor=None,
                 w_tensor=None, augmentations=None,
                 target_class_mapping=None,
                 class_to_idx_mapping=None,
                 training=True):
        """Construct a YOLOv3TFRecordsDataSource.

        Args:
            tfrecord_path (str): Path, or a list of paths to tfrecords file(s).
            image_dir (str): Path to directory where images referenced by examples are stored.
            extension (str): Extension of image files.
            height (int): Output image height.
            width (int): Output image width.
            channels (int): Number of channels for images stored in this dataset.
            depth(int): Image depth of bits per pixel per channel.
            subset_size (int): Number of images from tfrecord_path to use.
            preprocessing (Pipeline): Preprocessing processors specific to this dataset.
            sample_ratio (float): probability at which a sample from this data source is picked
                for inclusion in a batch.
            source_weight (float): Value by which to weight the loss for samples
                coming from this DataSource.
            min_side(int): Minimal side of the image.
            max_side(int): Maximal side of the image.
            h_tensor(Tensor): Image height tensor.
            w_tensor(Tensor): Image width tensor.
        """
        super(YOLOv3TFRecordsDataSource, self).__init__(
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
            assert depth in [8, 16], (
                f"Image depth can only support 8 or 16 bits, got {depth}"
            )
            self._parser = YOLOv3TFRecordsParser(
                tfrecord_path=self.tfrecord_path,
                image_dir=image_dir,
                extension=extension,
                channels=channels,
                depth=depth,
                source_weight=source_weight,
                h_tensor=h_tensor,
                w_tensor=w_tensor,
                augmentations=augmentations,
                target_class_mapping=target_class_mapping,
                class_to_idx_mapping=class_to_idx_mapping,
                training=training)

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
        """Parser for labels in TFRecords used by YOLOv3."""
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


class YOLOv3DataLoader:
    """YOLOv3DataLoader for object detection datasets such as KITTI and Cyclops.

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
                 h_tensor=None,
                 w_tensor=None,
                 training=False
                 ):
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
            h_tensor(Tensor): Image height tensor.
            w_tensor(Tensor): Image width tensor.
        """
        self.image_file_encoding = image_file_encoding
        self.augmentation_config = augmentation_config
        self.target_class_mapping = target_class_mapping
        # Get training data sources.
        self.training_data_sources = training_data_source_list
        # Now, potentially, get the validation data sources.
        self.validation_data_sources = validation_data_source_list
        self._h_tensor = h_tensor
        self._w_tensor = w_tensor
        self._sequence_length_in_frames = data_sequence_length_in_frames
        self._training = training
        self._augmentations = self._build_yolov3_augmentations_pipeline(
            augmentation_config
        )
        self.num_input_channels = augmentation_config.output_channel
        self.image_depth = int(augmentation_config.output_depth) or 8
        # Set up a look up table for class mapping.
        self._target_class_lookup = None
        if self.target_class_mapping is not None:
            self._target_class_lookup = tao_core.processors.LookupTable(
                keys=list(self.target_class_mapping.keys()),
                values=list(self.target_class_mapping.values()),
                default_value=tf.constant(UNKNOWN_CLASS)
            )
            self._class_idx_lookup = tao_core.processors.LookupTable(
                keys=sorted(list(self.target_class_mapping.values())),
                values=list(range(len(list(self.target_class_mapping.values())))),
                default_value=-1
            )
        self.training_sources, self.num_training_samples =\
            self._construct_data_sources(self.training_data_sources)
        if validation_data_source_list is not None:
            self.validation_sources, self.num_validation_samples =\
                self._construct_data_sources(self.validation_data_sources)
        else:
            self.validation_sources = None
            self.num_validation_samples = 0

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
                    YOLOv3TFRecordsDataSource(
                        tfrecord_path=data_source_config.dataset_files,
                        image_dir=data_source_config.images_path,
                        extension='.' + self.image_file_encoding,
                        height=0,
                        width=0,
                        channels=self.num_input_channels,
                        depth=self.image_depth,
                        subset_size=0,  # TODO(jrasanen) use this.
                        sample_ratio=1.0,  # TODO(jrasanen) use this.
                        preprocessing=[],
                        source_weight=data_source_config.source_weight,
                        min_side=0,
                        max_side=0,
                        h_tensor=self._h_tensor,
                        w_tensor=self._w_tensor,
                        augmentations=self._augmentations,
                        target_class_mapping=self._target_class_lookup,
                        class_to_idx_mapping=self._class_idx_lookup,
                        training=self._training
                    )
            else:
                raise ValueError("Unknown dataset type \'%s\'" % data_source_config.dataset_type)

            data_sources.append(data_source)

        # Scan through all data sources and compute the maximum image size. Needed so that we
        # can pad all images to the same size for minibatching.
        max_image_height = 0
        max_image_width = 0
        for data_source in data_sources:
            max_image_height = max(data_source.max_image_height, max_image_height)
            max_image_width = max(data_source.max_image_width, max_image_width)

        self._max_image_height = max_image_height
        self._max_image_width = max_image_width

        num_samples = 0
        for data_source in data_sources:
            data_source.set_target_size(
                height=self._max_image_height,
                width=self._max_image_width,
                min_side=0,
                max_side=0
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

    def get_dataset_tensors(self, batch_size, repeat=True):
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
        data_sources = self.training_sources if self._training else self.validation_sources
        if self._sequence_length_in_frames is not None:
            preprocessors = [TemporalBatcher(size=self._sequence_length_in_frames)]
        else:
            preprocessors = []

        if self._training:
            num_gpus = distribution.get_distributor().size()
            local_gpu = distribution.get_distributor().rank()
        else:
            # We want data to be unsharded during evaluation because currently only single-GPU
            # evaluation is enabled.
            num_gpus = 1
            local_gpu = 0
        data_loader = DataLoaderYOLOv3(
            data_sources=data_sources,
            augmentation_pipeline=[],
            batch_size=batch_size * num_gpus,
            shuffle=self._training,
            preprocessing=preprocessors,
            # This doesn't matter as we forced it to float32 in modulus
            pipeline_dtype=tf.uint8
        )
        data_loader.set_shard(shard_count=num_gpus, shard_index=local_gpu)

        # Instantiate the data loader pipeline.
        sequence_example = data_loader()
        images = sequence_example.instances[FEATURE_CAMERA].images
        labels = sequence_example.instances[FEATURE_CAMERA].labels

        if self._sequence_length_in_frames is None:
            images = images[:, 0, ...]

        if self.target_class_mapping is not None:
            labels = self._map_to_model_target_classes(labels)

        shapes = sequence_example.instances[FEATURE_CAMERA].shapes
        shapes = tf.reshape(shapes, (-1, 2))
        return images, labels, shapes, len(data_loader)

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

    def _build_yolov3_augmentations_pipeline(self, yolov3_augmentation_config):

        def _augmentations_list(image, labels, ratio, xmax):
            return inner_augmentations(image, labels, ratio, xmax, yolov3_augmentation_config)

        if self._training:
            return _augmentations_list
        return apply_letterbox_resize


def build_data_source_lists(dataset_proto):
    """Build training and validation data source lists from proto.

    Args:
        dataset_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config proto message)

    Returns:
        training_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for training.
        validation_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for validation. Can be None.
        validation_fold (int): Validation fold number (0-based). Indicates which fold from the
            training data to use as validation. Can be None.
    """
    # Determine how we are getting validation data sources.
    if len(dataset_proto.validation_data_sources) > 0:
        dataset_split_type = "validation_data_sources"
    else:
        dataset_split_type = "validation_fold"
    training_data_source_list = []
    validation_data_source_list = []
    validation_fold = None
    if dataset_proto.data_sources[0].WhichOneof("labels_format") == "tfrecords_path":
        for data_source_proto in dataset_proto.data_sources:
            source_weight = data_source_proto.source_weight
            images_path = get_absolute_data_path(
                str(data_source_proto.image_directory_path)
            )
            tfrecords_path = str(data_source_proto.tfrecords_path)
            tfrecords_files = _pattern_to_files(tfrecords_path)

            # Filter out files based on validation fold only if validation fold specified.
            if dataset_split_type == "validation_fold":
                # Defining the fold number for the glob pattern.
                fold_identifier = FOLD_STRING.format(dataset_proto.validation_fold)
                validation_fold = dataset_proto.validation_fold

                # Take all .tfrecords files except the one matching the validation fold.
                training_tfrecords_files = [filename for filename in tfrecords_files
                                            if fold_identifier not in filename]

                # Take only the file matching the validation fold.
                validation_tfrecords_files = [filename for filename in tfrecords_files
                                              if fold_identifier in filename]

                validation_data_source_list.append(DataSourceConfig(
                    dataset_type='tfrecord',
                    dataset_files=validation_tfrecords_files,
                    images_path=images_path,
                    export_format=None,
                    split_db_path=None,
                    split_tags=None,
                    source_weight=source_weight))
            else:
                training_tfrecords_files = tfrecords_files

            training_data_source_list.append(DataSourceConfig(
                dataset_type='tfrecord',
                dataset_files=training_tfrecords_files,
                images_path=images_path,
                export_format=None,
                split_db_path=None,
                split_tags=None,
                source_weight=source_weight))

    # Get validation data sources, if available.
    if dataset_split_type == "validation_data_sources":
        for data_source_proto in dataset_proto.validation_data_sources:
            source_weight = data_source_proto.source_weight
            images_path = get_absolute_data_path(
                str(data_source_proto.image_directory_path)
            )
            tfrecords_path = str(data_source_proto.tfrecords_path)
            tfrecords_files = _pattern_to_files(tfrecords_path)
            validation_data_source_list.append(DataSourceConfig(
                dataset_type='tfrecord',
                dataset_files=tfrecords_files,
                images_path=images_path,
                export_format=None,
                split_db_path=None,
                split_tags=None,
                source_weight=source_weight)
            )

    return training_data_source_list, validation_data_source_list, validation_fold


def build_dataloader(
    dataset_proto,
    augmentation_proto,
    h_tensor,
    w_tensor,
    training
):
    """Build a Dataloader from a proto.

    Args:
        dataset_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config.DatasetConfig)
        augmentation_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.augmentation_config.
            AugmentationConfig)

    Returns:
        dataloader (nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader.DefaultDataloader).
    """
    # Now, get the class mapping.
    dataset_config = dataset_proto
    source_to_target_class_mapping = dict(dataset_config.target_class_mapping)

    # Image file encoding.
    image_file_encoding = dataset_config.image_extension

    # Get the data source lists.
    training_data_source_list, validation_data_source_list, _ = \
        build_data_source_lists(dataset_config)
    if training:
        validation_data_source_list = []
    else:
        training_data_source_list = []
    dataloader_kwargs = dict(
        training_data_source_list=training_data_source_list,
        image_file_encoding=image_file_encoding,
        augmentation_config=augmentation_proto,
        validation_data_source_list=validation_data_source_list,
        target_class_mapping=source_to_target_class_mapping,
        h_tensor=h_tensor,
        w_tensor=w_tensor,
        training=training
    )
    return YOLOv3DataLoader(**dataloader_kwargs)
