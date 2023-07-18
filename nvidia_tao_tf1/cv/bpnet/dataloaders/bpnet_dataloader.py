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

"""Dataloader for BpNet datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, namedtuple
import logging
import os

import tensorflow as tf

from nvidia_tao_tf1.blocks.dataloader import DataLoader
import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core.processors import SpatialTransform
from nvidia_tao_tf1.cv.bpnet.dataloaders.dataset_config import DatasetConfig
from nvidia_tao_tf1.cv.bpnet.dataloaders.processors.augmentation import BpNetSpatialTransformer
from nvidia_tao_tf1.cv.bpnet.dataloaders.processors.label_processor import LabelProcessor

logger = logging.getLogger(__name__)

BpData = namedtuple('BpData', ['images', 'masks', 'labels'])


class BpNetDataloader(DataLoader):
    """Dataloader Class for BpNet.

    The dataloader parses the given TFRecord files and processes the inputs
    and ground truth labels to be used for training and validation.
    """

    TRAIN = "train"
    VAL = "val"
    SUPPORTED_MODES = [TRAIN, VAL]
    ITERATOR_INIT_OP_NAME = "iterator_init"

    @tao_core.coreobject.save_args
    def __init__(self,
                 batch_size,
                 pose_config,
                 image_config,
                 dataset_config,
                 augmentation_config,
                 label_processor_config,
                 normalization_params,
                 shuffle_buffer_size=None,
                 **kwargs):
        """Init function for the dataloader.

        Args:
            batch_size (int): Size of minibatch.
            pose_config (BpNetPoseConfig)
            image_config (dict): Basic information of input images.
            dataset_config (dict): Basic information of datasets used.
            augmentation_config (AugmentationConfig): Parameters used for input augmentations.
            label_processor_config (dict): Parameters used for transforming kpts
                to final label tensors consumable by the training pipeline.
            normalization_params (dict): Parameter values to be used for input normalization
            shuffle_buffer_size (int): Size of the shuffle buffer for feeding in data.
                If None, it will be 20000. Default None
        """
        super(BpNetDataloader, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.image_dims = image_config['image_dims']
        self.image_encoding = image_config['image_encoding']

        # Dataset config
        self._root_data_path = dataset_config['root_data_path']
        self._train_records_folder_path = dataset_config['train_records_folder_path']
        self._train_records_path = dataset_config['train_records_path']
        self._val_records_folder_path = dataset_config['val_records_folder_path']
        self._val_records_path = dataset_config['val_records_path']
        self.dataset2spec_map = dataset_config['dataset_specs']

        # Get the relative paths for the tfrecords filelist
        self.tfrecords_filename_list = [os.path.join(
            self._train_records_folder_path, train_record_path)
            for train_record_path in self._train_records_path]

        # If shuffle buffer size is not provided, set default.
        if shuffle_buffer_size is None:
            # Currently set to 20% of COCO dataset (after key person centric
            # mode in dataio)
            shuffle_buffer_size = 20000

        # Normalization parameters
        self._normalization_params = {}
        self._normalization_params["image_scale"] = tf.constant(
            normalization_params["image_scale"])
        self._normalization_params["image_offset"] = tf.constant(
            normalization_params["image_offset"])
        self._normalization_params["mask_scale"] = tf.constant(
            normalization_params["mask_scale"])
        self._normalization_params["mask_offset"] = tf.constant(
            normalization_params["mask_offset"])

        # TFRecords Iterator Attributes
        # NOTE: This will no longer needed after swithing to the parallel pptimized loader.
        # But the issue remains that the images are of varying sizes and hence tricky to
        # stack together.
        #
        # TODO: One option is to have augmentations and resizing before the the batching and after
        # load and decode. This ensures that all the images and mask are of same size.
        #
        # For Example:
        # dataset = dataset.map(load_image) // load single image
        # dataset = dataset.map(augment_image) // augment and resize image
        # dataset = dataset.batch(batch_size) // Batch
        self._tfrecords_iterator_attributes = {
            "batch_size": self.batch_size,
            "batch_as_list": True,
            "shuffle": True,
            "shuffle_buffer_size": shuffle_buffer_size,
            "repeat": True
        }

        # Get the file loader object
        self._load_file = tao_core.processors.LoadFile(
            prefix=self._root_data_path)

        # Get the image decoder object
        # TODO: Encoding type might vary with datasets. How to choose?
        # Figure out encoding based on datasets and switch decoders
        # accordingly
        self._decode_image, self._decode_mask = self._image_decoder()

        # Get the image shape
        self.image_shape = [self.image_dims['height'],
                            self.image_dims['width'],
                            self.image_dims['channels']]

        # Get the proto parser.
        self._proto_parser = self._tfrecord_parser()

        # Initialize the Dataset parsers for each supported dataset.
        self.pose_config = pose_config
        self.dataset_cfgs = {}
        for dataset in self.dataset2spec_map:
            self.dataset_cfgs[dataset] = DatasetConfig(
                self.dataset2spec_map[dataset], pose_config)

        # Intialize Label Processor
        self.label_processor = LabelProcessor(
            pose_config=pose_config,
            image_shape=self.image_shape,
            target_shape=pose_config.label_tensor_shape,
            **label_processor_config
        )

        # Initialize Spatial Transformer
        self.augmentation_config = augmentation_config

        # NOTE: This is currently used for obtaining the spatial transformer
        # matrix and to transform the keypoints. Images and masks are transformed
        # using the modulus spatial transformer on the GPU
        self.bpnet_spatial_transformer = BpNetSpatialTransformer(
            aug_params=augmentation_config.spatial_aug_params,
            identity_aug_params=augmentation_config.identity_spatial_aug_params,
            image_shape=self.image_shape,
            pose_config=pose_config,
            augmentation_mode=augmentation_config.spatial_augmentation_mode)
        # To transform images and masks
        self.modulus_spatial_transformer = SpatialTransform(
            method='bilinear',
            data_format="channels_last"
        )

        # TODO: Enable when training and disable during validation
        self.enable_augmentation = tf.constant(True)

    def __call__(self):
        """Get input images and ground truth labels as tensors for training and validation.

        Returns:
            A BpData namedtuple which contains the following tensors in 'NCHW' format:
            1. images (4D tensor): model input, images.
            2. masks (4D tensor): regions to mask out for training/validation loss.
            3. labels (4D tensor): gaussian peaks and part affinity fields representing
                    the human skeleton ground truth.
        """

        # Load and encode tfrecords.
        self._tfrecord_iterator = self._get_tfrecords_iterator()

        masks = []
        images = []
        labels = []

        # Iterate through each record in the batch
        for record in self._tfrecord_iterator():

            # Parse the tfrecord
            example = self._proto_parser(record)
            # Load and decode images, masks and labels.
            # This also converts the kpts from dataset format to bpnet format
            image, mask, joints, meta_data = self._load_and_decode(example)
            # Obtain the transformation matrix for desired augmentation
            #  and apply transformation to the keypoints.
            # TODO: Move keypoint transformation and flipping outside numpy_function.
            # to be done on GPU.
            joints, stm = tf.numpy_function(
                self.bpnet_spatial_transformer.call,
                [image, joints, meta_data['scales'],
                    meta_data['centers'], self.enable_augmentation],
                [tf.float64, tf.float32]
            )

            # Apply spatial transformations to the image
            image_crop_size = (self.image_shape[0], self.image_shape[1])
            image = self._apply_augmentations_to_image(
                image, stm, self.image_shape, crop_size=image_crop_size, background_value=127.0)

            # Apply spatial transformations to the mask
            mask_target_shape = [self.image_shape[0], self.image_shape[1], 1]
            # Mask is the same size as image. Transformation happens in image space
            # and it is cropped to network input shape similar to image
            mask_crop_size = (self.image_shape[0], self.image_shape[1])
            mask = self._apply_augmentations_to_image(
                mask, stm, mask_target_shape, crop_size=mask_crop_size, background_value=255.0)
            # Resize mask to target label shape
            mask = tf.image.resize_images(
                mask,
                size=(
                    self.pose_config.label_tensor_shape[0],
                    self.pose_config.label_tensor_shape[1]),
                method=tf.image.ResizeMethod.BILINEAR,
            )

            # Normalize image
            image = tf.compat.v1.math.divide(
                image, self._normalization_params["image_scale"])
            image = tf.compat.v1.math.subtract(
                image, self._normalization_params["image_offset"])
            # Normalize mask
            mask = tf.compat.v1.math.divide(
                mask, self._normalization_params["mask_scale"])
            mask = tf.compat.v1.math.subtract(
                mask, self._normalization_params["mask_offset"])

            # Repeat mask to match the number of channels in the labels tensor
            mask = tf.tile(
                mask, [1, 1, self.pose_config.label_tensor_shape[-1]])

            # Tranform the keypoints using the LabelProcessor
            # Converts them to heatmaps and part affinity fields
            label = self._transform_labels(joints)

            images.append(image)
            masks.append(mask)
            labels.append(label)

        images = tf.stack(images)
        masks = tf.stack(masks)
        labels = tf.stack(labels)

        # Set the shapes of the final tensors used for training.
        # NOTE: This is needed because of `tf.numpy_function`. TensorShape is unknown for tensors
        # computed within the `tf.numpy_function`.
        images.set_shape((
            self.batch_size,
            self.image_shape[0],
            self.image_shape[1],
            self.image_shape[2]
        ))
        masks.set_shape((
            self.batch_size,
            self.pose_config.label_tensor_shape[0],
            self.pose_config.label_tensor_shape[1],
            self.pose_config.label_tensor_shape[2]
        ))
        labels.set_shape((
            self.batch_size,
            self.pose_config.label_tensor_shape[0],
            self.pose_config.label_tensor_shape[1],
            self.pose_config.label_tensor_shape[2]
        ))

        # Cast the tensors to tf.float32 (Is this required here?)
        images = tf.cast(images, tf.float32)
        masks = tf.cast(masks, tf.float32)
        labels = tf.cast(labels, tf.float32)

        return BpData(images, masks, labels)

    def _apply_augmentations_to_image(
            self,
            input_tensor,
            stm,
            target_shape,
            crop_size=None,
            background_value=0.0):
        """
        Apply spatial and color transformations to an image.

        Spatial transform op maps destination image pixel P into source image location Q
        by matrix M: Q = P M. Here we first compute a forward mapping Q M^-1 = P, and
        finally invert the matrix.

        Args:
            input_tensor (Tensor): Input image frame tensors (HWC).
            sm (Tensor): 3x3 spatial transformation/augmentation matrix.
            target_shape (list): output shape of the augmented tensor
            crop_size (tuple): (height, width) of the crop area.
                It crops the region: top-left (0, 0) to bottom-right (h, w)
            background_value (float): The value the background canvas should have.

        Returns:
            image_augmented (Tensor, HWC): Augmented input tensor.
        """
        # Convert image to float if needed (stm_op requirement).
        if input_tensor.dtype != tf.float32:
            input_tensor = tf.cast(input_tensor, tf.float32)
        if stm.dtype != tf.float32:
            stm = tf.cast(stm, tf.float32)

        dm = tf.matrix_inverse(stm)
        # update background value
        self.modulus_spatial_transformer.background_value = background_value
        # Apply spatial transformations.
        # NOTE: Image and matrix need to be reshaped into a batch of one for
        # this op.
        image_augmented = self.modulus_spatial_transformer(
            images=tf.stack(
                [input_tensor]), stms=tf.stack(
                [dm]), shape=crop_size)[
            0, ...]

        image_augmented.set_shape(target_shape)

        return image_augmented

    def _get_tfrecords_iterator(self):
        """Get TFRecordsIterator for a given set of TFRecord files.

        Returns:
            tfrecords_iterator (tao_core.processors.TFRecordsIterator)
        """

        # Check validity of each tfrecord file.
        for filename in self.tfrecords_filename_list:
            assert tf.data.TFRecordDataset(filename), \
                ('Expects each file to be valid!', filename)

        # Print number of files
        num_samples = 0
        for filename in self.tfrecords_filename_list:
            num_samples_set = sum(
                1 for _ in tf.python_io.tf_record_iterator(filename))
            num_samples += num_samples_set
            print(filename + ': ' + str(num_samples_set))
        print("Total Samples: {}".format(num_samples))

        # Load and set up modulus TFRecordIterator Processor.
        tfrecords_iterator = tao_core.processors.TFRecordsIterator(
            file_list=self.tfrecords_filename_list,
            batch_size=self.batch_size,
            shuffle_buffer_size=self._tfrecords_iterator_attributes['shuffle_buffer_size'],
            shuffle=self._tfrecords_iterator_attributes['shuffle'],
            repeat=self._tfrecords_iterator_attributes['repeat'],
            batch_as_list=self._tfrecords_iterator_attributes['batch_as_list'])

        return tfrecords_iterator

    def _load_and_decode(self, example):
        """Reads and decodes the data within each record in the tfrecord.

        Args:
            example (dict): Contains the data encoded in the tfrecords.

        Returns:
            image (tf.Tensor): Tensor of shape (height, width, 3)
            mask (tf.Tensor): Tensor of shape (height, width, 1)
            joints (tf.Tensor): Tensor of shape (num_people, num_joints, 3)
            meta_data (dict): Contains meta data required for processing
                images and labels.
        """

        # Get the dataset name
        dataset = tf.cast(example['dataset'], tf.string)
        # Get the person scales
        scales = tf.decode_raw(example['person/scales'], tf.float64)
        # Get the person centers
        centers = tf.decode_raw(example['person/centers'], tf.float64)

        # Create a dict to store the metadata
        meta_data = defaultdict()
        meta_data['dataset'] = dataset
        meta_data['width'] = example['frame/width']
        meta_data['height'] = example['frame/height']
        meta_data['image_id'] = example['frame/image_id']
        meta_data['scales'] = tf.reshape(scales, [-1])
        meta_data['centers'] = tf.reshape(centers, [-1, 2])

        # Get the labeled joints and transform dataset joints to BpNet joints
        # format
        joints = tf.decode_raw(example['person/joints'], tf.float64)
        joints = self._convert_kpts_to_bpnet_format(dataset, joints)

        # Load and decode image frame
        image = self._read_image_frame(
            self._load_file, self._decode_image, example['frame/image_path'])

        # Load and decode mask
        mask = self._read_image_frame(
            self._load_file, self._decode_mask, example['frame/mask_path'])

        return image, mask, joints, meta_data

    def _convert_kpts_to_bpnet_format(self, dataset_name_tensor, joints):
        """Converts the keypoints to bpnet format using the dataset parser.

        Args:
            dataset_name_tensor (tf.Tensor): Name of the dataset
            joints (tf.Tensor): Labeled joints in the current dataset format.

        Returns:
            result (tf.Tensor): Tensor of shape (num_people, num_joints, 3)
        """

        # Iterate through all the supported datasets
        for dataset_name in self.dataset2spec_map:

            # Check if the current dataset matches with any of the supported
            # datasets
            tf_logical_check = tf.math.equal(
                dataset_name_tensor, tf.constant(dataset_name, tf.string))

            # If there is a match, the corresponding dataset parser is used to transform
            # the labels to bpnet format. Else, it stores a constant zero to
            # the result.
            result = tf.cond(
                tf_logical_check,
                lambda dataset_name=dataset_name: tf.numpy_function(
                    self.dataset_cfgs[dataset_name].transform_labels,
                    [joints],
                    tf.float64),
                lambda: tf.constant(
                    0.0,
                    dtype=tf.float64))

        # TODO: Throw error if the dataset is not supported (tf.Assert)

        return result

    def _transform_labels(self, joint_labels):
        """Transform keypoints to final label tensors with heatmap and pafmap.

        Args:
            joint_labels (np.ndarray): Ground truth keypoint annotations with shape
                (num_persons, num_joints, 3).

        Returns:
            labels (tf.Tensor): Final label tensors used for training with
                shape (TARGET_HEIGHT, TARGET_WIDTH, num_channels).
        """

        labels = tf.numpy_function(
            self.label_processor.transform_labels, [joint_labels], tf.float64
        )

        return labels

    @property
    def num_samples(self):
        """Return number of samples in all label files."""

        num_samples = sum([sum(1 for _ in tf.python_io.tf_record_iterator(
            filename)) for filename in self.tfrecords_filename_list])
        return num_samples

    @staticmethod
    def _tfrecord_parser():
        """Load and set up Modulus TFRecord features parsers.

        Returns:
            A dict of tensors with the same keys as the features dict, and dense tensor.
        """

        # Processor for parsing meta `features`
        features = {
            "frame/image_path": tf.FixedLenFeature([], tf.string),
            "frame/mask_path": tf.FixedLenFeature([], tf.string),
            "frame/width": tf.FixedLenFeature([], tf.int64),
            "frame/height": tf.FixedLenFeature([], tf.int64),
            "frame/image_id": tf.FixedLenFeature([], tf.int64),
            "person/joints": tf.FixedLenFeature([], tf.string),
            "person/scales": tf.FixedLenFeature([], tf.string),
            "person/centers": tf.FixedLenFeature([], tf.string),
            "dataset": tf.FixedLenFeature([], tf.string)
        }

        proto_parser = tao_core.processors.ParseExampleProto(
            features=features, single=True)
        return proto_parser

    def _image_decoder(self):
        """Create the image decoder.

        Returns:
            decode_frame_image (tao_core.processors.DecodeImage) : Frame DecodeImage Processor object
                                                                  for decoding frame inputs.
        """
        # Create the image decoder.
        decode_frame_image = tao_core.processors.DecodeImage(
            encoding=self.image_encoding,
            data_format='channels_last',
            channels=self.image_dims['channels']
        )
        # Create the mask decoder.
        decode_frame_mask = tao_core.processors.DecodeImage(
            encoding=self.image_encoding,
            data_format='channels_last',
            channels=1
        )
        return decode_frame_image, decode_frame_mask

    @staticmethod
    def _read_image_frame(load_func, decode_func, image_name):
        """Read and decode a single image on disk to a tensor.

        Args:
            load_func (tao_core.processors.LoadFile): File loading function.
            decode_func (tao_core.processors.DecodeImage): Image decoding function.
            image_name (str): Name of the image.

        Returns:
            image (Tensor): A decoded 3D image tensor (HWC).
        """
        data = load_func(image_name)
        image = decode_func(data)
        image = tf.cast(image, tf.float32)

        return image

    @staticmethod
    def _resize_image(image, output_height, output_width):
        """Pre-process the image by resizing.

        Args:
            image (Tensor): Input image (HWC) to be processed.
            output_height (int): Output image height.
            output_width (int): Output image width.

        Returns:
            image (Tensor): The image tensor (HWC) after resizing.
        """
        image = tf.image.resize_images(image, (output_height, output_width),
                                       method=tf.image.ResizeMethod.BILINEAR)
        return image
