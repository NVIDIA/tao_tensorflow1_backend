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
"""Default dataloader for FpeNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import six
import tensorflow as tf

from nvidia_tao_tf1.blocks.dataloader import DataLoader
import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.core.processors import ColorTransform
from nvidia_tao_tf1.core.processors import SpatialTransform
from nvidia_tao_tf1.core.processors.augment.color import get_random_color_transformation_matrix
from nvidia_tao_tf1.core.processors.augment.spatial import get_random_spatial_transformation_matrix
from nvidia_tao_tf1.cv.core.augment import RandomBlur
from nvidia_tao_tf1.cv.core.augment import RandomGamma
from nvidia_tao_tf1.cv.core.augment import RandomShift
# tf.compat.v1.enable_eager_execution()


def _lrange(*args, **kwargs):
    """Used for Python 3 compatibility since range() no longer returns a list."""
    return list(range(*args, **kwargs))


class FpeNetDataloader(DataLoader):
    """
    Dataloader with online augmentation for Fpe datasets.

    The dataloader reads labels and frame id from TFRecord files and compiles
    image and ground truth tensors used in training and validation.
    """

    ITERATOR_INIT_OP_NAME = "iterator_init"

    @tao_core.coreobject.save_args
    def __init__(self, batch_size, image_info, dataset_info, kpiset_info,
                 augmentation_info, num_keypoints, **kwargs):
        """
        Instantiate the dataloader.

        Args:
            batch_size (int): Size of minibatch.
            image_info (dict): Basic information of input images.
            dataset_info (dict): Basic information of datasets used for training.
            kpiset_info (dict): Basic information of KPI set.
            augmentation_info (dict): Parameters information for augmentation.
            keypoints (int): Number of facial keypoints.
        """
        super(FpeNetDataloader, self).__init__(**kwargs)
        self.batch_size = batch_size

        # Get data information from experiment specs.
        # Image information.
        self.image_width = image_info['image']['width']
        self.image_height = image_info['image']['height']
        self.image_channel = image_info['image']['channel']

        # Dataset information.
        self.image_extension = dataset_info['image_extension']
        self.root_path = dataset_info['root_path']
        self.tfrecords_directory_path = dataset_info['tfrecords_directory_path']
        # Occlusion specific sets
        self.no_occ_masksets = dataset_info['no_occlusion_masking_sets']
        self.tfrecords_set_id_train = dataset_info['tfrecords_set_id_train']
        self.tfrecord_folder_name = dataset_info['tfrecord_folder_name']
        self.file_name = dataset_info['tfrecord_file_name']
        # validation info
        self.tfrecords_set_id_val = dataset_info['tfrecords_set_id_val']
        # KPI testing dataset information.
        self.tfrecords_set_id_kpi = kpiset_info['tfrecords_set_id_kpi']

        self.num_keypoints = num_keypoints
        # assert self.num_keypoints in [68, 80, 104], \
        #     "Expect number of keypoints one of 68, 80 or 104"

        # Augmentation info
        self.augmentation_config = build_augmentation_config(augmentation_info)
        self.enable_online_augmentation = self.augmentation_config[
                                          "enable_online_augmentation"]
        self.enable_occlusion_augmentation = self.augmentation_config[
                                             "enable_occlusion_augmentation"]
        self.enable_resize_augmentation = self.augmentation_config[
                                          "enable_resize_augmentation"]
        self.augmentation_resize_scale = self.augmentation_config[
                                         "augmentation_resize_scale"]
        self.augmentation_resize_probability = self.augmentation_config[
                                               "augmentation_resize_probability"]
        self.patch_probability = self.augmentation_config["patch_probability"]
        self.size_to_image_ratio = self.augmentation_config["size_to_image_ratio"]
        self.mask_aug_patch = self.augmentation_config["mask_augmentation_patch"]

        # Flipping augmentation
        if self.augmentation_config['modulus_spatial_augmentation']['hflip_probability'] != 0:
            assert self.num_keypoints in [68, 80, 104], \
                ("Horizontal flip augmentation can only be applied to 68, 80, 104 face landmarks."
                 "Please set hflip_probability to be 0.0")

        self._flip_lm_ind_map = self._get_flip_landmark_mapping(num_keypoints=self.num_keypoints)

        frame_shape = [self.image_height, self.image_width, self.image_channel]
        frame_shape = map(float, frame_shape)
        self._stm_op, self._ctm_op, self._blur_op, \
            self._gamma_op, self._shift_op = \
            get_transformation_ops(self.augmentation_config, frame_shape)

        # Get the proto parser.
        self._proto_parser = self._tfrecord_parser(self.num_keypoints)

    def __call__(self, repeat=True, phase='training'):
        """
        Get input images and ground truth labels as tensors for training and validation.

        Returns the number of minibatches required to loop over all the datasets once.

        Args:
            repeat (bool): Whether the dataset can be looped over multiple times or only once.
            phase (str): Demonstrate the current phase: training, validation, kpi_testing.

        Returns:
            images (Tensor): Decoded input images of shape (NCHW).
            ground_truth_labels (Tensor): Ground truth labels of shape (1, num_outputs).
            num_samples (int): Total number of loaded data points.
            occ_masking_info (Tensor): Ground truth occlusions mask of shape (1, num_keypoints).
            face_bbox (Tensor): Face bounding box for kpi data.
            image_names (Tensor): Image names for kpi data.
        """
        # load and encode tfrecords.
        records, num_samples = self._loading_dataset(repeat=repeat, phase=phase)

        # Generate input images and ground truth labels.
        images, ground_truth_labels, masking_occ_info, face_bbox, \
            image_names = self._parse_records(records,
                                              self.num_keypoints, phase)

        if phase == 'kpi_testing':
            return images, ground_truth_labels, num_samples, \
                masking_occ_info, face_bbox, image_names

        return images, ground_truth_labels, num_samples, \
            masking_occ_info

    def _loading_dataset(self, repeat, phase):
        """Get TFRecordsIterator for a given set of TFRecord files.

        Args:
            repeat (bool): Whether the dataset can be looped over multiple times or only once.
            phase (str): Demonstrate the current phase: training, testing, validation, kpi_testing.

        Returns:
            records Dict<tf.Tensor>: Dict of Tensor represeting a batch of samples.
            num_samples: Number of samples in the training/validation dataset.
        """
        tfrecords_filename_list = []
        tf_folder_name = self.tfrecord_folder_name
        file_name = self.file_name

        if phase == 'kpi_testing':
            set_ids = self.tfrecords_set_id_kpi.split(' ')
        elif phase == 'validation':
            set_ids = self.tfrecords_set_id_val.split(' ')
        elif phase == 'training':
            set_ids = self.tfrecords_set_id_train.split(' ')
        else:
            raise NameError("Invalid phase")

        assert len(set_ids) != 0, \
            'Expects more than one dataset id in experiment_spec.'

        for set_id in set_ids:
            recordrootpath = self.tfrecords_directory_path
            folder_name = os.path.join(recordrootpath, set_id, tf_folder_name)
            tfrecord_filename = [os.path.join(folder_name, file_name)]
            tfrecords_filename_list.extend(tfrecord_filename)

        # Check validity of each file.
        for filename in tfrecords_filename_list:

            assert tf.data.TFRecordDataset(filename), \
                ('Expects each file to be valid!', filename)

        # Print number of files
        num_samples = 0
        for filename in tfrecords_filename_list:
            num_samples_set = sum(1 for _ in tf.python_io.tf_record_iterator(filename))
            num_samples += num_samples_set
            print(filename+': '+str(num_samples_set))
        print("Total Samples: {}".format(num_samples))

        # Create different iterators based on different phases.
        if phase == 'kpi_testing':
            shuffle_buffer_size = 0
            shuffle = False
            repeat = True
        else:
            shuffle_buffer_size = num_samples
            shuffle = True

        dataset = tf.data.TFRecordDataset(
            tfrecords_filename_list,
            num_parallel_reads=multiprocessing.cpu_count()
        )

        # Shard dataset in multi-gpu cases
        rank = distribution.get_distributor().rank()
        size = distribution.get_distributor().size()
        dataset = dataset.shard(size, rank)
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True
            )

        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.map(self._proto_parser, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self._load_and_decode,
                              num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.prefetch(3)
        iterator = tf.compat.v1.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes
        )
        iterator_init_op = iterator.make_initializer(dataset)
        tf.compat.v1.add_to_collection(
           self.ITERATOR_INIT_OP_NAME, iterator_init_op
        )
        # Pull the records from tensorflow dataset.
        records = iterator.get_next()

        return records, num_samples

    @staticmethod
    def _tfrecord_parser(num_keypoints):
        """
        Load and set up Modulus TFRecord features parsers.

        Args:
            num_keypoints (int): Number of keypoints.
        Returns:
            A dict of tensors with the same keys as the features dict, and dense tensor.
        """
        # Processor for parsing meta `features`
        features = {
            'train/image_frame_name': tf.FixedLenFeature([], dtype=tf.string),
            'train/image_frame_width': tf.FixedLenFeature([], dtype=tf.int64),
            'train/image_frame_height': tf.FixedLenFeature([], dtype=tf.int64),
            'train/facebbx_x': tf.FixedLenFeature([], dtype=tf.int64),
            'train/facebbx_y': tf.FixedLenFeature([], dtype=tf.int64),
            'train/facebbx_w': tf.FixedLenFeature([], dtype=tf.int64),
            'train/facebbx_h': tf.FixedLenFeature([], dtype=tf.int64),
            'train/landmarks': tf.FixedLenFeature([num_keypoints * 2], dtype=tf.float32),
            'train/landmarks_occ': tf.FixedLenFeature([num_keypoints], dtype=tf.int64)
        }

        proto_parser = tao_core.processors.ParseExampleProto(features=features, single=True)
        return proto_parser

    def _load_and_decode(self, records):
        """
        Load and decode images.

        Args:
            records (tf.Tensor): Records from dataset to process.
        Returns:
            records (tf.Tensor): Records contains loaded images.
        """
        file_loader = tao_core.processors.LoadFile(prefix=self.root_path)
        train_frames = []
        train_kpts = []
        for index in range(self.batch_size):
            image_frame_name = records['train/image_frame_name'][index]
            image_frame = self._read_image_frame(file_loader, image_frame_name)

            cropped_face, kpts_norm = self._crop_image(image_frame,
                                                       records['train/facebbx_x'][index],
                                                       records['train/facebbx_y'][index],
                                                       records['train/facebbx_w'][index],
                                                       records['train/facebbx_h'][index],
                                                       records['train/landmarks'][index],
                                                       self.image_height,
                                                       self.image_width,
                                                       channels=self.image_channel,
                                                       num_keypoints=self.num_keypoints)

            train_frames.append(cropped_face)
            train_kpts.append(kpts_norm)

        def _stack_frames(frames):
            if len(frames) > 0:
                return tf.stack(frames, 0)
            return tf.constant(0, shape=[self.batch_size, 0])

        records.update({
            'train/cropped_face': _stack_frames(train_frames),
            'train/kpts_norm': _stack_frames(train_kpts)
        })

        return records

    @staticmethod
    def _crop_image(image,
                    facebbox_x,
                    facebbox_y,
                    facebbox_width,
                    facebbox_height,
                    landmarks,
                    target_width,
                    target_height,
                    channels=3,
                    num_keypoints=80):
        """
        Crop bounding box from image & Scale the Keypoints to Target Resolution.

        Args:
            image (Tensor): Input image tensor.
            facebbox_x (scalar Tensor): top-right X pixel location of face bounding box.
            facebbox_y (scalar Tensor): top-right Y pixel location of face bounding box.
            facebbox_width (scalar Tensor): width of face bounding box.
            facebbox_height (scalar Tensor): height of face bounding box.
            landmarks (Tensor): Input keypoint (x,y) locations [num_keypoints X 2]
            target_width (int): Target width of bounding box.
            target_height (int): Target height of bounding box.
            channels (int): Number of channels in image.
            num_keypoints (int): Number of keypoints.

        Returns:
            image (Tensor): Output cropped image (HWC) & Scaled Keypoints for target resolution.
            kpts_target (Tensor): image keypoints after cropping and scaling.
        """
        kpts = landmarks[:2 * num_keypoints]

        kpts = tf.cast(kpts, dtype=tf.int32)
        x = tf.cast(facebbox_x, dtype=tf.int32)
        y = tf.cast(facebbox_y, dtype=tf.int32)
        h = tf.cast(facebbox_height, dtype=tf.int32)
        w = tf.cast(facebbox_width, dtype=tf.int32)

        img = tf.image.crop_to_bounding_box(image, y, x, h, w)
        img_shape = tf.stack([h, w, channels])
        image = tf.reshape(img, img_shape)

        image = tf.image.resize(image,
                                (target_height, target_width),
                                method=tf.image.ResizeMethod.BILINEAR)

        # make it channel first (channel, height, width)
        image = tf.transpose(image, (2, 0, 1))
        kpts_shape = tf.stack([num_keypoints, 2])
        kpts_norm = tf.reshape(kpts, kpts_shape)

        kpts_x = tf.cast(kpts_norm[:, 0], dtype=tf.float32)
        kpts_y = tf.cast(kpts_norm[:, 1], dtype=tf.float32)

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        w = tf.cast(w, dtype=tf.float32)
        h = tf.cast(h, dtype=tf.float32)

        kpts_norm_x = (kpts_x - x) / w
        kpts_norm_y = (kpts_y - y) / h

        kpts_x_target = kpts_norm_x * target_width
        kpts_y_target = kpts_norm_y * target_height

        kpts_target = tf.stack([kpts_x_target, kpts_y_target], axis=1)

        return image, kpts_target

    def _parse_records(self, records, num_keypoints, phase='validation'):
        """
        Return generators for input image and output target tensors.

        Args:
            records (Dict<tf.Tensor>): Dict of tf.Tensor represeting training samples.
            num_keypoints (int): Number of keypoints.
            phase (string): training, validation, kpi_testing.

        Returns:
            images (Tensor): 4D image tensors with shape (NCHW).
            labels (Tensor): 2D labels tensor of shape (1, num_outputs).
        """
        # Untack the batched tensor into list of tensors.
        records = {
            key: tf.unstack(value, axis=0)
            for key, value in records.items()
        }
        records = [
            {
                key: value[idx]
                for key, value in records.items()
            }
            for idx in range(self.batch_size)
        ]

        # Augmentation only enabled during training.
        enable_augmentation = phase == 'training' and self.enable_online_augmentation
        enable_occlude = enable_augmentation and self.enable_occlusion_augmentation
        enable_resize_augment = enable_augmentation and self.enable_resize_augmentation

        # Initialize lists for each input and output.
        data_image = []
        labels_kpts = []
        labels_occ = []
        face_bbox = []
        image_names = []
        masking_occ_info = []

        for record in records:

            image_frame_name = record['train/image_frame_name']
            cropped_face = tf.cast(record['train/cropped_face'], tf.float32)
            kpts_norm = tf.cast(record['train/kpts_norm'], tf.float32)

            kpts_occ = record['train/landmarks_occ']
            face_bbox.append((record['train/facebbx_x'],
                              record['train/facebbx_y'],
                              record['train/facebbx_w'],
                              record['train/facebbx_h']))

            kpts_mask = 1.0 - tf.cast(kpts_occ, dtype=tf.float32)[:num_keypoints]
            # 1-visible, 0-occluded

            if enable_augmentation:
                if enable_resize_augment:
                    cropped_face = self.resize_augmentations(
                                   cropped_face,
                                   self.augmentation_resize_scale,
                                   self.augmentation_resize_probability)

                if enable_occlude:
                    cropped_face, kpts_mask = self.random_patches(
                                              tf.transpose(cropped_face),
                                              kpts_norm,
                                              kpts_mask,
                                              probability=self.patch_probability,
                                              size_to_image_ratio=self.size_to_image_ratio)

                # obtain random spatial transformation matrices.
                sm, _ = get_all_transformations_matrices(self.augmentation_config,
                                                         self.image_height,
                                                         self.image_width,
                                                         enable_augmentation=enable_augmentation)

                # Apply augmentations to frame tensors.
                cropped_face = self._apply_augmentations_to_frame(cropped_face, sm)
                cropped_face = tf.transpose(cropped_face, perm=[2, 0, 1])

                # Apply gamma augmentation
                self._gamma_op.build()
                cropped_face = self._gamma_op(cropped_face)

                # Apply augmentations to keypoints
                kpts_norm = self._apply_augmentations_to_kpts(kpts_norm, num_keypoints, sm)

                # Apply flipping augmentation
                # if image is flipped then x value of landmark is flipped.
                flip_lr_flag = tf.equal(tf.sign(sm[0][0]), -1)
                kpts_norm, kpts_mask = self._flip_landmarks(kpts_norm, kpts_mask, flip_lr_flag)

            data_image.append(cropped_face)
            labels_kpts.append(kpts_norm)
            labels_occ.append(kpts_mask)
            image_names.append(image_frame_name)

            # occlusion masking exception handling
            masking_info = []
            for no_occ_set in self.no_occ_masksets.split(' '):
                regex_pattern = tf.compat.v1.string_join(['.*', no_occ_set, '.*'])
                masking_info.append(
                    tf.compat.v1.strings.regex_full_match(image_frame_name, regex_pattern)
                )
            masking_occ_info.append(tf.cast(tf.reduce_any(masking_info), tf.float32))

        # Batch together list of tensors.
        input_images = tf.stack(data_image)
        datalabels = [tf.stack(labels_kpts), tf.stack(labels_occ)]
        masking_occ_info = tf.stack(masking_occ_info)
        face_bbox = tf.stack(face_bbox)
        image_names = tf.stack(image_names)

        return input_images, datalabels, masking_occ_info, face_bbox, image_names

    def _read_image_frame(self, load_func, image_name):
        """Read and decode a single image on disk to a tensor.

        Args:
            load_func (tao_core.processors.LoadFile): File loading function.
            image_name (str): Name of the image.

        Returns:
            image (Tensor): A decoded 3D image tensor (HWC).
        """
        data = load_func(image_name)
        image = tf.image.decode_png(data, channels=self.image_channel)

        return image

    def _get_flip_landmark_mapping(self, num_keypoints=80):
        """
        Compute order of facial landmarks for horizontally flipped image.

        Face keypoints ordering listed here-
        https://docs.google.com/document/d/13q8NciZtGyx5TgIgELkCbXGfE7PstKZpI3cENBGWkVw/edit#

        Args:
            num_keypoints (int): Number of keypoints. Options- 68, 80, 104.
        Returns:
            flip_lm_ind_map (list): order of facial landmarks for flipped image.
        """
        # common face regions for 68 points
        chin_ind_flip = _lrange(17)[::-1] + _lrange(17, 27)[::-1]
        nose_ind_flip = _lrange(27, 31) + _lrange(31, 36)[::-1]
        eye_ind_flip = [45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40]
        mouth_ind_flip = (_lrange(48, 55)[::-1] + _lrange(55, 60)[::-1] + _lrange(60, 65)[::-1]
                          + _lrange(65, 68)[::-1])

        # For 80 points
        pupil_ind_flip = [74, 73, 72, 75, 70, 69, 68, 71]
        ear_ind_flip = [78, 79, 76, 77]

        # For 104 points
        extra_ind_flip = ([91, 92, 93] +
                          _lrange(94, 101)[::-1] +
                          [101] + [80, 81, 82] +
                          _lrange(83, 90)[::-1] +
                          [90] + [103, 102])

        # collection all face regions
        flip_lm_ind_map = (chin_ind_flip + nose_ind_flip + eye_ind_flip + mouth_ind_flip)

        if num_keypoints == 80:
            flip_lm_ind_map = (flip_lm_ind_map + pupil_ind_flip + ear_ind_flip)
        if num_keypoints == 104:
            flip_lm_ind_map = (flip_lm_ind_map + pupil_ind_flip + ear_ind_flip + extra_ind_flip)

        return flip_lm_ind_map

    def _apply_augmentations_to_frame(self, input_tensor, sm):
        """
        Apply spatial and color transformations to an image.

        Spatial transform op maps destination image pixel P into source image location Q
        by matrix M: Q = P M. Here we first compute a forward mapping Q M^-1 = P, and
        finally invert the matrix.

        Args:
            input_tensor (Tensor): Input image frame tensors (HWC).
            sm (Tensor): 3x3 spatial transformation/augmentation matrix.

        Returns:
            image (Tensor, CHW): Augmented input tensor.
        """
        # Convert image to float if needed (stm_op requirement).
        if input_tensor.dtype != tf.float32:
            input_tensor = tf.cast(input_tensor, tf.float32)

        dm = tf.matrix_inverse(sm)
        # NOTE: Image and matrix need to be reshaped into a batch of one for this op.
        # Apply spatial transformations.

        input_tensor = tf.transpose(input_tensor, perm=[1, 2, 0])
        image = self._stm_op(images=tf.stack([tf.image.grayscale_to_rgb(input_tensor)]),
                             stms=tf.stack([dm]))
        image = tf.image.rgb_to_grayscale(image)

        image = tf.reshape(image, [self.image_height, self.image_width,
                                   self.image_channel])
        return image

    def _apply_augmentations_to_kpts(self, key_points, num_keypoints, mapMatrix):
        """
        Apply augmentation to keypoints.

        This methods get matrix of keypoints and returns a matrix of
        their affine transformed location.

        Args:
            key_points: a matrix of key_point locations in the format (#key-points, 2)
            num_keypoints: number of keypoints
            MapMatrix: affine transformation of shape (2 * 3)

        Returns:
            A matrix of affine transformed key_point location in the
            format (#key-points, 2)
        """
        kpts = tf.concat([tf.transpose(key_points),
                          tf.ones([1, num_keypoints],
                          dtype=tf.float32)], axis=0)
        new_kpt_points = tf.matmul(tf.transpose(mapMatrix), kpts)
        new_kpt_points = tf.slice(new_kpt_points, [0, 0], [2, -1])

        return tf.transpose(new_kpt_points)

    def resize_augmentations(self,
                             cropped_face,
                             augmentation_resize_scale,
                             augmentation_resize_probability):
        """
        Obtain resize augmentations.

        This methods get a cropped face image and performs resize augmentation.

        Args:
            cropped_face (Tensor): Tensor of cropped image.
            augmentation_resize_scale (float): scale for resize image.
            augmentation_resize_probability (float): probability for applying augmentation.

        Returns:
            A matrix of affine transformed key_point location in the
            format (#key-points, 2)
        """
        def resize_aug(cropped_face, augmentation_resize_scale):

            resize_height = int(self.image_height * self.augmentation_resize_scale)
            resize_width = int(self.image_width * self.augmentation_resize_scale)
            resize_shape = (resize_height, resize_width)
            cropped_face = tf.image.resize(tf.transpose(cropped_face), resize_shape)

            cropped_face = tf.image.resize(cropped_face, (self.image_height, self.image_width))
            cropped_face = tf.transpose(cropped_face)
            return cropped_face

        def no_resize_aug(cropped_face):
            return cropped_face

        prob = tf.random.uniform([1], minval=0, maxval=1.0, dtype=tf.float32)

        augmentation_prob_condition = tf.reshape(tf.greater(prob,
                                                 tf.constant(augmentation_resize_probability)), [])
        cropped_face = tf.cond(augmentation_prob_condition,
                               lambda: resize_aug(cropped_face, augmentation_resize_scale),
                               lambda: no_resize_aug(cropped_face))

        return cropped_face

    def random_patches(self,
                       image,
                       kpts,
                       kpts_mask,
                       probability=0.5,
                       size_to_image_ratio=0.15):
        """
        Overlay a random sized patch on the image.

        Args:
            image (Tensor): Input image frame tensors.
            kpts (Tensor): Ground truth facial keypoints.
            kpts_mask (Tensor): Ground truth facial keypoints occlusion flags.
            probability: Probability to add occlusion.
            size_to_image_ratio: Maximum scale of occlusion.

        Returns:
            Image with an occluded region.
        """
        def occlusion(image, kpts, kpts_mask, size_to_image_ratio=0.15):
            image_shape = tf.shape(image)

            # get random location
            # get random size
            min_size = 10  # min pixel size of occlusion boxes
            max_size = tf.multiply(tf.cast(image_shape[0],
                                           dtype=tf.float32), tf.constant(size_to_image_ratio))

            size_x = tf.random.uniform([], minval=min_size, maxval=max_size)
            size_y = tf.random.uniform([], minval=min_size, maxval=max_size)

            # get box with ones
            ones_box = tf.ones([tf.cast(size_x, tf.int32), tf.cast(size_y, tf.int32), 1])

            # pad box to image size with zeros
            mask = tf.image.resize_with_crop_or_pad(ones_box, image_shape[0], image_shape[1])

            mask_zeros = tf.cast(-1.0 * (mask - 1.0), tf.float32)

            # apply masking to newly occluded points
            occ_aug_mask = tf.gather_nd(mask_zeros, tf.cast(kpts, tf.int32))
            kpts_mask_new = tf.math.multiply(kpts_mask, occ_aug_mask[:, 0])

            # multiply box with image
            mask_image = tf.multiply(image, mask_zeros)

            # get random color
            color_mask = tf.multiply(mask, tf.random.uniform([],
                                                             minval=0,
                                                             maxval=255,
                                                             dtype=tf.float32))

            # add box to image
            image = tf.add(mask_image, color_mask)

            return tf.transpose(image), kpts_mask_new

        def no_occlusion(image, kpts_mask):
            return tf.transpose(image), kpts_mask

        prob = tf.random.uniform([1], minval=0, maxval=1.0, dtype=tf.float32)

        image, kpts_mask_new = tf.cond(tf.reshape(tf.greater(prob, tf.constant(probability)), []),
                                       lambda: occlusion(image,
                                       kpts,
                                       kpts_mask,
                                       size_to_image_ratio),
                                       lambda: no_occlusion(image, kpts_mask))

        if self.mask_aug_patch:
            kpts_mask = kpts_mask_new
        return image, kpts_mask

    def _flip_landmarks(self, kpts_norm, kpts_mask, flip_lr_flag):
        """
        Utility to flip landmarks and occlusion masks.

        Args:
            kpts_norm (Tensor): Original keypoints.
            kpts_mask (Tensor): Original occlusion mask.
            flip_lr_flag (Bool): Bool flag for flipping keypoints.
        Returns:
            kpts_norm (Tensor): flipped keypoints.
            kpts_mask (Tensor): flipped occlusion mask.
        """
        kpts_norm = tf.cond(
            pred=flip_lr_flag,
            true_fn=lambda: tf.gather(kpts_norm, self._flip_lm_ind_map),
            false_fn=lambda: kpts_norm)
        kpts_mask = tf.cond(
            pred=flip_lr_flag,
            true_fn=lambda: tf.gather(kpts_mask, self._flip_lm_ind_map),
            false_fn=lambda: kpts_mask)

        return kpts_norm, kpts_mask


def build_augmentation_config(augmentation_info):
    """
    Creates a default augmentation config and updates it with user augmentation info.

    User provided augmentation specification is updated with default values for unspecified
    fields.

    Args:
        augmentation_info (dict): generated from yaml spec.

    Returns:
        config (dict): augmentation information with default values for unspecified.
    """

    modulus_spatial_augmentation = {
        'hflip_probability': 0.0,
        'zoom_min': 1.0,
        'zoom_max': 1.0,
        'translate_max_x': 0.0,
        'translate_max_y': 0.0,
        'rotate_rad_max': 0.0
    }
    modulus_color_augmentation = {
        'hue_rotation_max': 0.0,
        'saturation_shift_max': 0.0,
        'contrast_scale_max': 0.0,
        'contrast_center': 127.5,  # Should be 127.5 if images are in [0,255].
        'brightness_scale_max': 0,
        'brightness_uniform_across_channels': True,
    }
    gamma_augmentation = {
        'gamma_type': 'uniform',
        'gamma_mu': 1.0,
        'gamma_std': 0.3,
        'gamma_max': 1.0,
        'gamma_min': 1.0,
        'gamma_probability': 0.0
    }
    blur_augmentation = {
        'kernel_sizes': [],
        'blur_probability': 0.0,
        'channels': 1
    }
    random_shift_bbx_augmentation = {
        'shift_percent_max': 0.0,
        'shift_probability': 0.0
    }

    config = {
        'modulus_spatial_augmentation': modulus_spatial_augmentation,
        'modulus_color_augmentation': modulus_color_augmentation,
        'gamma_augmentation': gamma_augmentation,
        'enable_online_augmentation': False,
        'blur_augmentation': blur_augmentation,
        'random_shift_bbx_augmentation': random_shift_bbx_augmentation,
    }

    def _update(d, u):
        """Update nested dictionaries.

        Args:
            d (dict): Nested dictionary.
            u (dict): Nested dictionary.

        Returns:
            d (dict): Nested dictionary that has been updated.
        """
        for k, v in six.iteritems(u):
            if isinstance(v, dict):
                d[k] = _update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = _update(config, augmentation_info)

    return config


def get_transformation_ops(augmentation_config, frame_shape):
    """
    Generate ops which will apply spatial / color transformations, custom blur and gamma ops.

    Args:
        augmentation_config (dict): Contains configuration for augmentation.
        frame_shape (list): Shape of frame (HWC).

    Returns:
        stm_op (Modulus Processor): Spatial transformation op.
        ctm_op (Modulus Processor): Color transformation op.
        blur_op (Modulus Processor): Custom blur op.
        gamma_op (Modulus Processor): Custom gamma correction op.
        shift_op (Modulus Processor): Custom bounding box shifting op.
    """
    # Set up spatial transform op.
    stm_op = SpatialTransform(method='bilinear', background_value=0.0, data_format="channels_last")

    # Set up color transform op.
    # NOTE: Output is always normalized to [0,255] range.
    ctm_op = ColorTransform(min_clip=0.0, max_clip=255.0, data_format="channels_last")

    # Set up random blurring op.
    blur_choices = augmentation_config["blur_augmentation"]["kernel_sizes"]
    blur_probability = augmentation_config["blur_augmentation"]["blur_probability"]
    channels = augmentation_config["blur_augmentation"]["channels"]
    blur_op = RandomBlur(blur_choices=blur_choices,
                         blur_probability=blur_probability,
                         channels=channels)

    # Set up random gamma op.
    gamma_type = augmentation_config["gamma_augmentation"]["gamma_type"]
    gamma_mu = augmentation_config["gamma_augmentation"]["gamma_mu"]
    gamma_std = augmentation_config["gamma_augmentation"]["gamma_std"]
    gamma_max = augmentation_config["gamma_augmentation"]["gamma_max"]
    gamma_min = augmentation_config["gamma_augmentation"]["gamma_min"]
    gamma_probability = augmentation_config["gamma_augmentation"]["gamma_probability"]
    gamma_op = RandomGamma(gamma_type=gamma_type, gamma_mu=gamma_mu, gamma_std=gamma_std,
                           gamma_max=gamma_max, gamma_min=gamma_min,
                           gamma_probability=gamma_probability)

    # Set up random shift op.
    shift_percent_max = augmentation_config["random_shift_bbx_augmentation"]["shift_percent_max"]
    shift_probability = augmentation_config["random_shift_bbx_augmentation"]["shift_probability"]
    shift_op = RandomShift(shift_percent_max=shift_percent_max, shift_probability=shift_probability,
                           frame_shape=frame_shape)

    return stm_op, ctm_op, blur_op, gamma_op, shift_op


def get_spatial_transformations_matrix(spatial_augmentation_config, image_width, image_height):
    """Generate a spatial transformations matrix that applies both preprocessing and augmentations.

    Args:
        spatial_augmentation_config (dict): Contains configuration for spatial augmentation:
                                            'hflip_probability' (float)
                                            'translate_max_x' (int)
                                            'translate_max_y' (int)
                                            'zoom_min' (float)
                                            'zoom_max' (float)
                                            'rotate_rad_max' (float)
        image_width (int): Width of image canvas
        image_height (int): Height of image canvas

    Returns:
        stm (Tensor 3x3): Matrix that transforms from original image space to augmented space.
    """
    hflip_probability = spatial_augmentation_config["hflip_probability"]
    translate_max_x = int(spatial_augmentation_config["translate_max_x"])
    translate_max_y = int(spatial_augmentation_config["translate_max_y"])
    zoom_ratio_min = spatial_augmentation_config["zoom_min"]
    zoom_ratio_max = spatial_augmentation_config["zoom_max"]
    rotate_rad_max = spatial_augmentation_config["rotate_rad_max"]

    # Create spatial transformation matrices on CPU.
    # NOTE: Creating matrices on GPU is much much slower.
    with tf.device('/CPU'):
        stm = get_random_spatial_transformation_matrix(
                image_width, image_height,
                flip_lr_prob=hflip_probability,
                translate_max_x=translate_max_x,
                translate_max_y=translate_max_y,
                zoom_ratio_min=zoom_ratio_min,
                zoom_ratio_max=zoom_ratio_max,
                rotate_rad_max=rotate_rad_max)

    return stm


def get_color_augmentation_matrix(color_augmentation_config):
    """Generate a color transformations matrix applying augmentations.

    Args:
        color_augmentation_config (dict): Contains configuration for color augmentation:
                                          'hue_rotation_max' (float)
                                          'saturation_shift_max' (float)
                                          'contrast_scale_max' (float)
                                          'contrast_center' (float)
                                          'brightness_scale_max' (float)
                                          'brightness_uniform_across_channels' (bool)

    Returns:
        ctm (Tensor 4x4): Matrix describing the color transformation to be applied.
    """
    hue_rotation_max = color_augmentation_config["hue_rotation_max"]
    saturation_shift_max = color_augmentation_config["saturation_shift_max"]
    contrast_scale_max = color_augmentation_config["contrast_scale_max"]
    contrast_center = color_augmentation_config["contrast_center"]
    brightness_scale_max = color_augmentation_config["brightness_scale_max"]
    brightness_uniform = color_augmentation_config["brightness_uniform_across_channels"]

    # Create color transformation matrices on CPU.
    # NOTE: Creating matrices on GPU is much much slower.
    with tf.device('/CPU'):
        ctm = get_random_color_transformation_matrix(
                hue_rotation_max=hue_rotation_max,
                saturation_shift_max=saturation_shift_max,
                contrast_scale_max=contrast_scale_max,
                contrast_center=contrast_center,
                brightness_scale_max=brightness_scale_max,
                brightness_uniform_across_channels=brightness_uniform)
    return ctm


def get_all_transformations_matrices(augmentation_config, image_height, image_width,
                                     enable_augmentation=False):
    """Generate all the color and spatial transformations as defined in augmentation_config.

    Input image values are assumed to be in the [0, 1] range.

    Args:
        augmentation_config (dict): Contains augmentation configuration for
                                    'modulus_spatial_augmentation',
                                    'modulus_color_augmentation'.
        image_height (int): Height of image canvas.
        image_width (int): Width of image canvas.
        enable_augmentation (bool): Toggle to turn off augmentations during non-training phases.

    Returns:
        stm (Tensor 3x3): matrix that transforms from original image space to augmented space.
        ctm (Tensor 4x4): color transformation matrix.
    """
    if not enable_augmentation:
        # Default Spatial and Color Transformation matrices.
        stm = tf.eye(3, dtype=tf.float32)
        ctm = tf.eye(4, dtype=tf.float32)
        return stm, ctm

    spatial_augmentation_config = augmentation_config["modulus_spatial_augmentation"]
    color_augmentation_config = augmentation_config["modulus_color_augmentation"]

    # Compute spatial transformation matrix.
    stm = get_spatial_transformations_matrix(spatial_augmentation_config, image_width, image_height)

    # Compute color transformation matrix.
    ctm = get_color_augmentation_matrix(color_augmentation_config)

    return stm, ctm
