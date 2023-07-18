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

"""Converts a COCO pose estimation dataset to TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
from six.moves import range as xrange
import tensorflow as tf
import tqdm

from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _bytes_feature
from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _int64_feature
from nvidia_tao_tf1.cv.bpnet.dataio.coco_dataset import COCODataset
from nvidia_tao_tf1.cv.bpnet.dataio.dataset_converter_lib import DatasetConverter


logger = logging.getLogger(__name__)


class COCOConverter(DatasetConverter):
    """Converts a COCO dataset to TFRecords."""

    def __init__(self,
                 dataset_spec,
                 root_directory_path,
                 num_partitions,
                 num_shards,
                 output_filename,
                 mode,
                 generate_masks=False,
                 check_if_images_and_masks_exist=False):
        """Initialize the converter.

        Keypoint Ordering
        =================
        "keypoints": {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle"
        }
        ====================

        Args:
            root_directory_path (string): Dataset root directory path.
            dataset_spec (dict): Specifications and parameters to be used for
                the dataset export
            num_partitions (int): Number of partitions (folds).
            num_shards (int): Number of shards.
            output_filename (str): Path for the output file.
            mode (string): train/test.
            generate_masks (bool): Generate and save masks of regions with unlabeled people
            check_if_images_and_masks_exist (bool): check if the required files
                exist in the data location.
        """
        super(COCOConverter, self).__init__(
            root_data_directory_path=root_directory_path,
            num_partitions=num_partitions,
            num_shards=num_shards,
            output_filename=output_filename)

        # Create an instance of COCODataset class
        self.dataset = COCODataset(dataset_spec, parse_annotations=True)
        self.dataset_name = self.dataset.dataset_name
        # Get the person category config
        self.num_joints = self.dataset.pose_config["num_joints"]

        self.duplicate_data_with_each_person_as_center = \
            dataset_spec["duplicate_data_with_each_person_as_center"]

        self._check_if_images_and_masks_exist = check_if_images_and_masks_exist

        if mode == 'train':
            self.images_root_dir_path = self.dataset.train_images_root_dir_path
            self.mask_root_dir_path = self.dataset.train_masks_root_dir_path
            self.data = self.dataset.train_data
        else:
            self.images_root_dir_path = self.dataset.test_images_root_dir_path
            self.mask_root_dir_path = self.dataset.test_masks_root_dir_path
            self.data = self.dataset.test_data

        # Generate and save binary masks
        if generate_masks:
            mask_root_dir = os.path.join(self.root_dir, self.mask_root_dir_path)
            self.dataset.process_segmentation_masks(self.data, mask_root_dir)

        # Reformat data so it can be converted to tfrecords
        # by the base class `DatasetConverter`
        self.reformatted_data = self._reformat_dataset()

        # make output directory if not already existing
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))

    def _reformat_dataset(self):
        """Reformat the dataset as required before writing into tfrecords.

        This function compiles the necessary keys that would be stored in the tfrecords.
        It also provides an option to duplicate the pose data with multiple people
        with each data point starting with one person of interest (so that later
        during augmentation, the crop can be centered around this person)
        """

        reformatted_data = []
        for data_point in tqdm.tqdm(self.data):

            for _, main_person in enumerate(data_point['main_persons']):

                reformatted_data_point = {}
                reformatted_data_point['image_id'] = data_point['image_id']
                reformatted_data_point['image_meta'] = data_point['image_meta']

                image_path = os.path.join(
                    self.images_root_dir_path, data_point['image_meta']['file_name'])
                mask_path = os.path.join(
                    self.mask_root_dir_path, data_point['image_meta']['file_name'])
                reformatted_data_point['image_path'] = image_path
                reformatted_data_point['mask_path'] = mask_path

                # This path check takes a lot of time, so kept it optional if we need to verify.
                if self._check_if_images_and_masks_exist:
                    if not os.path.exists(os.path.join(self.root_dir, image_path)):
                        logger.warning(
                            "Skipping data point. Image doesn't exist: {}".format(image_path))
                        break
                    if not os.path.exists(os.path.join(self.root_dir, mask_path)):
                        logger.warning(
                            "Skipping data point. Mask doesn't exist: {}".format(mask_path))
                        break

                num_other_people = 0
                joints = [main_person["joint"]]
                scales = [main_person["scale_provided"]]
                centers = [main_person["objpos"]]

                iscrowd_flags = [main_person["iscrowd"]]
                segmentation_masks = [main_person["segmentation"]]
                num_keypoints = [main_person["num_keypoints"]]

                for oidx, other_person in enumerate(data_point['all_persons']):
                    if main_person is other_person:
                        person_idx = oidx
                        continue
                    if other_person["num_keypoints"] == 0:
                        continue

                    joints.append(other_person["joint"])
                    scales.append(other_person["scale_provided"])
                    centers.append(other_person["objpos"])
                    iscrowd_flags.append(other_person["iscrowd"])
                    segmentation_masks.append(other_person["segmentation"])
                    num_keypoints.append(other_person["num_keypoints"])
                    num_other_people += 1

                reformatted_data_point['person_idx'] = person_idx
                reformatted_data_point['num_other_people'] = num_other_people
                reformatted_data_point['joints'] = np.asarray(joints, dtype=np.float64)
                reformatted_data_point['scales'] = np.asarray(scales, dtype=np.float64)
                reformatted_data_point['centers'] = np.asarray(centers, dtype=np.float64)
                reformatted_data_point['iscrowd_flags'] = iscrowd_flags
                reformatted_data_point['segmentation_masks'] = segmentation_masks
                reformatted_data_point['num_keypoints'] = num_keypoints

                reformatted_data.append(reformatted_data_point)

                if not self.duplicate_data_with_each_person_as_center:
                    break

        return reformatted_data

    def _partition(self):
        """Partition dataset to self.output_partitions partitions based on sequences.

        Returns:
            partitions (list): A list of lists of data points, one list per partition.
        """
        if self.output_partitions > 1:
            partitions = [[] for _ in xrange(self.output_partitions)]

            # TODO: Sort the dataset based on imageid?

            for counter, data_point in enumerate(self.reformatted_data):
                partition_idx = counter % self.output_partitions

                partitions[partition_idx].append(data_point)
        else:
            partitions = [self.reformatted_data]

        return partitions

    def _create_example_proto(self, data_point):
        """Generate the example proto for this frame.

        Args:
            data_point (dict): Dictionary containing the details about the data sample.

        Returns:
            example (tf.train.Example): An Example containing all labels for the frame.
        """

        # Create proto for the training example. Populate with frame attributes.
        example = self._example_proto(data_point)

        if example is not None:
            # Add labels.
            self._add_person_labels(example, data_point)

        return example

    def _add_person_labels(self, example, data_point):
        """Add joint labels of all persons in the image to the Example protobuf.

        Args:
            data_point (dict): Dictionary containing the details about the data sample.
            example (tf.train.Example): An Example containing all labels for the frame.
        """

        joints = data_point['joints']
        scales = data_point['scales']
        centers = data_point['centers']
        person_idx = data_point['person_idx']
        num_other_people = data_point['num_other_people']

        f = example.features.feature
        f['person/joints'].MergeFrom(_bytes_feature(joints.tostring()))
        f['person/scales'].MergeFrom(_bytes_feature(scales.tostring()))
        f['person/centers'].MergeFrom(_bytes_feature(centers.tostring()))

        f['person/person_idx'].MergeFrom(_int64_feature(person_idx))
        f['person/num_other_people'].MergeFrom(_int64_feature(num_other_people))

    def _example_proto(self, data_point):
        """Generate a base Example protobuf to which COCO-specific features are added.

        Args:
            data_point (dict): Dictionary containing the details about the data sample.
        """

        image_id = data_point['image_id']
        width = data_point['image_meta']['width']
        height = data_point['image_meta']['height']
        image_path = data_point['image_path']
        mask_path = data_point['mask_path']

        example = tf.train.Example(features=tf.train.Features(feature={
            'frame/image_id': _int64_feature(image_id),
            'frame/height': _int64_feature(height),
            'frame/width': _int64_feature(width),
            'frame/image_path': _bytes_feature(str.encode(image_path)),
            'frame/mask_path': _bytes_feature(str.encode(mask_path)),
            'dataset': _bytes_feature(str.encode(self.dataset_name))
        }))

        return example
