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

"""Dataset config class for BpNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

from nvidia_tao_tf1.cv.bpnet.dataloaders.pose_config import Joints
from nvidia_tao_tf1.cv.bpnet.dataloaders.pose_config import JointVisibility


class DatasetConfig(object):
    """Class to hold all dataset related parameters."""

    def __init__(self,
                 dataset_spec_path,
                 bpnet_pose_config,
                 **kwargs):
        """Constructor.

        Args:
            pose_config (PoseConfig)
            dataset_spec (dict): Dataset configuration spec.
        """

        self.pose_config = bpnet_pose_config

        # Load the dataset config
        self.dataset_config = json.load(open(dataset_spec_path))

        # Get the person category config
        self.person_config = self._get_category(self.dataset_config, 'person')

        # Get number of joints and edges in the skeleton and assert correctness
        self.num_parts = self.person_config["num_joints"]
        self.parts = self.person_config["keypoints"]
        self.parts2idx = dict(zip(self.parts, range(self.num_parts)))
        assert self.num_parts == len(self.parts)

        self.skeleton = self.person_config["skeleton_edge_names"]
        self.skeleton2idx = self.person_config["skeleton"]
        self.num_connections = len(self.skeleton)

        # Check whether to estimate and add neck joint
        self._add_neck_joint = False
        neck_needed = (Joints.NECK in self.pose_config.parts2idx) \
            and (Joints.NECK not in self.parts2idx)
        shoulders_labeled = (Joints.RIGHT_SHOULDER in self.parts2idx) \
            and (Joints.LEFT_SHOULDER in self.parts2idx)
        if neck_needed and shoulders_labeled:
            self._add_neck_joint = True

        self.visibility_flag_map = DatasetConfig.get_visibility_flag_mapping(
            self.dataset_config['visibility_flags'])

    @staticmethod
    def _get_category(data, cat_name):
        """Get the configuration corresponding to the given category name."""

        return [c for c in data['categories'] if c['name'] == cat_name][0]

    @staticmethod
    def get_visibility_flag_mapping(visibility_flags):
        """Map visiblity flag convention in the dataset to JointVisiblity.

        Args:
            visibility_flags (dict): visibility flag convention used in dataset

        Returns:
            mapdict (dict): mapping between dataset and JointVisiblity convention.
        """

        mapdict = {}
        dest_str2var = JointVisibility.map_visibiltiy_str
        for key in visibility_flags['value'].keys():
            # source value of the corresponding occlusion flag
            source_val = visibility_flags['value'].get(key)
            # desired value of the corresponding occlusion flag
            dest_flag = visibility_flags['mapping'].get(key)
            if dest_flag is None:
                raise Exception("Visibility flag is missing {} key in `mapping`".format(key))
            dest_val = dest_str2var.get(dest_flag)
            if dest_val is None:
                raise Exception("Visibility flag is missing {} key in `mapping`".format(key))
            mapdict[source_val] = dest_val
        return mapdict

    def transform_labels(self, joint_labels):
        """Map the keypoints from current dataset format to bpnet format.

        Args:
            joint_labels (np.ndarray): Keypoints in current dataset format

        Returns:
            formatted_joint_labels (np.ndarray): Mapped keypoints in bpnet format
        """

        # Reshape the joints to (num_persons, num_joints in dataset, 3)
        joint_labels = np.reshape(joint_labels, (-1, self.num_parts, 3))

        # Intialize an array for formatted keypoints with shape (num_persons, num_bpnet_joints. 3)
        formatted_joint_labels = np.zeros(
            (joint_labels.shape[0], self.pose_config.num_parts, 3), dtype=np.float
        )

        # Set default visibility flag to not labeled in the dataset
        formatted_joint_labels[:, :, 2] = JointVisibility.NL_IN_DATASET

        # Iterate through each part and map them to the corresponding index in bpnet joint format
        for part in self.parts:
            dataset_part_id = self.parts2idx[part]

            if part in self.pose_config.parts2idx:
                bpnet_part_id = self.pose_config.parts2idx[part]
                formatted_joint_labels[:, bpnet_part_id, :] = joint_labels[:, dataset_part_id, :]

        # Convert the visibility flags
        tmp_array = np.copy(formatted_joint_labels)
        for flag in self.visibility_flag_map:
            formatted_joint_labels[:, :, 2] = np.where(
                np.isclose(tmp_array[:, :, 2], flag),
                self.visibility_flag_map[flag],
                formatted_joint_labels[:, :, 2]
            )

        # Add neck joint if it isn't part of the current dataset
        if self._add_neck_joint:
            formatted_joint_labels = self._estimate_neck_joint(joint_labels, formatted_joint_labels)

        return formatted_joint_labels

    def _estimate_neck_joint(self, joint_labels, formatted_joint_labels):
        """Estimate the neck joint using the average of left and right shoulders.

        Args:
            joint_labels (np.ndarray): Keypoints in current dataset format
            formatted_joint_labels (np.ndarray): Mapped keypoints in bpnet format

        Returns:
            formatted_joint_labels (np.ndarray): Mapped keypoints in bpnet format
                with neck keypoint added based on lsho and rsho keypoints.
        """

        # Get the relevant part indices
        neck_idx = self.pose_config.parts2idx[Joints.NECK]
        rsho_idx = self.parts2idx[Joints.RIGHT_SHOULDER]
        lsho_idx = self.parts2idx[Joints.LEFT_SHOULDER]

        # Get the visibility flags of the left and right shoulders
        lsho_labeled = joint_labels[:, lsho_idx, 2] < JointVisibility.L_TRUNCATED
        rsho_labeled = joint_labels[:, rsho_idx, 2] < JointVisibility.L_TRUNCATED
        both_sho_labeled = lsho_labeled & rsho_labeled

        # Update neck joint as average of the left and right shoulder joints
        formatted_joint_labels[:, neck_idx, 0:2] = \
            (joint_labels[:, lsho_idx, 0:2] + joint_labels[:, rsho_idx, 0:2]) / 2

        # Update the visibility of the neck joint based on the visibilities of the
        # left and right shoulders. Only when both are visible/labeled, the neck
        # is considered to be labeled.
        formatted_joint_labels[~both_sho_labeled, neck_idx, 2] = JointVisibility.NL_FOR_PERSON
        formatted_joint_labels[both_sho_labeled, neck_idx, 2] = np.minimum(
            joint_labels[both_sho_labeled, rsho_idx, 2],
            joint_labels[both_sho_labeled, lsho_idx, 2]
        )

        return formatted_joint_labels
