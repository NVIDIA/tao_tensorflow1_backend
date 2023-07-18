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

"""Pose config class for BpNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
import json

import numpy as np

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core.coreobject import TAOObject


class Joints(str, Enum):
    """Enum class containing the superset of body joints."""

    NOSE = "nose"
    NECK = "neck"
    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_WRIST = "right_wrist"
    LEFT_SHOULDER = "left_shoulder"
    LEFT_ELBOW = "left_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_HIP = "right_hip"
    RIGHT_KNEE = "right_knee"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HIP = "left_hip"
    LEFT_KNEE = "left_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_EYE = "right_eye"
    LEFT_EYE = "left_eye"
    RIGHT_EAR = "right_ear"
    LEFT_EAR = "left_ear"


class JointVisibility:
    """Class containing the visiblity flags of body joints."""

    # labeled and visible
    L_VISIBLE = 0
    # labeled and occluded,
    # but easy to annotate with good accuracy
    L_ANNOTATABLE = 1
    # labeled and occluded,
    # harder to annotate with good accuracy
    L_OCCLUDED = 2
    # truncated
    L_TRUNCATED = 3
    # not labeled for this person
    NL_FOR_PERSON = 4
    # not labeled in this dataset
    NL_IN_DATASET = 5
    # Define str->var mapping
    map_visibiltiy_str = {
        'visible': L_VISIBLE,
        'annotatable': L_ANNOTATABLE,
        'occluded': L_OCCLUDED,
        'truncated': L_TRUNCATED,
        'not_labeled': NL_FOR_PERSON,
        'not_labeled_in_dataset': NL_IN_DATASET
    }


class BpNetPoseConfig(TAOObject):
    """Class to hold all BpNet pose related parameters."""

    BACKGROUND = "background"

    @tao_core.coreobject.save_args
    def __init__(self,
                 target_shape,
                 pose_config_path,
                 **kwargs):
        """Constructor.

        Args:
            target_shape (list): List containing the dimensions of target label [height, width]
            pose_config_path (string): Absolute path to the pose config file.
        """

        # Load the pose config file
        self.pose_config_path = pose_config_path
        self.pose_config = self._load_pose_config(pose_config_path)

        # Get the person category config
        self.person_config = self._get_category(self.pose_config, 'person')

        # Get number of joints and edges in the skeleton and assert correctness
        self.num_parts = self.person_config["num_joints"]
        self.parts = self.person_config["keypoints"]
        self.parts2idx = dict(zip(self.parts, range(self.num_parts)))
        assert self.num_parts == len(self.parts)

        self.skeleton = self.person_config["skeleton_edge_names"]
        # The skeleton is 1- indexed. So convert them to 0- indexed list.
        self.skeleton2idx = [[jidx_1 - 1, jidx_2 - 1]
                             for (jidx_1, jidx_2) in self.person_config["skeleton"]]
        self.num_connections = len(self.skeleton)

        # Get left and right joints
        # If image was flipped, swap the left/right joint pairs accordingly
        self._left_parts = [self.parts2idx[part] for part in self.parts if 'left_' in part]
        self._right_parts = [self.parts2idx[part] for part in self.parts if 'right_' in part]
        self.flip_pairs = np.array(
            [[left_part, right_part]
             for left_part, right_part in zip(self._left_parts, self._right_parts)]
        )
        # TODO: assert flip pairs are correctly matched. The part name excluding the left/right
        # should be the same.

        # Add background layer
        self.parts += [self.BACKGROUND]
        self.num_parts_with_background = len(self.parts)

        self.num_paf_channels = 2 * self.num_connections
        self.num_heatmap_channels = self.num_parts_with_background
        self.num_total_channels = self.num_paf_channels + self.num_heatmap_channels

        self.label_slice_indices = {
            'paf': [0, self.num_paf_channels],
            'heatmap': [self.num_paf_channels,
                        self.num_paf_channels + self.num_parts],
            'heatmap_with_background': [self.num_paf_channels,
                                        self.num_paf_channels + self.num_parts_with_background],
            'background': [self.num_paf_channels + self.num_parts,
                           self.num_paf_channels + self.num_parts_with_background]
        }

        # Final label tensor shape
        self.label_tensor_shape = (target_shape[0], target_shape[1], self.num_total_channels)

        # Topology
        self.topology = self.pose_config_to_topology(self.pose_config, 'person')

    @staticmethod
    def _get_category(data, cat_name):
        """Get the configuration corresponding to the given category name."""

        return [c for c in data['categories'] if c['name'] == cat_name][0]

    @staticmethod
    def _load_pose_config(pose_config_path):
        """Load the pose config based on the pose config typpe provided.

        Args:
            pose_config_path (string): Absolute path to the pose config file.
            This determines the skeleton structure and joints to use.

        Returns:
            result (dict): Dictionary containing the bpnet pose params to use.
        """
        if pose_config_path is None:
            raise ValueError("Enter a valid path for the pose config.")

        return json.load(open(pose_config_path))

    @staticmethod
    def pose_config_to_topology(pose_config, category):
        """Gets topology tensor from bpnet pose config.

        Args:
            pose_config (dict): Dictionary containing the bpnet pose params to use.
        """
        category_config = BpNetPoseConfig._get_category(pose_config, category)
        skeleton = category_config['skeleton']
        K = len(skeleton)
        topology = np.zeros((K, 4), dtype=np.int)
        for k in range(K):
            topology[k][0] = 2 * k
            topology[k][1] = 2 * k + 1
            topology[k][2] = skeleton[k][0] - 1
            topology[k][3] = skeleton[k][1] - 1
        return topology
