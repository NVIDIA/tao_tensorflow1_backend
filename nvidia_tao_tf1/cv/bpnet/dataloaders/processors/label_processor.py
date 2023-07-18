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

"""Label transformer for BpNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

from nvidia_tao_tf1.cv.bpnet.dataloaders.pose_config import JointVisibility


logger = logging.getLogger(__name__)


class LabelProcessor(object):
    """Label Processor class.

    This sets up the labels so it can be consumed for training. It transforms the keypoint
    labels coming from the dataset into heatmaps (gaussian heatblobs) and part affinity maps
    (vector fields showing the limb connections).
    """

    HEATMAP_THRESHOLD = 0.001

    def __init__(self,
                 pose_config,
                 image_shape,
                 target_shape,
                 paf_gaussian_sigma=0.03,
                 heatmap_gaussian_sigma=0.15,
                 paf_ortho_dist_thresh=1.0):
        """__init__ method.

        Args:
            pose_config (PoseConfig)
            image_shape (list): List containing the shape of input image. [H, W, C]
            target_shape (list): List containing the shape of target labels. [H, W]
            paf_gaussian_sigma (float): Sigma value to be used for gaussian weighting
                of vector fields.
            heatmap_gaussian_sigma (float): Sigma value to be used for gaussian weighting
                of heatblobs.
            paf_ortho_dist_thresh (float): Orthogonal distance threshold for part affinity
                fields. Note that this will be multiplied by the stride.
        """

        self.TARGET_WIDTH = target_shape[1]
        self.TARGET_HEIGHT = target_shape[0]

        self.IMAGE_WIDTH = image_shape[1]
        self.IMAGE_HEIGHT = image_shape[0]

        # Get the stride of the network using the target and image shapes
        self.STRIDE = image_shape[0] // target_shape[0]
        assert (image_shape[0] // target_shape[0]) == (image_shape[1] // target_shape[1])

        # Get the heatmap exponential factor used for adding gaussian heatblobs
        self.HEATMAP_EXP_FACTOR = 1 / (2.0 * heatmap_gaussian_sigma * heatmap_gaussian_sigma)

        # Get the paf exponential factor used for adding part affinity fields
        # TODO: (This is currently not being used) Implement a exp weighted
        # pafmap instead of a binary mask based on orthgonal distance.
        self.PAF_EXP_FACTOR = 1 / (2.0 * paf_gaussian_sigma * paf_gaussian_sigma)

        # Get the orthogonal distance threshold for part affinity fields.
        self.PAF_ORTHO_DIST_THRESH = paf_ortho_dist_thresh * self.STRIDE

        # Get the slice indices for paf, heatmap and background within the label tensor
        self.pose_config = pose_config
        self.paf_slice_indices = self.pose_config.label_slice_indices['paf']
        self.heatmap_slice_indices = self.pose_config.label_slice_indices['heatmap']
        self.background_slice_indices = self.pose_config.label_slice_indices['background']

        # Intiliaze the vectors and meshgrid mapping target space to input space.
        self.x_vec, self.y_vec, \
            self.mgrid_x, self.mgrid_y = self._initialize_grids()

    def _initialize_grids(self):
        """Initialize the grid locations mapping indices in target labels to the input space.

        Returns:
            x_vec (np.ndarray): x indices mapped from target space to input space
            y_vec (np.ndarray): y indices mapped from target space to input space
            mgrid_x (np.ndarray): row repeated matrix returned by meshgrid
            mgrid_y (np.ndarray): column repeated matrix returned by meshgrid
        """

        # Vector of indices mapping from target space to input space (spacing equal to stride)
        x_vec = np.arange(self.TARGET_WIDTH) * self.STRIDE
        y_vec = np.arange(self.TARGET_HEIGHT) * self.STRIDE

        # Shifting to center indices of the input strides from top left indices
        x_vec = x_vec + (self.STRIDE - 1) / 2
        y_vec = y_vec + (self.STRIDE - 1) / 2

        # NOTE (Different from prev implementation):
        # Here we use centered grid locations as distaces for adding the part affinity fields
        # and heatmaps, whereas earlier we used top left locations for paf.
        # mgrid_y, mgrid_x = np.meshgrid(y_vec, x_vec, indexing='ij')
        # NEW NOTE: Switching to top-left location instead of center of grid temporarily (FIXME)
        mgrid_y, mgrid_x = np.mgrid[0:self.IMAGE_HEIGHT:self.STRIDE, 0:self.IMAGE_WIDTH:self.STRIDE]

        return x_vec, y_vec, mgrid_x, mgrid_y

    def transform_labels(self, joint_labels):
        """Transform the joint labels into final label tensors that can be used for training.

        Consists of joint heatmaps, background heatmaps and part affinity maps.

        Args:
            joint_labels (np.ndarray): Ground truth keypoint annotations with shape
                (num_persons, num_joints, 3).

        Returns:
            labels_arr (np.ndarray): Final label tensors used for training with
                shape (TARGET_HEIGHT, TARGET_WIDTH, num_channels).
        """
        # Initialize labels array
        labels_arr = np.zeros(self.pose_config.label_tensor_shape, dtype=np.float)

        # Check if the number of parts in the config match the joint labels
        assert self.pose_config.num_parts == joint_labels.shape[1]

        # TODO: add support to have different weights for heatmaps and part affinity fields
        # for keypoints with different occlusion flags

        # Split the label array into pafmaps, heatmaps, and background
        part_affinity_fields = \
            labels_arr[:, :, self.paf_slice_indices[0]:self.paf_slice_indices[1]]
        joint_heatmaps = \
            labels_arr[:, :, self.heatmap_slice_indices[0]:self.heatmap_slice_indices[1]]

        # For each part, find the joints to use (or the valid joints) based on
        # the visiblity flag.
        selected_joints_list = []
        for jidx in range(self.pose_config.num_parts):
            selected_joints = joint_labels[:, jidx, 2] == JointVisibility.L_VISIBLE
            selected_joints = np.logical_or(
                selected_joints, joint_labels[:, jidx, 2] == JointVisibility.L_ANNOTATABLE)
            # TODO: Should handle L_OCCLUDED differently from L_VISIBLE and L_ANNOTATABLE.
            selected_joints = np.logical_or(
                selected_joints, joint_labels[:, jidx, 2] == JointVisibility.L_OCCLUDED)
            selected_joints_list.append(selected_joints)

        # TODO: Take channel-wise mask as input to this function
        # When JointVisibility is NL_IN_DATASET, the corresponding mask channels
        # needs to be masked.

        # Generate Joint heatmaps using the joint labels.
        labels_arr[:, :, self.heatmap_slice_indices[0]:self.heatmap_slice_indices[1]] = \
            self.generate_joint_heatmaps(
                joint_heatmaps,
                joint_labels,
                selected_joints_list
            )

        # Generate Background heatmap using the joint heatmaps.
        labels_arr[:, :, self.background_slice_indices[0]:self.background_slice_indices[1]] = \
            self.generate_background_heatmap(
                joint_heatmaps
            )

        # Generate Part Affinity fields using the joint labels.
        labels_arr[:, :, self.paf_slice_indices[0]:self.paf_slice_indices[1]] = \
            self.generate_part_affinity_fields(
                part_affinity_fields,
                joint_labels,
                selected_joints_list
            )

        return labels_arr

    def generate_joint_heatmaps(self, heatmaps, joint_labels, selected_joints_list):
        """Transform keypoints / joint labels into gaussian heatmaps on lable array.

        Args:
            heatmaps (np.ndarray): Array of shape (TARGET_WIDTH, TARGET_HEIGHT, num_parts)
                initialized with zeroes.
            joint_labels (np.ndarray): Ground truth keypoint annotations with shape
                (num_persons, num_joints, 3).
            selected_joints_list (list): List of Boolean masks for each part that
                represent the joints to be used for label generation.

        Returns:
            heatmaps (np.ndarray): Array of shape (TARGET_WIDTH, TARGET_HEIGHT, num_parts)
                containing the gaussian heatblobs representing the joints as peaks.
        """

        num_joints = self.pose_config.num_parts

        for jidx in range(num_joints):

            # Get labels of the people for whom the current joint (jidx) should
            # be selected (or is marked/labeled).
            joints_to_use = joint_labels[selected_joints_list[jidx], jidx, 0:2]

            # TODO: Vectorize this part of the code.
            # Iterate over the selected persons with the current joint and add
            # them to the heatmap
            for _, joint_label in enumerate(joints_to_use):

                jx, jy = joint_label
                # Compute 1-D gaussian centered around jx and jy respectively.
                gaussian1d_x = np.exp(-np.power((self.x_vec - jx), 2) * self.HEATMAP_EXP_FACTOR)
                gaussian1d_y = np.exp(-np.power((self.y_vec - jy), 2) * self.HEATMAP_EXP_FACTOR)
                # Outer product of the 1-D gaussian results in 2-D gaussian
                # centered around (jx, jy) to get the desired heatblob for
                # the current joint
                gaussian_heatblob = np.outer(gaussian1d_y, gaussian1d_x)
                # Combine the current heatblob with the heatmap of the corresponding joint
                # by computing element-wise maximum. This is how the overlapping peaks are
                # handled.
                heatmaps[:, :, jidx] = np.maximum(heatmaps[:, :, jidx], gaussian_heatblob)

            # NOTE (Different from prev implementation):
            # Clip the values that are below threshold.
            # TODO: Check if necessary to clip the max of heatmaps to 1.
            heatmaps[:, :, jidx] = np.where(
                heatmaps[:, :, jidx] < LabelProcessor.HEATMAP_THRESHOLD, 0.0, heatmaps[:, :, jidx])

        return heatmaps

    def generate_background_heatmap(self, joint_heatmaps):
        """Generate the background heatmap using joint heatmaps.

        Args:
            joint_heatmaps (np.ndarray): Array of shape (TARGET_WIDTH, TARGET_HEIGHT, num_parts)
                containing the gaussian heatblobs representing the joints as peaks.

        Returns:
            heatmaps (np.ndarray): Array of shape (TARGET_WIDTH, TARGET_HEIGHT, 1)
                containing the background heatmap representing the background area
                in the image.
        """

        # The idea is to represent everything that is not joint labels as background
        # So the joint heatmaps are inverted
        # NOTE (Different from prev implementation):
        # Clip the values so that they are between 0 and 1.
        heatmap = np.clip(1.0 - np.amax(joint_heatmaps, axis=2), 0.0, 1.0)

        return np.expand_dims(heatmap, axis=-1)

    def generate_part_affinity_fields(self, pafmaps, joint_labels, selected_joints_list):
        """Transform keypoints / joint labels into part affinity fields on lable array.

        Args:
            pafmaps (np.ndarray): Array of shape (TARGET_WIDTH, TARGET_HEIGHT, num_connections * 2)
                initialized with zeroes.
            joint_labels (np.ndarray): Ground truth keypoint annotations with shape
                (num_persons, num_joints, 3).
            selected_joints_list (list): List of Boolean masks for each part that
                represent the joints to be used for label generation.

        Returns:
            pafmaps (np.ndarray): Array of shape (TARGET_WIDTH, TARGET_HEIGHT, num_connections * 2)
                containing the part affinity vector fields representing the connections b/w joints.
        """

        num_connections = len(self.pose_config.skeleton2idx)

        # Initialize the counter to keep track of overlapping paf vectors
        counter = np.zeros((
            self.TARGET_HEIGHT, self.TARGET_WIDTH, num_connections), dtype=np.int16)

        # Iterate through the joint connections
        for cidx, (jidx_1, jidx_2) in enumerate(self.pose_config.skeleton2idx):

            # Get labels of the people for whom the current connection involving
            # joints (jidx_1, jidx_2) should be selected.
            valid_connections = selected_joints_list[jidx_1] & selected_joints_list[jidx_2]
            connections_start = joint_labels[valid_connections, jidx_1, 0:2]
            connections_end = joint_labels[valid_connections, jidx_2, 0:2]

            # Iterate over the selected persons with the current connection and add
            # them to the pafmap
            for (cstart_x, cstart_y), (cend_x, cend_y) \
                    in zip(connections_start, connections_end):

                # X and Y components of the connection
                vec_x = cend_x - cstart_x
                vec_y = cend_y - cstart_y
                conn_length = np.sqrt(np.power(vec_x, 2) + np.power(vec_y, 2))

                if conn_length == 0:
                    logger.warning("Limb length is zeo. Skipping part affinity label.")
                    continue

                # Compute the unit vectors of the connection along x and y directions
                norm_x = vec_x / conn_length
                norm_y = vec_y / conn_length

                # Compute the location of the start of the connections in the target
                # space. Ensure they are within the target image bounds.
                min_x = max(0, int(round((
                    min(cstart_x, cend_x) - self.PAF_ORTHO_DIST_THRESH) / self.STRIDE)))
                min_y = max(0, int(round((
                    min(cstart_y, cend_y) - self.PAF_ORTHO_DIST_THRESH) / self.STRIDE)))
                # Compute the location of the end of the connections in the target
                # space. Ensure they are within the target image bounds.
                max_x = min(self.TARGET_WIDTH, int(round((
                    max(cstart_x, cend_x) + self.PAF_ORTHO_DIST_THRESH) / self.STRIDE)))
                max_y = min(self.TARGET_HEIGHT, int(round((
                    max(cstart_y, cend_y) + self.PAF_ORTHO_DIST_THRESH) / self.STRIDE)))
                if max_x < 0 or max_y < 0:
                    continue

                # Crop the region of interest where the paf vectors will be added
                mgrid_y_roi = self.mgrid_y[min_y:max_y, min_x:max_x]
                mgrid_x_roi = self.mgrid_x[min_y:max_y, min_x:max_x]
                # Compute the distance matrix comprising of orthogonal distance of every
                # location in the ROI from the vector representing the connection.
                dist_matrix = np.abs(
                    ((cstart_y - mgrid_y_roi) * vec_x - (cstart_x - mgrid_x_roi) * vec_y)
                    / conn_length
                )
                # Generate the filter matrix which is a binary array representing the
                # locations where the normalized vectors are to be added. Any location
                # having an orthogonal distance greater than the threshold is masked.
                filter_matrix = np.where(dist_matrix > self.PAF_ORTHO_DIST_THRESH, 0, 1)
                # NOTE (Different from prev implementation):
                # Use of filter_matrix to mask the pixels farther from the vector
                # than the given orthogonal paf distance threshold.
                counter[min_y:max_y, min_x:max_x, cidx] += filter_matrix
                pafmaps[min_y:max_y, min_x:max_x, cidx * 2 + 0] = norm_x * filter_matrix
                pafmaps[min_y:max_y, min_x:max_x, cidx * 2 + 1] = norm_y * filter_matrix

            # TODO: Normalize/Average the PAF (otherwise longer limb gives stronger
            # absolute strength)

        return pafmaps
