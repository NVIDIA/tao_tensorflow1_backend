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

"""Processor for applying augmentations and other transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core.coreobject import TAOObject
from nvidia_tao_tf1.cv.bpnet.dataloaders.processors.augmentation_utils import \
    AugmentationModes, get_spatial_transformation_matrix_np


class AugmentationConfig(TAOObject):
    """Class to hold and process all Augmentation related parameters."""

    # Please refer to `AugmentationModes` definition for details about each
    SUPPORTED_MODES = [
        AugmentationModes.PERSON_CENTRIC,
        AugmentationModes.STANDARD,
        AugmentationModes.STANDARD_WITH_FIXED_ASPECT_RATIO
    ]

    @tao_core.coreobject.save_args
    def __init__(self,
                 spatial_aug_params,
                 color_aug_params=None,
                 spatial_augmentation_mode=None,
                 **kwargs):
        """__init__ method.

        Args:
            aug_params (dict): Spatial transformations param ranges
            identity_aug_params (dict): Spatial transformations param ranges to
                use when augmentation is disabled.
            image_shape (list): List containing the shape of input image. [H, W, C]
            target_shape (list): List containing the shape of target labels. [H, W]
            pose_config (BpNetPoseConfig):
            augmentation_mode (AugmentationModes): Augmentation mode to apply for the
                images. Refer to the Enum class for details about the augmentation modes.
        """
        self.color_aug_params = color_aug_params
        self.spatial_aug_params = spatial_aug_params

        # Build the spatial and color augmentaion parameters
        # This will update the default parameters with the ones specified
        # by the user.
        self.spatial_aug_params, self.color_aug_params = \
            self.build_augmentation_config(spatial_aug_params, color_aug_params)

        # Build the params for identity transformation (no augmentation)
        self.identity_spatial_aug_params, self.identity_color_aug_params = \
            self.build_augmentation_config()

        # If the augmentation mode is not specified, use default
        self.spatial_augmentation_mode = spatial_augmentation_mode
        if spatial_augmentation_mode is None:
            self.spatial_augmentation_mode = AugmentationModes.PERSON_CENTRIC

    @staticmethod
    def get_default_augmentation_config():
        """Get the default the augmentation config.

        Returns:
            spatial_aug_params_default (dict): default params to use for spatial transformations
            color_aug_params_default (dict): default params to use for color transformations
        """

        spatial_aug_params_default = {
            'flip_lr_prob': 0.0,
            'flip_tb_prob': 0.0,
            'rotate_deg_max': 0.0,
            'rotate_deg_min': None,
            'zoom_prob': 0.0,
            'zoom_ratio_min': 1.0,
            'zoom_ratio_max': 1.0,
            'translate_max_x': 0.0,
            'translate_min_x': None,
            'translate_max_y': 0.0,
            'translate_min_y': None,
            'use_translate_ratio': False,
            'translate_ratio_max': 0.0,
            'translate_ratio_min': 0.0,
            'target_person_scale': 0.6
        }

        # TODO (color augmentations not implemented yet)
        color_aug_params_default = {}

        return spatial_aug_params_default, color_aug_params_default

    @staticmethod
    def build_augmentation_config(spatial_aug_params=None, color_aug_params=None):
        """Builds a default augmentation dict and updates with the user provided params.

        Args:
            spatial_aug_params (dict): User provided params for spatial transformations
            color_aug_params (dict): User provided params for color transformations

        Returns:
            spatial_aug_params_updated (dict): Updated spatial transformations params
            color_aug_params_updated (dict): Updated color transformations params
        """

        # Get the default spatial and color augmentation parameters
        spatial_aug_params_default, color_aug_params_default = \
            AugmentationConfig.get_default_augmentation_config()

        def _update(d, u):
            """Update the dictionaries.

            Args:
                d (dict): Dictionary to be updated.
                u (dict): Dictionary that is used to update.

            Returns:
                d (dict): Updated dictionary.
            """

            if u is not None:
                d.update(u)
            return d

        sparams_updated = _update(spatial_aug_params_default, spatial_aug_params)
        cparams_updated = _update(color_aug_params_default, color_aug_params)

        # If min of the translation range is None, use a range symmetrix about the center.
        if sparams_updated['translate_min_x'] is None:
            sparams_updated['translate_min_x'] = -sparams_updated['translate_max_x']

        if sparams_updated['translate_min_y'] is None:
            sparams_updated['translate_min_y'] = -sparams_updated['translate_max_y']

        # If min of the rotation range is None, use a range symmetrix about 0 deg.
        if sparams_updated['rotate_deg_min'] is None:
            sparams_updated['rotate_deg_min'] = -sparams_updated['rotate_deg_max']

        return sparams_updated, cparams_updated

    @staticmethod
    def generate_random_spatial_aug_params(aug_params, image_height, image_width):
        """Generates final params to be used for spatial augmentation using the provided ranges.

        Args:
            aug_params (dict): Spatial transformations param ranges

        Returns:
            final_params (dict): Final spatial transformations params to be
                used for the current sample.
        """

        # Determine whether to flip the image in this iteration
        flip_lr_flag = np.less(np.random.uniform(0.0, 1.0), aug_params['flip_lr_prob'])
        flip_tb_flag = np.less(np.random.uniform(0.0, 1.0), aug_params['flip_tb_prob'])

        # Determine the random translation along x and y in this iteration.
        # Check whether to use translation range as ratio of image or absolute values
        if aug_params['use_translate_ratio']:
            if aug_params['translate_ratio_min'] is None:
                aug_params['translate_ratio_min'] = -aug_params['translate_ratio_max']
            translate_x = np.random.uniform(
                aug_params['translate_ratio_min'], aug_params['translate_ratio_max']) * image_width
            translate_y = np.random.uniform(
                aug_params['translate_ratio_min'], aug_params['translate_ratio_max']) * image_height
        else:
            translate_x = np.random.random_integers(
                aug_params['translate_min_x'], aug_params['translate_max_x'])
            translate_y = np.random.random_integers(
                aug_params['translate_min_y'], aug_params['translate_max_y'])

        # Determine the random scale/zoom
        zoom_flag = np.less(np.random.uniform(0.0, 1.0), aug_params['zoom_prob'])
        zoom_ratio = np.random.uniform(
            aug_params['zoom_ratio_min'], aug_params['zoom_ratio_max']
        ) if zoom_flag else 1.0

        # Determine the random rotation angle based on the specified range.
        # If min of the range is None, use a range symetric about 0 deg.
        if aug_params['rotate_deg_min'] is None:
            aug_params['rotate_deg_min'] = -aug_params['rotate_deg_max']
        rotate_deg = np.random.uniform(aug_params['rotate_deg_min'], aug_params['rotate_deg_max'])
        rotate_rad = np.radians(rotate_deg)

        # Build the final params
        final_params = {
            'flip_lr': flip_lr_flag,
            'flip_tb': flip_tb_flag,
            'translate_x': translate_x,
            'translate_y': translate_y,
            'zoom_ratio': zoom_ratio,
            'rotate_rad': rotate_rad
        }

        return final_params


class BpNetSpatialTransformer(object):
    """Processor that obtains the spatial transformation matrix and applies to kpts."""

    def __init__(self,
                 aug_params,
                 identity_aug_params,
                 image_shape,
                 pose_config,
                 augmentation_mode=None,
                 **kwargs):
        """__init__ method.

        Args:
            aug_params (dict): Spatial transformations param ranges
            identity_aug_params (dict): Spatial transformations param ranges to
                use when augmentation is disabled.
            image_shape (list): List containing the shape of input image. [H, W, C]
            pose_config (BpNetPoseConfig):
            augmentation_mode (AugmentationModes): Augmentation mode to apply for the
                images. Refer to the Enum class for details about the augmentation modes.
        """
        self.IMAGE_WIDTH = image_shape[1]
        self.IMAGE_HEIGHT = image_shape[0]

        self.pose_config = pose_config
        self.aug_params = aug_params
        self.identity_aug_params = identity_aug_params
        self.augmentation_mode = augmentation_mode

        # Check if the augmentation mode within the supported modes
        if augmentation_mode not in AugmentationConfig.SUPPORTED_MODES:
            raise "Mode must be one of {}.".format(AugmentationConfig.SUPPORTED_MODES)

    def call(self, image, joint_labels, scales, centers, enable_aug=True):
        """Obtains transormation matrix and transforms labels using the given parameters.

        Args:
            joint_labels (np.ndarray): Ground truth keypoint annotations with shape
                (num_persons, num_joints, 3).
            scales (np.ndarray): Contains the bbox widths of all labeled persons
                in the image.
            centers (np.ndarray): Contains the center of bbox of all labeled persons
                in the image.
            enable_aug (bool): Boolean flag to toggle augmentation.

        Returns:
            joint_labels (np.ndarray): Output keypoints after spatial transformation.
        """

        # Get image shape
        image_height, image_width = image.shape[0:2]

        # Get the final augmentation params to build the transformation matrix
        if enable_aug:
            # If augmentation is enabled, generate final params using the random param ranges.
            random_aug_params = \
                AugmentationConfig.generate_random_spatial_aug_params(
                    self.aug_params, image_height, image_width)
        else:
            # If augmentation is disabled, generate final params using param ranges
            # for identity augmentation.
            random_aug_params = \
                AugmentationConfig.generate_random_spatial_aug_params(
                    self.identity_aug_params, image_height, image_width)

        # Update the config according to the augmentation mode
        if self.augmentation_mode == AugmentationModes.PERSON_CENTRIC:
            random_aug_params = \
                self._update_with_person_centric_params(scales, centers, random_aug_params)

        # Get the spatial transformation matrix
        stm = get_spatial_transformation_matrix_np(
                **random_aug_params,
                image_width=image_width,
                image_height=image_height,
                target_width=self.IMAGE_WIDTH,
                target_height=self.IMAGE_HEIGHT,
                augmentation_mode=self.augmentation_mode
            )

        # Transform keypoints
        joint_labels = self.apply_affine_transform_kpts(
            joint_labels, stm, self.pose_config, random_aug_params['flip_lr'])

        # NOTE: This is needed as the spatial transformer in modulus expects
        # transposed version of the matrix expected by cv2.warpAffine
        stm = np.transpose(stm)

        return joint_labels, stm

    def _update_with_person_centric_params(self, scales, centers, random_aug_params):
        """Update the spatial augmentation params with PERSON_CENTRIC params.

        This ensures that the augmentations are about the person of interest in the
        image. So the POI is centered about the image and the scaling is adjusted
        such that the POI is approximately scale normalized with respect to the image.
        NOTE: The way the data is generated for this mode of augmentation, each image
        is repeated `n` times where `n` is the number of persons of interest. Hence,
        every time, the image is augmented about each of these POIs. Please refer to
        `dataio` to understand how the data is generated for this mode.

        Args:
            scales (np.ndarray): Contains the bbox widths of all labeled persons
                in the image.
            centers (np.ndarray): Contains the center of bbox of all labeled persons
                in the image.
            random_aug_params (dict): Params drawn from a uniform distribution out of
                the specified ranges in the config.
        Returns:
            random_aug_params (dict): Updated `random_aug_params` considering the person
                of interest meta data.
        """
        # Update the random zoom ratio by considering the desired scale
        # of the person of interest (poi).
        poi_width = scales[0]
        poi_width_ratio = poi_width / self.IMAGE_WIDTH
        random_zoom_ratio = random_aug_params['zoom_ratio']
        # TODO: Check in what case the denominator is zero.
        zoom_ratio = \
            self.aug_params['target_person_scale'] / (poi_width_ratio * random_zoom_ratio + 0.00001)

        # Update the translation considering the center of the person of interest
        poi_center = centers[0]
        translate_x = -(poi_center[0] + random_aug_params['translate_x'])
        translate_y = -(poi_center[1] + random_aug_params['translate_y'])

        random_aug_params['zoom_ratio'] = zoom_ratio
        random_aug_params['translate_x'] = translate_x
        random_aug_params['translate_y'] = translate_y

        return random_aug_params

    @staticmethod
    def apply_affine_transform_image(image, size, affine_transform, border_value):
        """Apply affine transformation to the image given the transformation matrix.

        Args:
            image (np.ndarray): Input image to be augmented.
            size (tuple): Destination image size (height, width)
            affine_transform (np.ndarray): Spatial transformation matrix (stm)
                that is used for image augmentation
            border_value (tuple/int): Value used in case of a constant border

        Returns:
            result (np.ndarray): Transformed image.
        """

        return cv2.warpAffine(
            image, affine_transform[0:2], size, flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
        )

    @staticmethod
    def apply_affine_transform_kpts(keypoints, affine_transform, pose_config, flip_lr_flag):
        """Apply affine transformation to the keypoints given the transformation matrix.

        Args:
            keypoints (np.ndarray): Input keypoints to be tranformed.
            affine_transform (np.ndarray): Spatial transformation matrix (stm)
                that is used for the image augmentation.
            pose_config (BpNetPoseConfig): Needed to get the pairs of joints
                that needs to be swaped in case of flipped image.
            flip_lr_flag (bool): Indicates whether the image was flipped

        Returns:
            keypoints (np.ndarray): Transformed keypoints.
        """
        # Convert the keypoints to homogeneous coordinates
        num_persons, num_joints, _ = keypoints.shape

        homogeneous_coors = np.concatenate((
            keypoints[:, :, 0:2], np.ones((num_persons, num_joints, 1))), axis=-1)

        # Project the ground turth keypoints to the new space using the affine transformation
        transformated_coords = np.matmul(
            affine_transform[0:2], homogeneous_coors.transpose([0, 2, 1])
        )

        # Copy over the transformed keypoints back to the `keypoints` array to retain
        # the visibility flags
        keypoints[:, :, 0:2] = transformated_coords.transpose([0, 2, 1])

        # If image was flipped, swap the left/right joint pairs accordingly
        if flip_lr_flag:
            temp_left_kpts = keypoints[:, pose_config.flip_pairs[:, 0], :]
            temp_right_kpts = keypoints[:, pose_config.flip_pairs[:, 1], :]
            keypoints[:, pose_config.flip_pairs[:, 0], :] = temp_right_kpts
            keypoints[:, pose_config.flip_pairs[:, 1], :] = temp_left_kpts

        return keypoints


class BpNetColorTranformer(object):
    """Processor that transforms images using a given color transformation matrix."""

    # TODO: Implement the Color transformer module.
    pass
