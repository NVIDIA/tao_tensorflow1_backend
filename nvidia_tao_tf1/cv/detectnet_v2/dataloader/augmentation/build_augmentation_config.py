# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Build AugmentationConfig for DetectNet V2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation.augmentation_config import AugmentationConfig


def build_preprocessing_config(preprocessing_proto):
    """Build a Preprocessing object from a proto.

    Args:
        preprocessing_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.augmentation_config.
            AugmentationConfig.Preprocessing proto message).

    Returns:
        nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation_config.Preprocessing object.
    """
    return AugmentationConfig.Preprocessing(
        output_image_width=preprocessing_proto.output_image_width,
        output_image_height=preprocessing_proto.output_image_height,
        crop_left=preprocessing_proto.crop_left,
        crop_top=preprocessing_proto.crop_top,
        crop_right=preprocessing_proto.crop_right,
        crop_bottom=preprocessing_proto.crop_bottom,
        min_bbox_width=preprocessing_proto.min_bbox_width,
        min_bbox_height=preprocessing_proto.min_bbox_height,
        scale_width=preprocessing_proto.scale_width,
        scale_height=preprocessing_proto.scale_height,
        output_image_channel=preprocessing_proto.output_image_channel,
        output_image_min=preprocessing_proto.output_image_min,
        output_image_max=preprocessing_proto.output_image_max
    )


def build_spatial_augmentation_config(spatial_augmentation_proto):
    """Build a SpatialAugmentation object from a proto.

    Args:
        spatial_augmentation_proto (nvidia_tao_tf1.cv.detectnet_v2.dataloader.proto.augmentation_config.
            AugmentationConfig.SpatialAugmentation proto message).

    Returns:
        nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation_config.SpatialAugmentation object.
    """
    return AugmentationConfig.SpatialAugmentation(
        hflip_probability=spatial_augmentation_proto.hflip_probability,
        vflip_probability=spatial_augmentation_proto.vflip_probability,
        zoom_min=spatial_augmentation_proto.zoom_min,
        zoom_max=spatial_augmentation_proto.zoom_max,
        translate_max_x=spatial_augmentation_proto.translate_max_x,
        translate_max_y=spatial_augmentation_proto.translate_max_y,
        rotate_rad_max=spatial_augmentation_proto.rotate_rad_max,
        rotate_probability=spatial_augmentation_proto.rotate_probability,
    )


def build_color_augmentation_config(color_augmentation_proto):
    """Build a ColorAugmentation object from a proto.

    Args:
        color_augmentation_proto (nvidia_tao_tf1.cv.detectnet_v2.dataloader.proto.augmentation_config.
            AugmentationConfig.ColorAugmentation proto message).

    Returns:
        nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation_config.ColorAugmentation object.
    """
    return AugmentationConfig.ColorAugmentation(
        color_shift_stddev=color_augmentation_proto.color_shift_stddev,
        hue_rotation_max=color_augmentation_proto.hue_rotation_max,
        saturation_shift_max=color_augmentation_proto.saturation_shift_max,
        contrast_scale_max=color_augmentation_proto.contrast_scale_max,
        contrast_center=color_augmentation_proto.contrast_center
    )


def build_augmentation_config(augmentation_proto):
    """Build an AugmentationConfig object from a proto.

    Args:
        augmentation_proto (nvidia_tao_tf1.cv.detectnet_v2.dataloader.proto.augmentation_config.
            AugmentationConfig proto message).

    Returns:
        nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation_config.AugmentationConfig object.
    """
    preprocessing = build_preprocessing_config(augmentation_proto.preprocessing)
    spatial_augmentation = \
        build_spatial_augmentation_config(augmentation_proto.spatial_augmentation)
    color_augmentation = build_color_augmentation_config(augmentation_proto.color_augmentation)

    return AugmentationConfig(
        preprocessing=preprocessing,
        spatial_augmentation=spatial_augmentation,
        color_augmentation=color_augmentation
    )
