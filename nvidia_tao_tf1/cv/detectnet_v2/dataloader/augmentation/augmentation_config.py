# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Augmentation config classes for DetectNet V2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class AugmentationConfig(object):
    """Hold all preprocessing and augmentation related parameters."""

    def __init__(self,
                 preprocessing,
                 spatial_augmentation,
                 color_augmentation):
        """Constructor.

        Args:
            preprocessing (AugmentationConfig.Preprocessing):
            spatial_augmentation (AugmentationConfig.SpatialAugmentation):
            color_augmentation (AugmentationConfig.ColorAugmentation):
        """
        self.preprocessing = preprocessing
        self.spatial_augmentation = spatial_augmentation
        self.color_augmentation = color_augmentation

    # Define inner classes.
    class Preprocessing(object):
        """Hold the preprocessing parameters."""

        def __init__(self,
                     output_image_width,
                     output_image_height,
                     output_image_channel,
                     output_image_min,
                     output_image_max,
                     crop_left,
                     crop_top,
                     crop_right,
                     crop_bottom,
                     min_bbox_width,
                     min_bbox_height,
                     scale_width,
                     scale_height):
            """Constructor.

            Args:
                output_image_width/height (int): Dimensions of the image after preprocessing has
                    been applied.
                output_image_min/max (int): Image smaller side size and larger side size
                    in dynamic shape case.
                crop_left/top (int): How much top crop from the left/top vertex of an image.
                crop_right/bottom (int): How much to crop to. e.g. If the full image is 1000 pixels
                    in width, but we want only up till the 900th pixel, <crop_right> should be set
                    to 900.
                min_bbox_width/height (float): Minimum bbox dimensions.
                scale_width/height (float): Scaling factor for resizing width/height after crop.

            Raises:
                ValueError: if the provided values are not in expected ranges.
            """
            if output_image_min == 0:
                if output_image_width <= 0 or output_image_height <= 0:
                    raise ValueError("Preprocessing.output_image_height/width must be > 0.")
            if crop_left > crop_right or crop_top > crop_bottom:
                raise ValueError("Preprocessing crop parameters lead to null output dim(s).")
            if scale_width < 0 or scale_height < 0:
                raise ValueError("Preprocessing.scale_width/height must be positive.")
            self.output_image_width = output_image_width
            self.output_image_height = output_image_height
            self.crop_left = crop_left
            self.crop_top = crop_top
            self.crop_right = crop_right
            self.crop_bottom = crop_bottom
            self.min_bbox_width = min_bbox_width
            self.min_bbox_height = min_bbox_height
            self.scale_width = scale_width
            self.scale_height = scale_height
            self.output_image_channel = output_image_channel
            self.output_image_min = output_image_min
            self.output_image_max = output_image_max

    class SpatialAugmentation(object):
        """Hold the spatial augmentation parameters."""

        def __init__(self,
                     hflip_probability,
                     vflip_probability,
                     zoom_min,
                     zoom_max,
                     translate_max_x,
                     translate_max_y,
                     rotate_rad_max,
                     rotate_probability):
            """Constructor.

            Args:
                hflip_probability (float): Probability value for flipping an image horizontally
                    (i.e. from left to right).
                vflip_probability (float): Same but for vertical axis.
                zoom_min/max (float): Minimum/maximum zoom ratios. Set min = max = 1 to keep
                    original size.
                translate_max_x/y (float): Maximum translation along the x/y axis in pixel values.

            Raises:
                ValueError: if the provided values are not in expected ranges.
            """
            if not 0.0 <= hflip_probability <= 1.0:
                raise ValueError("hflip_probability should be in [0., 1.] range.")
            if not 0.0 <= vflip_probability <= 1.0:
                raise ValueError("vflip_probability should be in [0., 1.] range.")
            if zoom_min > zoom_max:
                raise ValueError("zoom_min must be <= zoom_max.")
            if zoom_min <= 0.0:
                raise ValueError("zoom_min must be > 0.0")
            if translate_max_x < 0.0 or translate_max_y < 0.0:
                raise ValueError("translate_max_x/y must be >= 0.0.")
            self.hflip_probability = hflip_probability
            self.vflip_probability = vflip_probability
            self.zoom_min = zoom_min
            self.zoom_max = zoom_max
            self.translate_max_x = translate_max_x
            self.translate_max_y = translate_max_y
            self.rotate_rad_max = rotate_rad_max if rotate_rad_max else 0.0
            self.rotate_probability = rotate_probability if rotate_probability else 1.0

    class ColorAugmentation(object):
        """Hold the color augmentation parameters."""

        def __init__(self,
                     color_shift_stddev,
                     hue_rotation_max,
                     saturation_shift_max,
                     contrast_scale_max,
                     contrast_center):
            """Constructor.

            Args:
                color_shift_stddev (float): Standard deviation for color shift augmentation.
                hue_rotation_max (float): Maximum hue rotation, in degrees.
                saturation_shift_max (float): Maximum value for saturation shift.
                constrast_scale_max (float): Maximum scale shift for contrast augmentation. Set to
                    0.0 to disable.
                contrast_center (float): Center point for contrast augmentation. Set to 0.5 to
                    disable.
            """
            self.color_shift_stddev = color_shift_stddev
            self.hue_rotation_max = hue_rotation_max
            self.saturation_shift_max = saturation_shift_max
            self.contrast_scale_max = contrast_scale_max
            self.contrast_center = contrast_center
