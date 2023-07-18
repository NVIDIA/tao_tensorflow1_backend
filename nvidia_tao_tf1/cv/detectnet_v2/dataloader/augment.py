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

"""Data augmentation for DetectNet V2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.processors import ColorTransform
from nvidia_tao_tf1.core.processors import SpatialTransform
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.augment.color import get_random_color_transformation_matrix
from nvidia_tao_tf1.core.processors.augment.spatial import get_random_spatial_transformation_matrix
import tensorflow as tf


def _matrix3(values):
    return tf.reshape(tf.stack(tf.cast(values, tf.float32)), [3, 3])


def _matrix4(values):
    return tf.reshape(tf.stack(tf.cast(values, tf.float32)), [4, 4])


def _crop_image(crop_left, crop_top, crop_right, crop_bottom):
    """
    Compute spatial transformation matrix and output image size for cropping.

    All the crop positions specified are in the original image space.
    Args:
      crop_left: (int) left boundary of the cropped part.
      crop_top: (int) top boundary of the cropped part.
      crop_right: (int) right boundary of the cropped part.
      crop_bottom: (int) bottom boundary of the cropped part.

    Returns:
      stm: (tf.Tensor) spatial transformation matrix
      output_width: (int) width of the resulting cropped image
      output_height: (int) height of the resulting cropped image
    """
    # Compute spatial transformation matrix (translation).
    crop_matrix = _matrix3([1., 0., 0.,
                            0., 1., 0.,
                            -crop_left, -crop_top, 1.])

    # Compute output image dimensions.
    output_width = crop_right - crop_left
    output_height = crop_bottom - crop_top

    return crop_matrix, output_width, output_height


def get_transformation_ops():
    """Generate ops which will apply spatial / color transformations.

    Returns:
        stm_op: spatial transformation op.
        ctm_op: color transformation op.
    """
    # Set up spatial transform op.
    stm_op = SpatialTransform(method='bilinear', background_value=0.0, data_format='channels_last')

    # Set up color transform op.
    # Note that its output is always normalized to [0,1] range.
    ctm_op = ColorTransform(min_clip=0.0, max_clip=1.0, data_format='channels_last')

    return stm_op, ctm_op


def get_spatial_preprocessing_matrix(crop_left, crop_top, crop_right, crop_bottom,
                                     scale_width, scale_height):
    """Generate a spatial preprocessing matrix.

    Args:
        crop_left (int): Left boundary of a crop to extract in original image space.
        crop_top (int): Top boundary of a crop to extract in original image space.
        crop_right (int): Right boundary of a crop to extract in original image space.
        crop_bottom (int): Bottom boundary of a crop to extract in original image space.
        scale_width (float): Resize factor used for scaling width.
        scale_height (float): Resize factor used for scaling height.

    Returns:
        sm (tf.Tensor): matrix that transforms from original image space to augmented space.
    """
    # Start from an identity spatial transformation matrix.
    sm = tf.eye(3)

    # Cropping happens if the crop rectangle has width > 0 and height > 0. Otherwise
    # cropping is considered to be disabled.
    if crop_right > crop_left and crop_bottom > crop_top:
        crop_matrix, _, _ = _crop_image(crop_left, crop_top, crop_right, crop_bottom)
        sm = crop_matrix

    # Image is resized if valid scale factors are provided for width and height.
    if scale_width > 0 and scale_height > 0:
        scale_stm = spatial.zoom_matrix((scale_width, scale_height))
        sm = tf.matmul(scale_stm, sm)

    return sm


def get_spatial_transformations_matrix(preprocessing_config,
                                       spatial_augmentation_config):
    """
    Generate a spatial transformations matrix that applies both preprocessing and augmentations.

    Args:
        preprocessing_config: AugmentationConfig.Preprocessing object.
        spatial_augmentation_config: AugmentationConfig.SpatialAugmentation object. Can be None,
            which disables random spatial augmentations.

    Returns:
        sm: matrix that transforms from original image space to augmented space.
    """
    width = preprocessing_config.output_image_width
    height = preprocessing_config.output_image_height

    # Get spatial transformation matrix corresponding to preprocessing ops.
    sm = get_spatial_preprocessing_matrix(
        crop_left=preprocessing_config.crop_left,
        crop_top=preprocessing_config.crop_top,
        crop_right=preprocessing_config.crop_right,
        crop_bottom=preprocessing_config.crop_bottom,
        scale_width=preprocessing_config.scale_width,
        scale_height=preprocessing_config.scale_height)
    # Apply resizing and spatial augmentation ops to spatial transformation matrix.
    if spatial_augmentation_config is not None:
        sm = get_random_spatial_transformation_matrix(
                width, height,
                stm=sm,
                flip_lr_prob=spatial_augmentation_config.hflip_probability,
                translate_max_x=int(spatial_augmentation_config.translate_max_x),
                translate_max_y=int(spatial_augmentation_config.translate_max_y),
                zoom_ratio_min=spatial_augmentation_config.zoom_min,
                zoom_ratio_max=spatial_augmentation_config.zoom_max,
                rotate_rad_max=spatial_augmentation_config.rotate_rad_max)

    return sm


def get_color_augmentation_matrix(color_augmentation_config):
    """
    Generate a color transformations matrix applying augmentations.

    Args:
        color_augmentation_config: AugmentationConfig.ColorAugmentation object. Can be None, which
            disables random color transformations.

    Returns:
        Matrix describing the color transformation to be applied.
    """
    cm = tf.eye(4, dtype=tf.float32)
    if color_augmentation_config is not None:
        # Compute color transformation matrix.
        cm = get_random_color_transformation_matrix(
                ctm=cm,
                hue_rotation_max=color_augmentation_config.hue_rotation_max,
                saturation_shift_max=color_augmentation_config.saturation_shift_max,
                contrast_scale_max=color_augmentation_config.contrast_scale_max,
                contrast_center=color_augmentation_config.contrast_center,
                brightness_scale_max=color_augmentation_config.color_shift_stddev * 2.0,
                brightness_uniform_across_channels=False)
    return cm


def get_all_transformations_matrices(augmentation_config, enable_augmentation):
    """
    Generate all the color and spatial transformations as defined in augmentation_config.

    Input image values are assumed to be in the [0, 1] range.

    Args:
        augmentation_config: AugmentationConfig object.
        enable_augmentation (bool): Toggle for enabling/disabling augmentations.

    Returns:
        sm: matrix that transforms from original image space to augmented space.
        cm: color transformation matrix.
    """
    if enable_augmentation:
        spatial_augmentation_config = augmentation_config.spatial_augmentation
        color_augmentation_config = augmentation_config.color_augmentation
    else:
        spatial_augmentation_config = None
        color_augmentation_config = None

    # Compute spatial transformation matrix (preprocessing + augmentation).
    sm = get_spatial_transformations_matrix(
        preprocessing_config=augmentation_config.preprocessing,
        spatial_augmentation_config=spatial_augmentation_config)

    # Compute color transformation matrix.
    cm = get_color_augmentation_matrix(color_augmentation_config=color_augmentation_config)
    return sm, cm


def apply_spatial_transformations_to_polygons(sm, vertices_x, vertices_y):
    """Apply spatial transformations to polygons.

    Args:
        sm: spatial transform.
        vertices_x: vector of x-coordinates.
        vertices_y: vector of y-coordinates.
    Returns:
        vectors of transformed polygon coordinates in the augmented space.
    """
    # TODO(@drendleman) Should we use the maglev PolygonTransform processor here?
    ones = tf.ones(shape=[tf.size(input=vertices_x)])
    transformed_coords = tf.transpose(
        a=tf.matmul(tf.transpose(a=[vertices_x, vertices_y, ones]), sm))

    return transformed_coords[0], transformed_coords[1]


def apply_spatial_transformations_to_bboxes(sm, x1, y1, x2, y2):
    """Apply spatial transformations to bboxes.

    Transform top-left and bottom-right bbox coordinates by the matrix and
    compute new bbox coordinates. Note that the code below assumes that the
    matrix contains just scaling, mirroring, and 90 degree rotations.
    TODO(jrasanen) do we need to allow generic transformations? In that case we'd need to
    transform all four corners of the bbox and recompute a new axis aligned
    box. This will be overly pessimistic for elliptical ground truth, so
    it would be better to compute a tight fit around a transformed ellipse.

    This will be used only for legacy dataloader if needed.

    Args:
        sm: spatial transform.
        x1: vector of left edge coordinates of the input bboxes. Coordinates
            are in input image scale.
        y1: vector of top edge coordinates of the input bboxes.
        x2: vector of right edge coordinates of the input bboxes.
        y2: vector of bottom edge coordinates of the input bboxes.
    Returns:
        vectors of transformed bbox coordinates in the augmented space.
    """
    one = tf.ones(shape=[tf.size(x1)])
    top_left = tf.transpose(tf.matmul(tf.transpose([x1, y1, one]), sm))
    bottom_right = tf.transpose(tf.matmul(tf.transpose([x2, y2, one]), sm))
    # The following lines are to be able to return the bounding box coordinates in the expected
    # L, T, R, B order when (potential) horizontal flips happen.
    x1 = tf.minimum(top_left[0], bottom_right[0])
    y1 = tf.minimum(top_left[1], bottom_right[1])
    x2 = tf.maximum(top_left[0], bottom_right[0])
    y2 = tf.maximum(top_left[1], bottom_right[1])

    return x1, y1, x2, y2


def apply_color_transformations(image, ctm_op, cm):
    """
    Apply color transformations to an image.

    Args:
        image: input image of shape (height, width, 3) with values in [0, 1].
        ctm_op: instance of ColorTransform processor.
        cm: color transform matrix.

    Returns:
        image: transformed image of shape (height, width, 3) with values in [0, 1].
    """
    # Note 1: color matrix needs to be reshaped into a batch of one 4x4 matrix
    # Note 2: input colors must be unnormalized, ie. in [0,255] range, have 3 channels
    # Note 3: output colors are always normalized to [0,1] range, 3 channels
    # TODO it would be faster to batch transform the images. Not sure if it makes a difference.
    # TODO the fastest implementation would combine spatial and color transforms into a single op.
    return ctm_op(images=image, ctms=tf.stack([cm]))


def apply_spatial_transformations_to_image(image, height, width, stm_op, sm):
    """
    Apply spatial transformations to an image.

    Spatial transform op maps destination image pixel P into source image location Q
    by matrix M: Q = P M. Here we first compute a forward mapping Q M^-1 = P, and
    finally invert the matrix.

    Args:
        image (tf.Tensor): Input image of shape (height, width, 3) with values in [0, 1].
        height (int): Height of the output image.
        width (int): Width of the output image.
        stm_op (SpatialTransform): Instance of SpatialTransform processor.
        sm (tf.Tensor): 3x3 spatial transformation matrix.

    Returns:
        image: Transformed image of shape (height, width, 3) with values in [0, 1].
        dm: Matrix that transforms from augmented space to the original image space.
    """
    dm = tf.matrix_inverse(sm)

    # Convert image to float if needed (stm_op requirement)
    if image.dtype != tf.float32:
        image = tf.cast(image, tf.float32)
    # TODO undo image gamma? Probably doesn't make a difference.

    # Apply spatial transformations
    # Note 1: spatial transform op currently only supports 3 channel input images
    # Note 2: since the input image sizes may vary, this op can't be batched
    # Note 3: image and matrix need to be reshaped into a batch of one for this op
    image = stm_op(images=tf.stack([image]), stms=tf.stack([dm]), shape=[height, width])

    return image, dm


def apply_all_transformations_to_image(height, width, stm_op, ctm_op, sm, cm, image, num_channels):
    """Apply spatial and color transformations to an image.

    Args:
        height: height of the output image.
        width: width of the output image.
        stm_op: instance of SpatialTransform processor.
        ctm_op: instance of ColorTransform processor.
        sm: spatial transform matrix.
        cm: color transform matrix.
        image: input image of shape (height, width, 3). Values are assumed to be in [0, 1].
    Returns:
        transformed image of shape (height, width, 3) with values in [0, 1].
        matrix that transforms from augmented space to the original image space.
    """
    image, dm = apply_spatial_transformations_to_image(image, height, width, stm_op, sm)
    if num_channels == 3:
        image = apply_color_transformations(image, ctm_op, cm)
    # Reshape into a single image
    return tf.reshape(image, [height, width, num_channels]), dm
