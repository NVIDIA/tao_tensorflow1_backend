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

"""Augmentation Utils."""

from enum import Enum
import math
import cv2
import numpy as np


class AugmentationModes(str, Enum):
    """Enum class containing the augmentation modes.

    1. PERSON_CENTRIC: Augementations are centered around each person in the image.
        When the train data is compiled for this mode, each image is replicated
        N times where N is the number of people in image with certain size criteria.
    2. STANDARD: Augmentations are standard, i.e centered around the center of the image
        and the aspect ratio of the image is retained.
    3. STANDARD_WITH_FIXED_ASPECT_RATIO: Augmentations are standard, i.e centered around
        the center of the image. But the aspect ratio is fixed to the network input aspect
        ratio.
    """

    PERSON_CENTRIC = "person_centric"
    STANDARD = "standard"
    STANDARD_WITH_FIXED_ASPECT_RATIO = "standard_with_fixed_aspect_ratio"


def flip_matrix_np(horizontal, vertical, width=None, height=None):
    """Construct a spatial transformation matrix that flips.

    Note that if width and height are supplied, it will move the object back into the canvas
    together with the flip.

    Args:
        horizontal (bool): If the flipping should be horizontal. Scalar.
        vertical (bool): If the flipping should be vertical. Scalar.
        width (int): the width of the canvas. Used for translating the coordinates into the canvas.
            Defaults to None (no added translation).
        height (int): the height of the canvas. Used for translating the coordinates back into the
            canvas. Defaults to None (no added translation).
    Returns:
        (np.ndarray): A fp32 array of shape (3, 3), spatial transformation matrix if horizontal and
            vertical are scalars.
    """

    if (width is None) ^ (height is None):
        raise ValueError(
            "Variables `width` and `height` should both be defined, or both `None`."
        )
    elif width is not None and height is not None:
        x_t = horizontal * width
        y_t = vertical * height
    else:
        x_t = 0.0
        y_t = 0.0

    m = [[1 - 2.0 * horizontal, 0.0, x_t],
         [0.0, 1 - 2.0 * vertical, y_t],
         [0.0, 0.0, 1.0]]

    return np.array(m, dtype=np.float32)


def rotation_matrix_np(theta, width=None, height=None):
    """Construct a rotation transformation matrix.

    Note that if width and height are supplied, it will rotate the coordinates around the canvas
    center-point, so there will be a translation added to the rotation matrix.

    Args:
        theta (float): the rotation radian. Scalar.
        width (int): the width of the canvas. Used for center rotation. Defaults to None
            (no center rotation).
        height (int): the height of the canvas. Used for center rotation. Defaults to None
            (no center rotation).
    Returns:
        (np.ndarray): A fp32 tensor of shape (3, 3), spatial transformation matrix if theta is
            scalar.
    """

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    if (width is None) ^ (height is None):
        raise ValueError(
            "Variables `width` and `height` should both be defined, or both `None`."
        )
    elif width is not None and height is not None:
        x_t = height * sin_t / 2.0 - width * cos_t / 2.0 + width / 2.0
        y_t = -1 * height * cos_t / 2.0 + height / 2.0 - width * sin_t / 2.0
    else:
        x_t = 0.0
        y_t = 0.0

    m = [[cos_t, -sin_t, x_t],
         [sin_t, cos_t, y_t],
         [0.0, 0.0, 1.0]]

    return np.array(m, dtype=np.float32)


def translation_matrix_np(x, y):
    """Construct a spatial transformation matrix for translation.

    Args:
        x (float): the horizontal translation. Scalar.
        y (float): the vertical translation. Scalar.
    Returns:
        (np.ndarray): A fp32 tensor of shape (3, 3), spatial transformation matrix if x and y are
            scalars.
    """

    m = [[1.0, 0.0, x],
         [0.0, 1.0, y],
         [0.0, 0.0, 1.0]]

    return np.array(m, dtype=np.float32)


def zoom_matrix_np(ratio, width=None, height=None):
    """Construct a spatial transformation matrix for zooming.

    Note that if width and height are supplied, it will perform a center-zoom by translation.

    Args:
        ratio (float or tuple(2) of float): the zoom ratio. If a tuple of length 2 is supplied,
            they distinguish between the horizontal and vertical zooming. Scalar or
            a tuple of scalars.
        width (int): the width of the canvas. Used for center-zooming. Defaults to None (no added
            translation).
        height (int): the height of the canvas. Used for center-zooming. Defaults to None (no added
            translation).
    Returns:
        (tf.Tensor): A fp32 tensor of shape (3, 3), spatial transformation matrix if ratio is
            scalar. If ratio is a vector, (len(ratio), 3, 3).
    """
    if isinstance(ratio, tuple) and len(ratio) == 2:
        r_x, r_y = ratio
    else:
        r_x, r_y = ratio, ratio

    if (width is None) ^ (height is None):
        raise ValueError(
            "Variables `width` and `height` should both be defined, or both `None`."
        )
    elif width is not None and height is not None:
        x_t = (width - width * r_x) * 0.5
        y_t = (height - height * r_y) * 0.5
    else:
        x_t = 0.0
        y_t = 0.0

    m = [[r_x, 0.0, x_t],
         [0.0, r_y, y_t],
         [0.0, 0.0, 1.0]]

    return np.array(m, dtype=np.float32)


def get_spatial_transformation_matrix_np(
    target_width,
    target_height,
    image_width=None,
    image_height=None,
    stm=None,
    flip_lr=False,
    flip_tb=False,
    translate_x=0,
    translate_y=0,
    zoom_ratio=1.0,
    rotate_rad=0.0,
    shear_ratio_x=0.0,
    shear_ratio_y=0.0,
    augmentation_mode=None,
    batch_size=None,
):
    """
    The spatial transformation matrix (stm) generator used for augmentation.

    This function creates a spatial transformation matrix (stm) that can be used for
    generic data augmentation, usually images or coordinates.

    Args:
        target_width (int): the width of the destination image canvas.
        target_height (int): the height of the destination image canvas.
        image_width (int): the width of the source image.
        image_height (int): the height of the source image.
        stm ((3,3) fp32 np.ndarray or None): A spatial transformation matrix produced in this
            function and will be used to transform images and coordinates spatiallly.
            If ``None`` (default), an identity matrix will be generated.
        flip_lr (bool): Flag to indicate whether to flip the image left/right or not.
        flip_tb (bool): Flag to indicate whether to flip the image top/bottom or not.
        translate_x (int): The amount by which to translate the image horizontally.
        translate_y (int): The amount by which to translate the image vertically.
        zoom_ratio (float): The ratio by which to zoom into the image. A zooming ratio of 1.0
            will not affect the image, while values higher than 1 will result in 'zooming out'
            (image gets rendered smaller than the canvas), and vice versa for values below 1.0.
        rotate_rad (float): The rotation in radians.
        shear_ratio_x (float): The amount to shear the horizontal direction per y row.
        shear_ratio_y (float): The amount to shear the vertical direction per x column.
        augmentation_mode (AugmentationModes): Augmentation mode to apply for the
                images. Refer to the Enum class for details about the augmentation modes.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
    Returns:
        (np.ndarray): If batch_size is None, a spatial transformation matrix of shape (3,3)
        and type np.float32. If batch_size is not None, a tensor of shape (batch_size,3,3).
    """

    # Initialize the spatial transform matrix as a 3x3 identity matrix
    if stm is None:
        stm = np.eye(3, dtype=np.float32)

    if augmentation_mode == AugmentationModes.PERSON_CENTRIC:

        # Align the center of the person of interest with the origin of the image.
        # NOTE: This also includes a random shift in addition to the center of
        # POI.
        translate_transformation = translation_matrix_np(
            translate_x, translate_y
        )
        stm = np.matmul(translate_transformation, stm)

        # Apply rotation transform.
        rotation_transformation = rotation_matrix_np(rotate_rad)
        stm = np.matmul(rotation_transformation, stm)

        # Apply zoom/scale transform.
        zoom_transformation = zoom_matrix_np(zoom_ratio)
        stm = np.matmul(zoom_transformation, stm)

        # Apply horizontal flipping.
        flip_transformation = flip_matrix_np(flip_lr, flip_tb)
        stm = np.matmul(flip_transformation, stm)

        # Align the origin back with the center of the image (once all the
        # transformations are applied).
        translate_transformation_2 = translation_matrix_np(
            target_width // 2, target_height // 2
        )
        stm = np.matmul(translate_transformation_2, stm)

    elif augmentation_mode in (
            AugmentationModes.STANDARD, AugmentationModes.STANDARD_WITH_FIXED_ASPECT_RATIO):

        # If mode is standard, retain aspect ratio of the original image
        if augmentation_mode == AugmentationModes.STANDARD:
            aspect_ratio = float(image_width) / float(image_height)
        else:
            aspect_ratio = 1.0

        # Estimate the spatial tranformation matrix using quad->rect mapping
        quad_stm = get_quad_tranformation_matrix_np(
            image_width,
            image_height,
            target_width,
            target_height,
            flip_lr=flip_lr,
            flip_tb=flip_tb,
            translate_x=translate_x,
            translate_y=translate_y,
            zoom_ratio=zoom_ratio,
            rotate_rad=rotate_rad,
            shear_ratio_x=shear_ratio_x,
            shear_ratio_y=shear_ratio_y,
            aspect_ratio=aspect_ratio
        )

        quad_stm = np.asarray(quad_stm, np.float32)
        stm = np.matmul(quad_stm, stm)

    return stm


def get_quad_tranformation_matrix_np(
    image_width,
    image_height,
    target_width,
    target_height,
    flip_lr=False,
    flip_tb=False,
    translate_x=0,
    translate_y=0,
    zoom_ratio=1.0,
    rotate_rad=0.0,
    shear_ratio_x=0.0,
    shear_ratio_y=0.0,
    aspect_ratio=1.0
):
    """
    The spatial transformation matrix (stm) generated from quad -> rect mapping.

    Args:
        target_width (int): the width of the destination image canvas.
        target_height (int): the height of the destination image canvas.
        image_width (int): the width of the source image.
        image_height (int): the height of the source image.
        flip_lr (bool): Flag to indicate whether to flip the image left/right or not.
        flip_tb (bool): Flag to indicate whether to flip the image top/bottom or not.
        translate_x (int): The amount by which to translate the image horizontally.
        translate_y (int): The amount by which to translate the image vertically.
        zoom_ratio (float): The ratio by which to zoom into the image. A zooming ratio of 1.0
            will not affect the image, while values higher than 1 will result in 'zooming out'
            (image gets rendered smaller than the canvas), and vice versa for values below 1.0.
        rotate_rad (float): The rotation in radians.
        shear_ratio_x (float): The amount to shear the horizontal direction per y row.
        shear_ratio_y (float): The amount to shear the vertical direction per x column.
        aspect_ratio (float): The desired aspect ratio of the image in target canvas

    Returns:
        (np.ndarray): a spatial transformation matrix of shape (3,3) and type np.float32.
    """

    # TODO: Add support for shearing
    # NOTE: The quad is being estimated for unit scale
    translate_x_ratio = translate_x / image_width
    translate_y_ratio = translate_y / image_height
    quad_ratio = get_quad_ratio_np(
        flip_lr=flip_lr,
        flip_tb=flip_tb,
        translate=(translate_x_ratio, translate_y_ratio),
        zoom_ratio=zoom_ratio,
        rotate_rad=rotate_rad,
        aspect_ratio=aspect_ratio)

    # Convert to original image space
    quad = np.zeros(quad_ratio.shape, quad_ratio.dtype)
    quad[:, 0] = quad_ratio[:, 0] * image_width
    quad[:, 1] = quad_ratio[:, 1] * image_height

    # Convert to (top-left, top-right, bottom-right, bottom-left)
    # cv2.getPerspectiveTransform expects in clockwise format
    quad = np.array([
        [quad[0][0], quad[0][1]],
        [quad[3][0], quad[3][1]],
        [quad[2][0], quad[2][1]],
        [quad[1][0], quad[1][1]]], dtype=np.float32)

    dst_rect = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]], dtype=np.float32)

    # Compute perspective transformation from the source and destination points
    stm = cv2.getPerspectiveTransform(quad, dst_rect)

    return stm


def get_quad_ratio_np(
    flip_lr=False,
    flip_tb=False,
    translate=(0.0, 0.0),
    zoom_ratio=1.0,
    rotate_rad=0.0,
    shear_ratio_x=0.0,
    shear_ratio_y=0.0,
    aspect_ratio=1.0
):
    """
    The quad to rectangle mapping (stm) generated using the desired augmentation params.

    Note that the points of the quad are returned in unit scale and need to be scaled
    to image space.
    Args:
        flip_lr (bool): Flag to indicate whether to flip the image left/right or not.
        flip_tb (bool): Flag to indicate whether to flip the image top/bottom or not.
        translate (tuple): (translate_x_ratio, translate_y_ratio).
        zoom_ratio (float): The ratio by which to zoom into the image. A zooming ratio of 1.0
            will not affect the image, while values higher than 1 will result in 'zooming out'
            (image gets rendered smaller than the canvas), and vice versa for values below 1.0.
        rotate_rad (float): The rotation in radians.
        shear_ratio_x (float): The amount to shear the horizontal direction per y row.
        shear_ratio_y (float): The amount to shear the vertical direction per x column.
        aspect_ratio (float): The desired aspect ratio of the image in target canvas

    Returns:
        (np.ndarray): a quad array of shape (4,2) and type np.float32.
    """

    if aspect_ratio > 1.0:
        # Scenario where image_width > image_height
        # Increase height region
        quad = np.array([
            [0.0, 0.5 - 0.5 * aspect_ratio],
            [0.0, 0.5 + 0.5 * aspect_ratio],
            [1.0, 0.5 + 0.5 * aspect_ratio],
            [1.0, 0.5 - 0.5 * aspect_ratio]])
    elif aspect_ratio < 1.0:
        # Scenario where image_width < image_height
        # Increase width region
        quad = np.array([
            [0.5 - 0.5 / aspect_ratio, 0.0],
            [0.5 - 0.5 / aspect_ratio, 1.0],
            [0.5 + 0.5 / aspect_ratio, 1.0],
            [0.5 + 0.5 / aspect_ratio, 0.0]])
    else:
        # Scenario where image_width = image_height
        # No change to target height and width
        quad = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ])

    # Shift to origin to center of image
    # to perform augmentations about the center
    quad -= 0.5

    # Fet rotation matrix and apply rotation
    R = np.array([
        [np.cos(rotate_rad), -np.sin(rotate_rad)],
        [np.sin(rotate_rad), np.cos(rotate_rad)]
    ])
    quad = np.dot(quad, R)

    # Apply translation
    quad -= np.array(translate)

    # Apply scaling
    quad /= zoom_ratio

    # Shift origin back to top left
    quad += 0.5

    # Flip let-right if true
    if flip_lr:
        quad = np.array([
            [quad[3][0], quad[3][1]],
            [quad[2][0], quad[2][1]],
            [quad[1][0], quad[1][1]],
            [quad[0][0], quad[0][1]]])

    # Flip top-bottom if true
    if flip_tb:
        quad = np.array([
            [quad[1][0], quad[1][1]],
            [quad[0][0], quad[0][1]],
            [quad[3][0], quad[3][1]],
            [quad[2][0], quad[2][1]]])

    return quad
