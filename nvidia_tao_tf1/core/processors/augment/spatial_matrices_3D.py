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
"""Homogeneous 3D spatial transformation matrices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rotation_matrix_3D(x=0.0, y=0.0, z=0.0, order="ZYX"):
    """
    3D rotation matrix for counter-clockwise rotation.

    Rotations are performed in the order specified by the input argument `order`. E.g.,
    order = 'ZYX' rotates first about the z-axis, then about the y-axis, and lastly about
    the x-axis.

    The output rotation matrix is defined such that it is to be used to post-multiply
    row vectors (w*R_tot).

    Args:
        x: (0-D Tensor of type tf.float32) Rotation angle about x-axis in radians.
        y: (0-D Tensor of type tf.float32) Rotation angle about y-axis in radians.
        z: (0-D Tensor of type tf.float32) Rotation angle about z-axis in radians.
        order: (str) Rotation order ['X', 'Y', 'Z' or their combination, e.g., 'ZYX'].

    Returns:
        R_tot: (4x4 Tensor) Rotation matrix in homogeneous coordinates.
    """
    R = dict()

    if "X" in order:
        cos_x = tf.cos(x)
        sin_x = tf.sin(x)
        R["X"] = tf.stack(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                cos_x,
                sin_x,
                0.0,
                0.0,
                -sin_x,
                cos_x,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )
        R["X"] = tf.reshape(R["X"], [4, 4])

    if "Y" in order:
        cos_y = tf.cos(y)
        sin_y = tf.sin(y)
        R["Y"] = tf.stack(
            [
                cos_y,
                0.0,
                -sin_y,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                sin_y,
                0,
                cos_y,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )
        R["Y"] = tf.reshape(R["Y"], [4, 4])

    if "Z" in order:
        cos_z = tf.cos(z)
        sin_z = tf.sin(z)
        R["Z"] = tf.stack(
            [
                cos_z,
                sin_z,
                0.0,
                0.0,
                -sin_z,
                cos_z,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )
        R["Z"] = tf.reshape(R["Z"], [4, 4])

    # Overall rotation.
    R_tot = tf.eye(4)
    for ax in order:
        if ax in R:
            R_tot = tf.matmul(R_tot, R[ax])
        else:
            raise ValueError("Unsupported rotation order: %s" % order)
    return R_tot


def translation_matrix_3D(x, y, z):
    """
    Spatial transformation matrix for translation.

    The output translation matrix is defined such that it is to be used to post-multiply
    row vectors (w*T).

    Args:
        x: (0-D Tensor of type tf.float32) Translation in x-coordinate.
        y: (0-D Tensor of type tf.float32) Translation in y-coordinate.
        z: (0-D Tensor of type tf.float32) Translation in z-coordinate.
    Returns:
        T: (4x4 Tensor) Translation matrix in homogeneous coordinates.
    """
    T = tf.stack(
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0]
    )
    T = tf.reshape(T, [4, 4])
    return T


def scaling_matrix_3D(x, y, z):
    """
    Spatial transformation matrix for scaling.

    Args:
        x: (0-D Tensor of type tf.float32) Scaling in x-coordinate.
        y: (0-D Tensor of type tf.float32) Scaling in y-coordinate.
        z: (0-D Tensor of type tf.float32) Scaling in z-coordinate.
    Returns:
        S: (4x4 Tensor) Scaling matrix in homogeneous coordinates.
    """
    S = tf.stack(
        [x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    S = tf.reshape(S, [4, 4])
    return S


def flip_matrix_3D(x, y, z):
    """
    Spatial transformation matrix for flipping (=reflection) along the coordinate axes.

    Args:
        x: (0-D Tensor of type tf.bool) If x-coordinate should be flipped.
        y: (0-D Tensor of type tf.bool) If y-coordinate should be flipped.
        z: (0-D Tensor of type tf.bool) If z-coordinate should be flipped.
    Returns:
        F: (4x4 Tensor) Flipping matrix in homogeneous coordinates.
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.float32)

    F = tf.stack(
        [
            1.0 - 2 * x,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 - 2 * y,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 - 2 * z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    F = tf.reshape(F, [4, 4])
    return F
