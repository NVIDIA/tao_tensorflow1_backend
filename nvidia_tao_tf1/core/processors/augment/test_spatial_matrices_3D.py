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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.processors.augment.spatial_matrices_3D import flip_matrix_3D
from nvidia_tao_tf1.core.processors.augment.spatial_matrices_3D import rotation_matrix_3D
from nvidia_tao_tf1.core.processors.augment.spatial_matrices_3D import scaling_matrix_3D
from nvidia_tao_tf1.core.processors.augment.spatial_matrices_3D import translation_matrix_3D


@pytest.mark.parametrize("rotations", [1, 2, 4, 9])
@pytest.mark.parametrize("order", ["X", "Y", "Z"])
def test_rotation_matrix_3D_single_axis(rotations, order):
    """Perform a full rotation (2*pi) in a few steps, and check it yields the identity matrix."""
    x = np.pi * 2 / rotations
    y = np.pi * 2 / rotations
    z = np.pi * 2 / rotations
    m = rotation_matrix_3D(x=x, y=y, z=z, order=order)
    out = tf.eye(4, dtype=tf.float32)
    for _ in range(rotations):
        out = tf.matmul(out, m)
    out_np, m_np = tf.compat.v1.Session().run([out, m])

    # Check that our single-rotation matrix is different than the output.
    if rotations > 1:
        np.testing.assert_equal(np.any(np.not_equal(m_np, out_np)), True)

    # Check that our full rotation yields the identity matrix.
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(
        expected,
        out_np,
        atol=1e-4,
        err_msg="Full rotation through "
        "multiple steps did not result in the identity matrix.",
    )


@pytest.mark.parametrize("x", [1.2523])
@pytest.mark.parametrize("y", [0.7452])
@pytest.mark.parametrize("z", [-2.156])
@pytest.mark.parametrize("order", ["XY", "YZ", "XZ", "ZYX", "YXZ"])
def test_rotation_matrix_3D_multiple_axes(x, y, z, order):
    """Rotate first in the given order and then in the opposite order with negated angles.

    The result should be an identity matrix."""
    m = rotation_matrix_3D(x=x, y=y, z=z, order=order)
    out = tf.eye(4, dtype=tf.float32)
    out = tf.matmul(out, m)
    m2 = rotation_matrix_3D(x=-x, y=-y, z=-z, order=order[::-1])
    out = tf.matmul(out, m2)
    out_np, m_np = tf.compat.v1.Session().run([out, m])

    # Check that our single-rotation matrix is different than the output.
    np.testing.assert_equal(np.any(np.not_equal(m_np, out_np)), True)

    # Check that our full rotation yields the identity matrix.
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(
        expected,
        out_np,
        atol=1e-4,
        err_msg="Full rotation through "
        "two opposite steps did not result in the identity matrix.",
    )


@pytest.mark.parametrize("x", [1.2523])
@pytest.mark.parametrize("y", [0.7452])
@pytest.mark.parametrize("z", [-2.156])
@pytest.mark.parametrize("order", ["XYZ", "ZYX", "YXZ"])
def test_rotation_matrix_3D_order(x, y, z, order):
    """Check that order of multiplication is correct by comparing to single step rotations."""

    # Rotation matrix.
    m = rotation_matrix_3D(x=x, y=y, z=z, order=order)
    out = tf.eye(4, dtype=tf.float32)
    out = tf.matmul(out, m)

    # Rotation matrix with multiple single step rotations.
    out2 = tf.eye(4, dtype=tf.float32)
    for ax in order:
        m2 = rotation_matrix_3D(x=x, y=y, z=z, order=ax)
        out2 = tf.matmul(out2, m2)
    out_np, out2_np = tf.compat.v1.Session().run([out, out2])

    # Check that the two rotation matrices are identical.
    np.testing.assert_allclose(
        out_np,
        out2_np,
        atol=1e-4,
        err_msg="Rotation matrix defined by "
        "order is not same as the multiplication of individual matrices.",
    )


@pytest.mark.parametrize("x", [-5.0, 3.0, -4.0])
@pytest.mark.parametrize("y", [-5.0, 3.0, -4.0])
@pytest.mark.parametrize("z", [-5.0, 3.0, -4.0])
def test_translation_matrix_3D(x, y, z):
    """Test translation by translating and inversely translating, to yield an identity matrix."""
    m = translation_matrix_3D(x=x, y=y, z=z)
    m_inv = translation_matrix_3D(x=-x, y=-y, z=-z)
    out = tf.matmul(m, m_inv)
    out_np, m_np, m_inv_np = tf.compat.v1.Session().run([out, m, m_inv])

    # Check that our translation and its inverse translation are different
    np.testing.assert_equal(np.any(np.not_equal(m_np, m_inv_np)), True)

    # Check that our roundtrip yields the identity matrix.
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_array_equal(
        expected,
        out_np,
        err_msg="Flip roundtrip did not result in the " "identity matrix.",
    )


@pytest.mark.parametrize("x", [0.666, 1.0, 1.337])
@pytest.mark.parametrize("y", [0.666, 1.0, 1.337])
@pytest.mark.parametrize("z", [0.666, 1.0, 1.337])
def test_scaling_matrix_3D(x, y, z):
    """Test scaling and applying the inverse scaling to yield the identity matrix."""
    m = scaling_matrix_3D(x=x, y=y, z=z)
    m_inv = scaling_matrix_3D(x=1.0 / x, y=1.0 / y, z=1.0 / z)
    out = tf.matmul(m, m_inv)
    out_np, m_pos_np, m_neg_np = tf.compat.v1.Session().run([out, m, m_inv])

    # Check that our translation and its inverse translation are different.
    if x != 1.0 or y != 1.0 or z != 1.0:
        np.testing.assert_equal(np.any(np.not_equal(m_pos_np, m_neg_np)), True)

    # Check that our roundtrip yields the identity matrix.
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(
        expected,
        out_np,
        atol=1e-5,
        err_msg="Flip roundtrip did not result " "in the identity matrix.",
    )


@pytest.mark.parametrize(
    "x, y, z",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (False, True, True),
        (True, False, True),
        (True, True, True),
    ],
)
def test_flip_matrix_3D(x, y, z):
    """Test a double flip with the same matrix, as it should return the identity matrix."""
    m = flip_matrix_3D(x=x, y=y, z=z)
    out = tf.matmul(m, m)
    out_np, m_np = tf.compat.v1.Session().run([out, m])

    # Check that our single-flip matrix is different than the output.
    np.testing.assert_equal(np.any(np.not_equal(m_np, out_np)), True)

    # Check that our roundtrip yields the identity matrix.
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_array_equal(
        expected,
        out_np,
        err_msg="Flip roundtrip did not result in the " "identity matrix.",
    )
