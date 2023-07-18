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
from six.moves import mock
import tensorflow as tf


from nvidia_tao_tf1.core.processors.augment.spatial import (
    flip_matrix,
    rotation_matrix,
    shear_matrix,
)
from nvidia_tao_tf1.core.processors.augment.spatial import get_random_spatial_transformation_matrix
from nvidia_tao_tf1.core.processors.augment.spatial import (
    random_flip_matrix,
    random_rotation_matrix,
)
from nvidia_tao_tf1.core.processors.augment.spatial import (
    random_shear_matrix,
    random_translation_matrix,
)
from nvidia_tao_tf1.core.processors.augment.spatial import (
    random_zoom_matrix,
    translation_matrix,
    zoom_matrix,
)

from nvidia_tao_tf1.core.processors.augment.testing_utils import (
    assert_bernoulli_distribution,
    assert_uniform_distribution,
    sample_tensors,
)
from nvidia_tao_tf1.core.utils import set_random_seed

NUM_SAMPLES = 1000
_WIDTH = 255
_HEIGHT = 255


def tile_spatial_matrix(stm, batch_size):
    """Tile a spatial matrix batch_size number of times."""
    if batch_size is None:
        return stm
    return np.tile(np.reshape(stm, [1, 3, 3]), [batch_size, 1, 1])


def identity_spatial_matrix(batch_size):
    """Return a batched identity matrix."""
    stm = np.eye(3, dtype=np.float32)
    return tile_spatial_matrix(stm, batch_size)


@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize(
    "horizontal, vertical", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("width, height", [(None, None), (_WIDTH, _HEIGHT)])
def test_flip_matrix(batch_size, horizontal, vertical, width, height):
    """Test a double flip with the same matrix, as it should return the identity matrix."""
    h = -1.0 if horizontal else 1.0
    v = -1.0 if vertical else 1.0
    x_t = width if horizontal and width is not None else 0.0
    y_t = height if vertical and height is not None else 0.0
    expected_stm = np.array([[h, 0.0, 0.0], [0.0, v, 0.0], [x_t, y_t, 1.0]])
    expected_stm = tile_spatial_matrix(expected_stm, batch_size)

    if batch_size is not None:
        horizontal = tf.constant(horizontal, shape=[batch_size], dtype=tf.bool)
        vertical = tf.constant(vertical, shape=[batch_size], dtype=tf.bool)

    m = flip_matrix(
        horizontal=horizontal, vertical=vertical, width=width, height=height
    )
    out = tf.matmul(m, m)
    out_np, m_np = tf.compat.v1.Session().run([out, m])

    if batch_size is None:
        assert m_np.shape == (3, 3)
    else:
        assert m_np.shape == (batch_size, 3, 3)

    # Check that our single-flip matrix is different than the output
    np.testing.assert_equal(np.any(np.not_equal(m_np, out_np)), True)

    # Check that our roundtrip yields the identity matrix
    expected = identity_spatial_matrix(batch_size)

    np.testing.assert_array_equal(
        expected,
        out_np,
        err_msg="Flip roundtrip did not result in the " "identity matrix.",
    )

    np.testing.assert_allclose(
        expected_stm,
        m_np,
        atol=1e-4,
        err_msg="Flip matrix does not match expected value.",
    )


@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("rotations", [1, 2, 4, 9])
@pytest.mark.parametrize("width, height", [(None, None), (_WIDTH, _HEIGHT)])
def test_rotation_matrix(batch_size, rotations, width, height):
    """Perform a full rotation (2*pi) in a few steps, and check it yields the identity matrix."""
    theta = np.pi * 2 / rotations

    # Compute expected rotation matrix.
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    if width is not None and height is not None:
        x_t = height * sin_t / 2.0 - width * cos_t / 2.0 + width / 2.0
        y_t = -1 * height * cos_t / 2.0 + height / 2.0 - width * sin_t / 2.0
    else:
        x_t = y_t = 0.0
    expected_stm = np.array(
        [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [x_t, y_t, 1.0]]
    )
    expected_stm = tile_spatial_matrix(expected_stm, batch_size)

    if batch_size is not None:
        theta = tf.constant(theta, shape=[batch_size], dtype=tf.float32)

    m = rotation_matrix(theta, width=width, height=height)
    batch_shape = [] if batch_size is None else [batch_size]
    out = tf.eye(3, batch_shape=batch_shape, dtype=tf.float32)
    for _ in range(rotations):
        out = tf.matmul(out, m)
    out_np, m_np = tf.compat.v1.Session().run([out, m])

    if batch_size is None:
        assert m_np.shape == (3, 3)
    else:
        assert m_np.shape == (batch_size, 3, 3)

    np.testing.assert_allclose(
        expected_stm,
        m_np,
        atol=1e-4,
        err_msg="Rotation matrix does not match expected value.",
    )

    # Check that our single-rotation matrix is different than the output
    if rotations > 1:
        np.testing.assert_equal(np.any(np.not_equal(m_np, out_np)), True)

    # Check that our full rotation yields the identity matrix
    expected = identity_spatial_matrix(batch_size)
    np.testing.assert_allclose(
        expected,
        out_np,
        atol=1e-4,
        err_msg="Full rotation through "
        "multiple steps did not result in the identity matrix.",
    )


@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("x", [0.5, 0.0])
@pytest.mark.parametrize("y", [1.5, 1.0])
@pytest.mark.parametrize("width, height", [(None, None), (_WIDTH, _HEIGHT)])
def test_shear_matrix(batch_size, x, y, width, height):
    """Test shear transform by shearing and inversely shearing,
    and check it yields the identity matrix."""
    # Compute expected matrix.
    if width and height:
        x_t = width / 2.0 * y * x
        y_t = height / 2.0 * x * y
    else:
        x_t, y_t = 0.0, 0.0
    diag = 1.0 - x * y
    expected_stm = np.array(
        [[diag, 0.0, 0.0], [0.0, diag, 0.0], [x_t, y_t, 1.0]], dtype=np.float32
    )
    expected_stm = tile_spatial_matrix(expected_stm, batch_size)

    if batch_size is not None:
        x = tf.constant(x, shape=[batch_size], dtype=tf.float32)
        y = tf.constant(y, shape=[batch_size], dtype=tf.float32)

    m = shear_matrix(ratio_x=x, ratio_y=y, width=width, height=height)
    m_inv = shear_matrix(ratio_x=-x, ratio_y=-y, width=width, height=height)
    out = tf.matmul(m, m_inv)
    out_np, m_np, m_inv_np = tf.compat.v1.Session().run([out, m, m_inv])

    if batch_size is None:
        assert m_np.shape == (3, 3)
    else:
        assert m_np.shape == (batch_size, 3, 3)

    # Check that one single shear is different with the output.
    np.testing.assert_equal(np.any(np.not_equal(m_np, m_inv_np)), True)

    # Check that our shear transform generate expected matrix.
    np.testing.assert_allclose(
        expected_stm,
        out_np,
        atol=1e-4,
        err_msg="Shear and unshear in "
        "the same direction with same amount does not result in "
        "an expected matrix.",
    )


@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("x", [-5, 3, -4])
@pytest.mark.parametrize("y", [-5, 3, -4])
def test_translation_matrix(batch_size, x, y):
    """Test translation by translating and inversely translating, to yield an identity matrix."""
    expected_stm = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [x, y, 1.0]], dtype=np.float32
    )
    expected_stm = tile_spatial_matrix(expected_stm, batch_size)

    if batch_size is not None:
        x = tf.constant(x, shape=[batch_size], dtype=tf.float32)
        y = tf.constant(y, shape=[batch_size], dtype=tf.float32)

    m = translation_matrix(x=x, y=y)
    m_inv = translation_matrix(x=-x, y=-y)
    out = tf.matmul(m, m_inv)
    out_np, m_np, m_inv_np = tf.compat.v1.Session().run([out, m, m_inv])

    if batch_size is None:
        assert m_np.shape == (3, 3)
    else:
        assert m_np.shape == (batch_size, 3, 3)

    np.testing.assert_allclose(
        expected_stm,
        m_np,
        atol=1e-4,
        err_msg="Translation matrix does not match expected value.",
    )

    # Check that our translation and its inverse translation are different
    np.testing.assert_equal(np.any(np.not_equal(m_np, m_inv_np)), True)

    # Check that our roundtrip yields the identity matrix
    expected = identity_spatial_matrix(batch_size)
    np.testing.assert_array_equal(
        expected,
        out_np,
        err_msg="Flip roundtrip did not result in the " "identity matrix.",
    )


@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("ratio", [0.666, 1.0, 1.337])
@pytest.mark.parametrize("width, height", [(None, None), (_WIDTH, _HEIGHT)])
def test_zoom_matrix(batch_size, ratio, width, height):
    """Test zooming in and applying the inverse zoom to yield the identity matrix."""
    # Compute expected zoom matrix.
    r_x = ratio
    r_y = ratio
    if width is not None and height is not None:
        x_t = (width - width * r_x) * 0.5
        y_t = (height - height * r_y) * 0.5
    else:
        x_t = y_t = 0.0
    expected_stm = np.array(
        [[r_x, 0.0, 0.0], [0.0, r_y, 0.0], [x_t, y_t, 1.0]], dtype=np.float32
    )
    expected_stm = tile_spatial_matrix(expected_stm, batch_size)

    expect_difference = ratio != 1.0
    if batch_size is not None:
        ratio = tf.constant(ratio, shape=[batch_size], dtype=tf.float32)
    m = zoom_matrix(ratio=ratio, width=width, height=height)
    m_inv = zoom_matrix(ratio=1.0 / ratio, width=width, height=height)
    out = tf.matmul(m, m_inv)
    out_np, m_pos_np, m_neg_np = tf.compat.v1.Session().run([out, m, m_inv])

    if batch_size is None:
        assert m_pos_np.shape == (3, 3)
    else:
        assert m_pos_np.shape == (batch_size, 3, 3)

    np.testing.assert_allclose(
        expected_stm,
        m_pos_np,
        atol=1e-4,
        err_msg="Zoom matrix does not match expected value.",
    )

    # Check that our translation and its inverse translation are different
    if expect_difference:
        np.testing.assert_equal(np.any(np.not_equal(m_pos_np, m_neg_np)), True)

    # Check that our roundtrip yields the identity matrix
    expected = identity_spatial_matrix(batch_size)
    np.testing.assert_allclose(
        expected,
        out_np,
        atol=1e-5,
        err_msg="Flip roundtrip did not result " "in the identity matrix.",
    )


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.spatial.flip_matrix",
    side_effect=lambda horizontal, vertical, width, height: horizontal,
)
@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("flip_lr_prob", [0.0, 1.0, 0.5])
def test_random_flip_matrix_horizontal(patched, batch_size, flip_lr_prob):
    """Test that random_flip_matrix produces correct distributions."""
    set_random_seed(42)
    flip_tensor = random_flip_matrix(
        flip_lr_prob, 0.0, _WIDTH, _HEIGHT, batch_size=batch_size
    )
    flips = sample_tensors([flip_tensor], NUM_SAMPLES)

    assert_bernoulli_distribution(flips[0], p=flip_lr_prob)


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.spatial.flip_matrix",
    side_effect=lambda horizontal, vertical, width, height: vertical,
)
@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("flip_tb_prob", [1.0, 0.5])
def test_random_flip_matrix_vertical(patched, batch_size, flip_tb_prob):
    """Test that random_flip_matrix produces correct distributions."""
    # Using a different random seed because 42 generates numbers that fails this test.
    set_random_seed(40)
    flip_tensor = random_flip_matrix(
        0.0, flip_tb_prob, _WIDTH, _HEIGHT, batch_size=batch_size
    )
    flips = sample_tensors([flip_tensor], NUM_SAMPLES)

    assert_bernoulli_distribution(flips[0], p=flip_tb_prob)


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.spatial.translation_matrix",
    side_effect=lambda x, y: (x, y),
)
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize(("translate_max_x", "translate_max_y"), [(0, 0), (16, 16)])
@pytest.mark.parametrize(
    ("translate_min_x", "translate_min_y"), [(-16, -16), (0, 0), (None, None)]
)
def test_random_translation_matrix(
    patched,
    batch_size,
    translate_max_x,
    translate_max_y,
    translate_min_x,
    translate_min_y,
):
    """Test that random_translation_matrix produces correct distributions."""
    set_random_seed(42)
    translation_tensors = random_translation_matrix(
        translate_max_x, translate_max_y, batch_size, translate_min_x, translate_min_y
    )
    translate_xs, translate_ys = sample_tensors(translation_tensors, NUM_SAMPLES)

    if translate_min_x is None:
        translate_min_x = -translate_max_x
    if translate_min_y is None:
        translate_min_y = -translate_max_y

    assert_uniform_distribution(translate_xs, translate_min_x, translate_max_x)
    assert_uniform_distribution(translate_ys, translate_min_y, translate_max_y)


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.spatial.shear_matrix",
    side_effect=lambda x, y, w, h: (x, y),
)
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize(("max_ratio_x", "max_ratio_y"), [(0, 0), (0.2, 0.5)])
@pytest.mark.parametrize(
    ("min_ratio_x", "min_ratio_y"), [(-0.5, -0.2), (0, 0), (None, None)]
)
def test_random_shear_matrix(
    patched, batch_size, max_ratio_x, max_ratio_y, min_ratio_x, min_ratio_y
):
    """Test that random_shear_matrix produces correct distributions."""
    set_random_seed(42)
    shear_tensors = random_shear_matrix(
        max_ratio_x,
        max_ratio_y,
        _WIDTH,
        _HEIGHT,
        batch_size=batch_size,
        min_ratio_x=min_ratio_x,
        min_ratio_y=min_ratio_y,
    )
    # Sample 2x the regular amount to make sure the test passes.
    shear_xs, shear_ys = sample_tensors(shear_tensors, NUM_SAMPLES * 2)

    if min_ratio_x is None:
        min_ratio_x = -max_ratio_x
    if min_ratio_y is None:
        min_ratio_y = -max_ratio_y

    assert_uniform_distribution(shear_xs, min_ratio_x, max_ratio_x)
    assert_uniform_distribution(shear_ys, min_ratio_y, max_ratio_y)


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.spatial.rotation_matrix", side_effect=lambda a, w, h: a
)
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("rotate_rad_max", [0.0, 1.56, 3.1415])
@pytest.mark.parametrize("rotate_rad_min", [0.0, -1.50, -3.0, None])
def test_random_rotation_matrix(patched, batch_size, rotate_rad_max, rotate_rad_min):
    """Test that random_rotation_matrix produces correct distributions."""
    set_random_seed(42)
    rotation_tensor = random_rotation_matrix(
        rotate_rad_max,
        _WIDTH,
        _HEIGHT,
        batch_size=batch_size,
        rotate_rad_min=rotate_rad_min,
    )
    rotations = sample_tensors([rotation_tensor], NUM_SAMPLES)

    if rotate_rad_min is None:
        rotate_rad_min = -rotate_rad_max

    assert_uniform_distribution(rotations, rotate_rad_min, rotate_rad_max)


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.spatial.zoom_matrix", side_effect=lambda ratio: ratio
)
@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.spatial.translation_matrix",
    side_effect=lambda x, y: (x, y),
)
@mock.patch("tensorflow.matmul", side_effect=lambda x, y: (x[0], x[1], y))
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize(
    "zoom_ratio_min, zoom_ratio_max", [(0.5, 0.8), (1.0, 1.0), (1.2, 1.5), (0.5, 1.5)]
)
def test_random_zoom_matrix(
    patched_zoom,
    patched_translation,
    patched_Matmul,
    batch_size,
    zoom_ratio_min,
    zoom_ratio_max,
):
    """Test that random_zoom_matrix produces correct distributions."""
    set_random_seed(42)
    tensors = random_zoom_matrix(
        zoom_ratio_min, zoom_ratio_max, _WIDTH, _HEIGHT, batch_size=batch_size
    )
    translate_xs, translate_ys, scales = sample_tensors(tensors, NUM_SAMPLES)

    assert_uniform_distribution(scales, zoom_ratio_min, zoom_ratio_max)

    # Check that translation values are within boundaries. Note that translation isn't sampled from
    # distribution with constant min/max parameters, but distribution with min/max bounds varying
    # based on zoom ratio. This means that we need to find the maximum bound for every zoom ratio.
    # Further complications arise from a fact that the max value of distribution can be negative
    # when zoom_ratio < 1.0. To handle this, we're working with absolute value of both bounds and
    # sampled translation values.
    max_x_lower_bound = _WIDTH - (_WIDTH / zoom_ratio_min)
    max_x_upper_bound = _WIDTH - (_WIDTH / zoom_ratio_max)

    # This is maximum possible absolute value of translation.
    max_x = np.maximum(np.abs(max_x_lower_bound), np.abs(max_x_upper_bound))

    assert np.max(np.abs(translate_xs)) <= max_x
    assert np.min(np.abs(translate_xs)) >= 0

    max_y_lower_bound = _HEIGHT - (_HEIGHT / zoom_ratio_min)
    max_y_upper_bound = _HEIGHT - (_HEIGHT / zoom_ratio_max)

    max_y = np.maximum(np.abs(max_y_lower_bound), np.abs(max_y_upper_bound))

    assert np.max(np.abs(translate_ys)) <= max_y
    assert np.min(np.abs(translate_ys)) >= 0


@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("width, height, flip_lr_prob", [(1, 1, 0), [1, 1, 1]])
def test_get_random_spatial_transformation_matrix(
    batch_size, width, height, flip_lr_prob
):
    """
    Test generate random spatial transform matrix.
    """
    set_random_seed(42)
    stm = get_random_spatial_transformation_matrix(
        width=width,
        height=height,
        flip_lr_prob=flip_lr_prob,
        translate_max_x=0,
        translate_max_y=0,
        zoom_ratio_min=1.0,
        zoom_ratio_max=1.0,
        rotate_rad_max=0.0,
        shear_max_ratio_x=0.0,
        shear_max_ratio_y=0.0,
        batch_size=batch_size,
    )
    stm_np = tf.compat.v1.Session().run(stm)
    if flip_lr_prob:
        stm_ref = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        stm_ref = tile_spatial_matrix(stm_ref, batch_size)
        np.testing.assert_allclose(stm_ref, stm_np)
    else:
        stm_ref = identity_spatial_matrix(batch_size)
        np.testing.assert_array_equal(stm_ref, stm_np)


def test_no_op_spatial_transform():
    """Tests that supplying no kwargs results in a no-op spatial transformation matrix."""
    height, width = np.random.randint(1000, size=2)
    stm = get_random_spatial_transformation_matrix(width, height)
    stm_np = tf.compat.v1.Session().run(stm)
    np.testing.assert_equal(
        stm_np,
        np.eye(3),
        verbose=True,
        err_msg="Default spatial transformation matrix is not the identity matrix.",
    )
