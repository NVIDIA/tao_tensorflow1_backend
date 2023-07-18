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
from six.moves import mock, xrange
import tensorflow as tf

from nvidia_tao_tf1.core.processors.augment import color
from nvidia_tao_tf1.core.processors.augment.testing_utils import (
    assert_truncated_normal_distribution,
    assert_uniform_distribution,
    sample_tensors,
)
from nvidia_tao_tf1.core.utils import set_random_seed

NUM_SAMPLES = 1000


def tile_color_matrix(ctm, batch_size):
    """Tile a color matrix batch_size number of times."""
    if batch_size is None:
        return ctm
    return np.tile(np.reshape(ctm, [1, 4, 4]), [batch_size, 1, 1])


def identity_color_matrix(batch_size):
    """Return a batched identity matrix."""
    ctm = np.eye(4, dtype=np.float32)
    return tile_color_matrix(ctm, batch_size)


@pytest.fixture(scope="module")
def get_random_image(batch_size, start=0.0, stop=1.0):
    """Create a batch of images, with values within a linspace, that are then randomly shuffled."""
    shape = (batch_size, 16, 64, 3)
    images = np.linspace(start, stop, batch_size * 3072, dtype=np.float32).reshape(
        shape
    )
    return np.random.permutation(images)


offset_tests = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0), (0.1, 0.2, 0.3)]


@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("offset", offset_tests)
def test_brightness_offset_matrix(batch_size, offset):
    """Test the brightness offset matrix, by checking it's an identity matrix with offsets."""
    if batch_size is not None:
        offset = np.tile(offset, [batch_size, 1])
    m = color.brightness_offset_matrix(offset)
    m_np = tf.compat.v1.Session().run(m)
    if batch_size is not None:
        assert m_np.shape == (batch_size, 4, 4)
        created_offsets = m_np[:, 3, 0:3]
    else:
        assert m_np.shape == (4, 4)
        created_offsets = m_np[3, 0:3]

    # Test the validity of the offsets
    np.testing.assert_allclose(
        offset,
        created_offsets,
        rtol=1e-6,
        err_msg="Offset matrix contains different offset values than those "
        "supplied.",
    )

    # Test the rest of the matrix is untouched (identity)
    # Zero out the offests, so we can test versus an identity matrix.
    if batch_size is not None:
        m_np[:, 3, 0:3] = 0.0
    else:
        m_np[3, 0:3] = 0.0
    expected = identity_color_matrix(batch_size)
    np.testing.assert_allclose(
        expected,
        m_np,
        rtol=1e-6,
        err_msg="Brightness offset matrix introduced non-identity values "
        "in elements other than the expected offsets.",
    )


@pytest.mark.parametrize("batch_size", [None, 10])
def test_brightness_offset_matrix2(batch_size):
    """Test that brightness offset matrix matches expected value."""
    if batch_size is None:
        r = 0.5
        g = 1.0
        b = 1.5
        offset = [r, g, b]
        expected_ctm = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [r, g, b, 1.0],
            ]
        )
    else:
        r = np.linspace(-0.5, 0.5, batch_size)
        g = np.linspace(-1.0, 1.0, batch_size)
        b = np.linspace(-1.5, 1.5, batch_size)
        offset = np.transpose(np.array([r, g, b]))

        zero = np.zeros_like(r)
        one = np.ones_like(zero)
        expected_ctm = np.array(
            [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [r, g, b, one],
            ]
        )
        # Swap the batch dimension first.
        expected_ctm = np.transpose(expected_ctm, [2, 0, 1])

    m = color.brightness_offset_matrix(offset)
    m_np = tf.compat.v1.Session().run(m)

    np.testing.assert_allclose(
        expected_ctm,
        m_np,
        atol=1e-2,
        err_msg="Brightness offset matrix does not match expected value.",
    )


@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("contrast", [-0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("center", [1.0 / 2.0, 255.0 / 2.0])
def test_contrast_matrix(batch_size, contrast, center):
    """Test the contrast matrix."""
    zero_contrast = contrast == 0.0

    if batch_size is not None:
        contrast = np.tile(contrast, [batch_size])
        center = np.tile(center, [batch_size])

    m = color.contrast_matrix(contrast=contrast, center=center)
    m_np = tf.compat.v1.Session().run(m)

    if zero_contrast:
        np.testing.assert_allclose(
            identity_color_matrix(batch_size),
            m_np,
            rtol=1e-6,
            err_msg="Zero contrast did not result in the identity matrix.",
        )

    if batch_size is not None:
        assert m_np.shape == (batch_size, 4, 4)
        m = m_np[0]
    else:
        assert m_np.shape == (4, 4)
        m = m_np
    bias = np.unique(m[3, 0:3])
    scale = np.unique([m[0, 0], m[1, 1], m[2, 2]])
    assert len(scale) == 1, "Contrast scale is different across channels."
    assert len(bias) == 1, "Contrast bias is different across channels."


@pytest.mark.parametrize("batch_size", [None, 10])
def test_contrast_matrix2(batch_size):
    """Test that contrast matrix matches expectation."""
    if batch_size is None:
        contrast = 1.5
        center = 0.5
    else:
        contrast = np.linspace(0.0, 2.0, batch_size)
        center = np.linspace(-1.0, 1.0, batch_size)

    m = color.contrast_matrix(contrast=contrast, center=center)
    m_np = tf.compat.v1.Session().run(m)

    zero = np.zeros_like(contrast)
    one = np.ones_like(contrast)
    scale = one + contrast
    bias = -contrast * center
    expected_ctm = np.array(
        [
            [scale, zero, zero, zero],
            [zero, scale, zero, zero],
            [zero, zero, scale, zero],
            [bias, bias, bias, one],
        ]
    )
    if batch_size is not None:
        # Swap the batch dimension first.
        expected_ctm = np.transpose(expected_ctm, [2, 0, 1])

    np.testing.assert_allclose(
        expected_ctm,
        m_np,
        atol=1e-2,
        err_msg="Contrast matrix does not match expected value.",
    )


@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("hue", [0.0, 360.0])
@pytest.mark.parametrize("saturation", [0.0, 1.0])
def test_hue_saturation_matrix(batch_size, hue, saturation):
    """
    Test the hue and saturation matrix.

    The tests are quite tolerant because a perfect HSV conversion cannot be done with a linear
    matrices. For more information, review the docs of the method.
    """
    check_identity = hue in [0.0, 360.0] and saturation == 1.0
    zero_saturation = saturation == 0.0

    if batch_size is not None:
        hue = np.tile(hue, [batch_size])
        saturation = np.tile(saturation, [batch_size])

    m = color.hue_saturation_matrix(hue=hue, saturation=saturation)
    m_np = tf.compat.v1.Session().run(m)

    if batch_size is None:
        assert m_np.shape == (4, 4)
    else:
        assert m_np.shape == (batch_size, 4, 4)

    if check_identity:
        np.testing.assert_allclose(
            identity_color_matrix(batch_size),
            m_np,
            atol=1e-2,
            err_msg="No hue and saturation changed did not result in the "
            "identity matrix.",
        )

    # Zero saturation should result in equal weighting of all channels
    if zero_saturation:
        for c in range(1, 3):
            # Compare the 2nd and 3rd channel with the first.
            if batch_size is not None:
                m0 = m_np[:, 0:3, 0]
                mc = m_np[:, 0:3, c]
            else:
                m0 = m_np[0:3, 0]
                mc = m_np[0:3, c]
            np.testing.assert_array_equal(
                m0,
                mc,
                err_msg="Zero saturation resulted in differences across " "channels.",
            )


@pytest.mark.parametrize("batch_size", [None, 10])
def test_hue_saturation_matrix2(batch_size):
    """Test that hue and saturation matrix matches expected value."""
    if batch_size is None:
        hue = 45.0
        saturation = 1.0
    else:
        hue = np.linspace(0.0, 360.0, batch_size)
        saturation = np.linspace(0.0, 2.0, batch_size)

    m = color.hue_saturation_matrix(hue=hue, saturation=saturation)
    m_np = tf.compat.v1.Session().run(m)

    const_mat = np.array(
        [
            [0.299, 0.299, 0.299, 0.0],
            [0.587, 0.587, 0.587, 0.0],
            [0.114, 0.114, 0.114, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sch_mat = np.array(
        [
            [0.701, -0.299, -0.300, 0.0],
            [-0.587, 0.413, -0.588, 0.0],
            [-0.114, -0.114, 0.886, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    ssh_mat = np.array(
        [
            [0.168, -0.328, 1.25, 0.0],
            [0.330, 0.035, -1.05, 0.0],
            [-0.497, 0.292, -0.203, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    angle = hue * (np.pi / 180.0)

    if batch_size is not None:
        const_mat = np.tile(const_mat, [batch_size, 1, 1])
        sch_mat = np.tile(sch_mat, [batch_size, 1, 1])
        ssh_mat = np.tile(ssh_mat, [batch_size, 1, 1])
        angle = np.reshape(angle, [batch_size, 1, 1])
        saturation = np.reshape(saturation, [batch_size, 1, 1])

    expected_ctm = const_mat + saturation * (
        np.cos(angle) * sch_mat + np.sin(angle) * ssh_mat
    )

    np.testing.assert_allclose(
        expected_ctm,
        m_np,
        atol=1e-2,
        err_msg="Hue and saturation matrix does not match expected value.",
    )


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.color.hue_saturation_matrix",
    side_effect=lambda h, s: (h, s),
)
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("hue_rotation_max", [0.0, 100.0])
@pytest.mark.parametrize("saturation_shift_max", [0.0, 0.5])
@pytest.mark.parametrize("hue_center", [0.0, 50.0])
@pytest.mark.parametrize("saturation_shift_min", [-100.0, 0.0, None])
def test_random_hue_saturation_matrix(
    patched,
    batch_size,
    hue_rotation_max,
    saturation_shift_max,
    hue_center,
    saturation_shift_min,
):
    """Test that random_hue_saturation_matrix produces correct distributions."""
    set_random_seed(42)
    tensors = color.random_hue_saturation_matrix(
        hue_rotation_max,
        saturation_shift_max,
        batch_size=batch_size,
        hue_center=hue_center,
        saturation_shift_min=saturation_shift_min,
    )
    hue_rotations, saturation_shifts = sample_tensors(tensors, NUM_SAMPLES)

    assert_truncated_normal_distribution(
        hue_rotations, mean=hue_center, stddev=hue_rotation_max / 2.0
    )

    if saturation_shift_min is None:
        saturation_shift_min = -saturation_shift_max

    min_bound = 1.0 + saturation_shift_min
    max_bound = 1.0 + saturation_shift_max
    assert_uniform_distribution(saturation_shifts, min_bound, max_bound)


@mock.patch("nvidia_tao_tf1.core.processors.augment.color.tf.random.truncated_normal")
@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.color.hue_saturation_matrix",
    side_effect=color.hue_saturation_matrix,
)
@pytest.mark.parametrize("batch_size", [None, 4])
def test_random_hue_saturation_matrix_samples_hue(
    mocked_hue_saturation_matrix, mocked_truncated_normal, batch_size
):
    hue = tf.constant(42, dtype=tf.float32)
    mocked_truncated_normal.return_value = hue
    color.random_hue_saturation_matrix(
        hue_rotation_max=180.0, saturation_shift_max=0.0, batch_size=batch_size
    )

    expected_shape = [] if batch_size is None else [batch_size]
    mocked_truncated_normal.assert_called_with(expected_shape, mean=0.0, stddev=90.0)
    mocked_hue_saturation_matrix.assert_called_with(hue, mock.ANY)


@mock.patch("nvidia_tao_tf1.core.processors.augment.color.tf.random.uniform")
@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.color.hue_saturation_matrix",
    side_effect=color.hue_saturation_matrix,
)
@pytest.mark.parametrize("batch_size", [None, 4])
def test_random_hue_saturation_matrix_samples_saturation(
    mocked_hue_saturation_matrix, mocked_random_uniform, batch_size
):
    saturation = 0.42
    mocked_random_uniform.return_value = saturation
    color.random_hue_saturation_matrix(
        hue_rotation_max=0.0, saturation_shift_max=0.5, batch_size=batch_size
    )
    expected_shape = [] if batch_size is None else [batch_size]
    mocked_random_uniform.assert_called_with(expected_shape, minval=-0.5, maxval=0.5)
    mocked_hue_saturation_matrix.assert_called_with(mock.ANY, 1 + saturation)


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.color.contrast_matrix",
    side_effect=lambda c, cs: (c, cs),
)
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("contrast_scale_max", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("contrast_center", [1.0 / 2.0, 255.0 / 2.0])
@pytest.mark.parametrize("contrast_scale_center", [0.0, 0.5, 1.0])
def test_random_contrast_matrix(
    patched, batch_size, contrast_scale_max, contrast_center, contrast_scale_center
):
    """Test that random_contrast_matrix produces correct distributions."""
    set_random_seed(42)
    contrast_scale_tensor, contrast_center_value = color.random_contrast_matrix(
        contrast_scale_max,
        contrast_center,
        batch_size=batch_size,
        scale_center=contrast_scale_center,
    )
    contrast_scales = sample_tensors([contrast_scale_tensor], NUM_SAMPLES)

    assert_truncated_normal_distribution(
        contrast_scales, mean=contrast_scale_center, stddev=contrast_scale_max / 2.0
    )

    assert contrast_center == contrast_center_value


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.color.brightness_offset_matrix",
    side_effect=lambda offset: offset,
)
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("brightness_scale_max", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("brightness_uniform_across_channels", [True, False])
@pytest.mark.parametrize("brightness_center", [-0.5, 0.0, 0.5])
def test_random_brightness_matrix(
    patched,
    batch_size,
    brightness_scale_max,
    brightness_uniform_across_channels,
    brightness_center,
):
    """Test that random_brightness_matrix produces correct distributions."""
    set_random_seed(42)
    brightness_scale_tensor = color.random_brightness_matrix(
        brightness_scale_max,
        brightness_uniform_across_channels,
        batch_size=batch_size,
        brightness_center=brightness_center,
    )

    brightness_scales = sample_tensors([brightness_scale_tensor], NUM_SAMPLES)
    brightness_scales = np.array(brightness_scales[0])

    assert_truncated_normal_distribution(
        brightness_scales, mean=brightness_center, stddev=brightness_scale_max / 2.0
    )

    if brightness_uniform_across_channels:
        # If ``brightness_uniform_across_channels`` is True, check that values for each channel
        # match. This is done by subtracting value of red channel from all channels and checking
        # that result is zero.
        if batch_size is None:
            assert all(
                [
                    np.allclose(brightness_scales[i, :] - brightness_scales[i, 0], 0.0)
                    for i in xrange(len(brightness_scales))
                ]
            )
        else:
            for b in xrange(len(brightness_scales)):
                assert all(
                    [
                        np.allclose(
                            brightness_scales[b, i, :] - brightness_scales[b, i, 0], 0.0
                        )
                        for i in xrange(len(brightness_scales[b]))
                    ]
                )
    elif brightness_scale_max > 0.0:
        # If ``brightness_uniform_across_channels`` is False, check that values for each channel
        # match. This is done by negating test for ``brightness_uniform_across_channels`` True.
        # Note that we're not checking value of red channel after subtracting value of red channel
        # since that will be always zero. Similarly, values will be all zero and hence the same
        # if ``brightness_scale_max`` == 0.0.
        if batch_size is None:
            assert all(
                [
                    not np.allclose(
                        brightness_scales[i, 1:] - brightness_scales[i, 0], 0.0
                    )
                    for i in xrange(len(brightness_scales))
                ]
            )
        else:
            for b in xrange(len(brightness_scales)):
                assert all(
                    [
                        not np.allclose(
                            brightness_scales[b, i, 1:] - brightness_scales[b, i, 0],
                            0.0,
                        )
                        for i in xrange(len(brightness_scales[b]))
                    ]
                )


@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize(
    "hue_rotation_max, saturation_shift_max, contrast_scale_max, "
    "brightness_scale_max, brightness_uniform_across_channels",
    [
        (0, 0, 0, 0, True),
        (0, 0, 0, 0, False),
        (0, 0, 0, 0.5, True),
        (0, 0, 0, 0.5, False),
    ],
)
def test_get_random_color_transformation_matrix(
    batch_size,
    hue_rotation_max,
    saturation_shift_max,
    contrast_scale_max,
    brightness_scale_max,
    brightness_uniform_across_channels,
):
    """
    Test generate random color transform matrix.
    """
    set_random_seed(42)
    # No linter approved way to break up the brightness_uniform_across_channels=
    # brightness_uniform_across_channels line and maintain indentation, so using
    # a dummy variable.
    uniform_bright = brightness_uniform_across_channels
    ctm = color.get_random_color_transformation_matrix(
        hue_rotation_max=hue_rotation_max,
        saturation_shift_max=saturation_shift_max,
        contrast_scale_max=contrast_scale_max,
        contrast_center=0.5,
        brightness_scale_max=brightness_scale_max,
        brightness_uniform_across_channels=uniform_bright,
        batch_size=batch_size,
    )
    ctm_np = tf.compat.v1.Session().run(ctm)
    if brightness_scale_max > 0:
        if batch_size is None:
            ctm = ctm_np[3, 0:3]
        else:
            ctm = ctm_np[:, 3, 0:3]
        if brightness_uniform_across_channels:
            # Tests that the first three values in the last row of the transform matrix
            # (the offset channels) have the same value.
            np.testing.assert_allclose(
                np.sum(np.diff(ctm)),
                0,
                atol=1e-2,
                err_msg="color transform matrix is not correctly "
                "generated when brightness is uniform.",
            )
        else:
            # Tests that the first three values in the last row of the transform matrix
            # (the offset channels) have different values.
            np.testing.assert_equal(
                np.not_equal(np.sum(np.diff(ctm)), 0),
                True,
                err_msg="color transform matrix is not correctly "
                "generated when brightness is not uniform.",
            )
    else:
        np.testing.assert_allclose(
            identity_color_matrix(batch_size),
            ctm_np,
            atol=1e-2,
            err_msg="color transform matrix is not correctly generated.",
        )


def test_no_op_color_transform():
    """Tests that supplying no kwargs results in an almost-no-op color transformation matrix."""
    ctm = color.get_random_color_transformation_matrix()
    ctm_np = tf.compat.v1.Session().run(ctm)
    # 'Almostness' comes from saturation matrix.
    np.testing.assert_allclose(
        ctm_np,
        np.eye(4),
        atol=2e-3,
        verbose=True,
        err_msg="Default color transformation matrix is too 'far' from the identity matrix.",
    )
