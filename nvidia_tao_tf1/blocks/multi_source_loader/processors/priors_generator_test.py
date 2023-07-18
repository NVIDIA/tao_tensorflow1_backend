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

"""Main test for PriorsGenerator object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.priors_generator import (
    PriorsGenerator,
)

_COORDINATES_PER_POINT = 4
_FEATURE_MAP_SIZES = [(3, 5)]
_FLOATING_POINT_TOLERANCE = 0.01
_IMAGE_HEIGHT = 504
_IMAGE_WIDTH = 960
_NPOINT_PRIORS = 1
_NLINEAR_PRIORS = 8
_NPRIORS = _NPOINT_PRIORS + _NLINEAR_PRIORS
_POINTS_PER_PRIOR = 2
_PRIOR_THRESHOLD = 0.1


@pytest.fixture(scope="session")
def _priors_generator():
    return PriorsGenerator(
        _NPOINT_PRIORS,
        _NLINEAR_PRIORS,
        _POINTS_PER_PRIOR,
        _PRIOR_THRESHOLD,
        _FEATURE_MAP_SIZES,
        _IMAGE_HEIGHT,
        _IMAGE_WIDTH,
    )


# Test all input parameters with positive and negative values.
transform_tests = [
    # Test rotation clockwise.
    ((0, 0), 1.0, 0.0, -np.pi / 2, 1.0, 1.0, 0.0, 0.0, (0.0, -1.0)),
    # Test rotation counter-clockwise.
    ((0, 0), 1.0, 0.0, np.pi / 2, 1.0, 1.0, 0.0, 0.0, (0.0, 1.0)),
    # Test rotation clockwise about other end.
    ((1.0, 0), 0.0, 0.0, -np.pi / 2, 1.0, 1.0, 0.0, 0.0, (1.0, 1.0)),
    # Test rotation counter-clockwiseabout other end.
    ((1.0, 0), 0.0, 0.0, np.pi / 2, 1.0, 1.0, 0.0, 0.0, (1.0, -1.0)),
    # Test scale up x.
    ((0, 0), 1.0, 1.0, 0.0, 10.0, 1.0, 0.0, 0.0, (10.0, 1.0)),
    # Test scale up y.
    ((0, 0), 1.0, 1.0, 0.0, 1.0, 10.0, 0.0, 0.0, (1.0, 10.0)),
    # Test scale down x.
    ((0, 0), 1.0, 1.0, 0.0, 0.1, 1.0, 0.0, 0.0, (0.1, 1.0)),
    # Test scale down y.
    ((0, 0), 1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 0.0, (1.0, 0.1)),
    # Test translate x to the left.
    ((0, 0), 1.0, 1.0, 0.0, 1.0, 1.0, -10.0, 0.0, (-9.0, 1.0)),
    # Test translate y downward.
    ((0, 0), 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, -10.0, (1.0, -9.0)),
    # Test translate x to the right.
    ((0, 0), 1.0, 1.0, 0.0, 1.0, 1.0, 10.0, 0.0, (11.0, 1.0)),
    # Test translate y upward.
    ((0, 0), 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 10.0, (1.0, 11.0)),
]


@pytest.mark.parametrize(
    "origin, xs, ys, angle, scale_x, scale_y, tx, ty, expected_point", transform_tests
)
def test_transform_prior(
    _priors_generator, origin, xs, ys, angle, scale_x, scale_y, tx, ty, expected_point
):
    """Test the rotation, translation and scaling functionality."""
    transformed_point = _priors_generator._transform_prior(
        origin, xs, ys, angle, scale_x, scale_y, tx, ty
    )

    with tf.compat.v1.Session() as sess:
        actual_point = sess.run(tf.reshape(transformed_point, [-1]))
    np.testing.assert_almost_equal(actual_point, expected_point, decimal=4)


def test_get_prior_locations(_priors_generator):
    """Test locations of priors encoded correctly."""
    receptive_field_x = np.floor(_IMAGE_WIDTH / 5.0)
    receptive_field_y = np.floor(_IMAGE_HEIGHT / 3.0)

    prior_locations = _priors_generator._get_prior_locations(
        receptive_field_x,
        receptive_field_y,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
    )

    assert prior_locations[0].shape == (3, 5)
    assert prior_locations[0][0][0] == (_IMAGE_WIDTH / 5.0) / 2.0
    assert prior_locations[1][0][0] == (_IMAGE_HEIGHT / 3.0) / 2.0


# Test point priors.
point_prior_tests = [
    # One prior.
    (100.0, 200.0, [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0])
]


@pytest.mark.parametrize("tx, ty, expected_priors", point_prior_tests)
def test_generate_point_priors(_priors_generator, tx, ty, expected_priors):
    """Test that correct number and shape of point priors are created."""
    priors = _priors_generator._generate_point_priors(
        tx, ty, image_width=1, image_height=1
    )
    with tf.compat.v1.Session() as sess:
        actual_priors = sess.run(tf.reshape(priors, [-1]))

    np.testing.assert_array_almost_equal(actual_priors, expected_priors, decimal=4)


# Test one and three linear priors.
linear_prior_tests = [
    # One prior.
    (
        [0.5 * np.pi],
        1.0,
        1.0,
        100.0,
        100.0,
        [0.1562, 0.2004, 0.0521, 0.2004, 0.1562, 0.1964, 0.0521, 0.1964],
    ),
    # Three priors, invalid angles.
    ([0.0, -0.5 * np.pi, 1.5 * np.pi], 1.0, 1.0, 100.0, 100.0, []),
    # Three priors.
    (
        [0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi],
        1.0,
        1.0,
        100.0,
        100.0,
        [
            0.1417,
            0.1297,
            0.0681,
            0.2700,
            0.1403,
            0.1269,
            0.0666,
            0.2672,
            0.1562,
            0.2004,
            0.0521,
            0.2004,
            0.1562,
            0.1964,
            0.0521,
            0.1964,
            0.1403,
            0.2700,
            0.0666,
            0.1297,
            0.1417,
            0.2672,
            0.0681,
            0.1269,
        ],
    ),
]


@pytest.mark.parametrize(
    "angles, scale_x, scale_y, tx, ty, expected_priors", linear_prior_tests
)
def test_generate_linear_priors(
    _priors_generator, angles, scale_x, scale_y, tx, ty, expected_priors
):
    """Test that correct number and shape of linear priors are created."""
    if all(a >= 0.0 for a in angles) and all(a <= np.pi for a in angles):
        priors = _priors_generator._generate_linear_priors(
            angles,
            scale_x,
            scale_y,
            tx,
            ty,
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
        )
        with tf.compat.v1.Session() as sess:
            actual_priors = sess.run(tf.reshape(priors, [-1]))

        np.testing.assert_array_almost_equal(actual_priors, expected_priors, decimal=4)
    else:
        with pytest.raises(Exception):
            priors = _priors_generator._generate_linear_priors(
                angles,
                scale_x,
                scale_y,
                tx,
                ty,
                image_height=_IMAGE_HEIGHT,
                image_width=_IMAGE_WIDTH,
            )


# Test two different feature map sizes and more than one feature map.
prior_tests = [
    # One feature map.
    ([(5, 3)], 540),
    # Second feature map.
    ([(10, 6)], 2056),
    # Two feature maps.
    ([(2, 1), (5, 3)], 612),
]


@pytest.mark.parametrize("feature_map_sizes, expected_prior_points", prior_tests)
def test_get_priors(_priors_generator, feature_map_sizes, expected_prior_points):
    """Test that the correct priors are generated."""
    priors = _priors_generator._get_priors(
        feature_map_sizes, _IMAGE_HEIGHT, _IMAGE_WIDTH
    )
    with tf.compat.v1.Session() as sess:
        prior_x, prior_y = sess.run(tf.split(priors, num_or_size_splits=2, axis=1))

    assert len(prior_x) == expected_prior_points
    assert len(prior_y) == expected_prior_points


# Test two different feature map sizes and more than one feature map.
nprior_tests = [
    # One feature map.
    ([(5, 3)], 135),
    # Second feature map.
    ([(10, 6)], 540),
    # Two feature maps.
    ([(2, 1), (5, 3)], 153),
]


@pytest.mark.parametrize("feature_map_sizes, expected_nall_priors", nprior_tests)
def test_get_nall_priors(_priors_generator, feature_map_sizes, expected_nall_priors):
    """Test that the correct priors are generated."""
    nall_priors = _priors_generator._get_nall_priors(feature_map_sizes)

    assert nall_priors == expected_nall_priors


# Test zero and a combination of npriors.
prior_nprior_tests = [
    # Only point priors.
    (1, 0),
    # Only linear priors.
    (0, 8),
    # Linear and point priors.
    (1, 8),
]


@pytest.mark.parametrize("npoint_priors, nlinear_priors", prior_nprior_tests)
def test_get_priors_npriors(_priors_generator, npoint_priors, nlinear_priors):
    """Test that the correct priors are generated."""
    nprior_locations = sum([np.prod(fmaps) for fmaps in _FEATURE_MAP_SIZES])
    _priors_generator.npoint_priors = npoint_priors
    _priors_generator.nlinear_priors = nlinear_priors
    priors = _priors_generator._get_priors(
        _FEATURE_MAP_SIZES, _IMAGE_HEIGHT, _IMAGE_WIDTH
    )
    with tf.compat.v1.Session() as sess:
        prior_x, prior_y = sess.run(tf.split(priors, num_or_size_splits=2, axis=1))

    assert len(prior_x) == (
        _COORDINATES_PER_POINT / 2
    ) * _POINTS_PER_PRIOR * nprior_locations * (npoint_priors + nlinear_priors)
    assert len(prior_y) == (
        _COORDINATES_PER_POINT / 2
    ) * _POINTS_PER_PRIOR * nprior_locations * (npoint_priors + nlinear_priors)


def test_raise_error_with_negative_npoints_priors():
    with pytest.raises(ValueError) as exception:
        PriorsGenerator(
            npoint_priors=-1,
            nlinear_priors=_NLINEAR_PRIORS,
            points_per_prior=_POINTS_PER_PRIOR,
            prior_threshold=_PRIOR_THRESHOLD,
            feature_map_sizes=_FEATURE_MAP_SIZES,
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
        )
    assert str(exception.value) == "npoints_priors must be positive, it is -1."


def test_raise_error_with_negative_nlinear_priors():
    with pytest.raises(ValueError) as exception:
        PriorsGenerator(
            npoint_priors=_NPOINT_PRIORS,
            nlinear_priors=-1,
            points_per_prior=_POINTS_PER_PRIOR,
            prior_threshold=_PRIOR_THRESHOLD,
            feature_map_sizes=_FEATURE_MAP_SIZES,
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
        )
    assert str(exception.value) == "nlinear_priors must be positive, it is -1."


def test_raise_error_with_zero_npriors():
    with pytest.raises(ValueError) as exception:
        PriorsGenerator(
            npoint_priors=0,
            nlinear_priors=0,
            points_per_prior=_POINTS_PER_PRIOR,
            prior_threshold=_PRIOR_THRESHOLD,
            feature_map_sizes=_FEATURE_MAP_SIZES,
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
        )
    assert str(exception.value) == "npriors must be > 0, it is 0."


def test_raise_error_with_zero_points_per_prior():
    with pytest.raises(ValueError) as exception:
        PriorsGenerator(
            npoint_priors=_NPOINT_PRIORS,
            nlinear_priors=_NLINEAR_PRIORS,
            points_per_prior=0,
            prior_threshold=_PRIOR_THRESHOLD,
            feature_map_sizes=_FEATURE_MAP_SIZES,
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
        )
    assert str(exception.value) == "points_per_prior must be positive, not 0."


@mock.patch(
    "nvidia_tao_tf1.blocks.multi_source_loader.processors.priors_generator_test.\
PriorsGenerator._get_nall_priors",
    return_value=0,
)
def test_raise_error_with_zero_nall_priors(mocked_get_nall_priors):
    with pytest.raises(ValueError) as exception:
        PriorsGenerator(
            npoint_priors=_NPOINT_PRIORS,
            nlinear_priors=_NLINEAR_PRIORS,
            points_per_prior=_POINTS_PER_PRIOR,
            prior_threshold=_PRIOR_THRESHOLD,
            feature_map_sizes=_FEATURE_MAP_SIZES,
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
        )
    assert str(exception.value) == "There must be at least one prior, instead 0."


@mock.patch(
    "nvidia_tao_tf1.blocks.multi_source_loader.processors.priors_generator_test.\
PriorsGenerator._get_priors",
    return_value=None,
)
def test_raise_error_with_no_priors(mocked_get_priors):
    with pytest.raises(ValueError) as exception:
        PriorsGenerator(
            npoint_priors=_NPOINT_PRIORS,
            nlinear_priors=_NLINEAR_PRIORS,
            points_per_prior=_POINTS_PER_PRIOR,
            prior_threshold=_PRIOR_THRESHOLD,
            feature_map_sizes=_FEATURE_MAP_SIZES,
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
        )
    assert str(exception.value) == "There is not any prior set."
