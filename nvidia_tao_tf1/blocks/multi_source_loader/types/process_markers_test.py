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

"""Test functions that process front / back markers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.process_markers import (
    FRONT_BACK_TOLERANCE,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.process_markers import (
    map_markers_to_orientations,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.process_markers import (
    map_orientation_to_markers,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.process_markers import (
    SIDE_ONLY_TOLERANCE,
)


TEST_INVALID_ORIENTATION = -123.0

# First, define cases that can be used for both translation functions.
_COMMON_CASES = [
    # front_markers, back_markers, ref_angle, orientations
    # Front + right side.
    ([0.5, 0.4], [0.0] * 2, 0.0, False, [-0.75 * np.pi, -0.8 * np.pi]),
    # Front + left side.
    (
        [0.2, 0.5, 0.71],
        [1.0] * 3,
        0.0,
        False,
        [0.6 * np.pi, 0.75 * np.pi, 0.855 * np.pi],
    ),
    # Back + right side.
    ([1.0] * 2, [0.4, 0.7], 0.0, False, [-0.3 * np.pi, -0.15 * np.pi]),
    # Back + left side.
    ([0.0] * 3, [0.1, 0.3, 0.8], 0.0, False, [0.05 * np.pi, 0.15 * np.pi, 0.4 * np.pi]),
    # Left only.
    ([0.0] * 5, [1.0] * 5, 0.0, False, [np.pi / 2.0] * 5),
    # Right only.
    ([1.0] * 5, [0.0] * 5, 0.0, False, [-np.pi / 2.0] * 5),
    # Legacy (rumpy) test cases.
    # Left only.
    ([0.0] * 5, [1.0] * 5, -np.pi / 2.0, True, [np.pi] * 5),
    # Back + left side.
    (
        [0.0] * 3,
        [0.1, 0.3, 0.8],
        -np.pi / 2.0,
        True,
        [-0.55 * np.pi, -0.65 * np.pi, -0.9 * np.pi],
    ),
    # Back + right side.
    ([1.0] * 2, [0.4, 0.7], -np.pi / 2.0, True, [-0.2 * np.pi, -0.35 * np.pi]),
    # Front + right side.
    ([0.5, 0.4], [0.0] * 2, -np.pi / 2.0, True, [0.25 * np.pi, 0.3 * np.pi]),
    # Front + left side.
    (
        [0.2, 0.5, 0.71],
        [1.0] * 3,
        -np.pi / 2.0,
        True,
        [0.9 * np.pi, 0.75 * np.pi, 0.645 * np.pi],
    ),
]


@pytest.mark.parametrize(
    "front_markers,back_markers,ref_angle,clockwise,expected_orientations",
    [
        # Illegal / nonsensical marker values map to TEST_INVALID_ORIENTATION.
        (
            [-0.5, 5.0, 0.5],
            [0.0, -5.0, -1.0],
            0.0,
            False,
            [TEST_INVALID_ORIENTATION] * 3,
        ),
        # Same test but with values that should be rounded.
        ([0.005, 0.991], [0.009, 0.996], 0.0, False, [TEST_INVALID_ORIENTATION] * 2),
        # Invalid and valid values are both present. Valid value is front + right.
        ([0.0, 0.4], [0.0] * 2, 0.0, False, [TEST_INVALID_ORIENTATION, -0.8 * np.pi]),
        # Back only: 0.0.
        ([-1.0] * 2, [0.0, 1.0], 0.0, False, [0.0] * 2),
        # Front only: pi (180 degrees).
        ([0.0, 1.0], [-1.0] * 2, 0.0, False, [np.pi] * 2),
        # Back only, rumpy legacy system.
        ([-1.0] * 2, [0.0, 1.0], -np.pi / 2.0, True, [-np.pi / 2.0] * 2),
    ]
    + _COMMON_CASES,
)
def test_markers_to_orientations(
    front_markers, back_markers, ref_angle, clockwise, expected_orientations
):
    """Test that the translation from (front_marker, back_marker) to orientations is correct."""
    # First, get tensors for the markers.
    front_markers_tensor, back_markers_tensor = map(
        tf.constant, [front_markers, back_markers]
    )
    # Then, translate to orientation values.
    orientations_tensor = map_markers_to_orientations(
        front_markers=front_markers_tensor,
        invalid_orientation=TEST_INVALID_ORIENTATION,
        back_markers=back_markers_tensor,
        ref_angle=ref_angle,
        clockwise=clockwise,
    )
    # Evaluate.
    with tf.compat.v1.Session() as session:
        orientations = session.run(orientations_tensor)
    # Check values are correct.
    assert np.allclose(orientations, expected_orientations)


@pytest.mark.parametrize(
    "expected_front_markers,expected_back_markers,ref_angle,clockwise,orientations",
    [
        # Front only. Here, we are slightly below the tolerance value away from the ref_angle.
        (
            [-1.0] * 2,
            [0.0] * 2,
            0.0,
            True,
            [-0.99 * FRONT_BACK_TOLERANCE, 0.99 * FRONT_BACK_TOLERANCE],
        ),
        # Same, but anticlockwise.
        (
            [-1.0] * 2,
            [0.0] * 2,
            0.0,
            False,
            [-0.99 * FRONT_BACK_TOLERANCE, 0.99 * FRONT_BACK_TOLERANCE],
        ),
        # Same, but with Rumpy scheme.
        (
            [-1.0] * 2,
            [0.0] * 2,
            -np.pi / 2.0,
            True,
            [
                -np.pi / 2.0 - 0.99 * FRONT_BACK_TOLERANCE,
                -np.pi / 2.0 + 0.99 * FRONT_BACK_TOLERANCE,
            ],
        ),
        # Back only.
        (
            [0.0] * 2,
            [-1.0] * 2,
            0.0,
            True,
            [np.pi - 0.99 * FRONT_BACK_TOLERANCE, np.pi + 0.99 * FRONT_BACK_TOLERANCE],
        ),
        # Same, but anticlockwise.
        (
            [0.0] * 2,
            [-1.0] * 2,
            0.0,
            False,
            [np.pi - 0.99 * FRONT_BACK_TOLERANCE, -np.pi + 0.99 * FRONT_BACK_TOLERANCE],
        ),
        # Same, but with Rumpy scheme.
        (
            [0.0] * 2,
            [-1.0] * 2,
            -np.pi / 2.0,
            True,
            [
                np.pi / 2.0 - 0.99 * FRONT_BACK_TOLERANCE,
                np.pi / 2.0 + 0.99 * FRONT_BACK_TOLERANCE,
            ],
        ),
        # Left only.
        (
            [0.0] * 2,
            [1.0] * 2,
            0.0,
            False,
            [
                np.pi / 2.0 - 0.99 * SIDE_ONLY_TOLERANCE,
                np.pi / 2.0 + 0.99 * SIDE_ONLY_TOLERANCE,
            ],
        ),
        # Same, but clockwise.
        (
            [0.0] * 2,
            [1.0] * 2,
            0.0,
            True,
            [
                -np.pi / 2.0 - 0.99 * SIDE_ONLY_TOLERANCE,
                -np.pi / 2.0 + 0.99 * SIDE_ONLY_TOLERANCE,
            ],
        ),
        # Same, but with Rumpy scheme.
        (
            [0.0] * 2,
            [1.0] * 2,
            -np.pi / 2.0,
            True,
            [np.pi - 0.99 * SIDE_ONLY_TOLERANCE, -np.pi + 0.99 * SIDE_ONLY_TOLERANCE],
        ),
        # Right only.
        (
            [1.0] * 2,
            [0.0] * 2,
            0.0,
            False,
            [
                -np.pi / 2.0 - 0.99 * SIDE_ONLY_TOLERANCE,
                -np.pi / 2.0 + 0.99 * SIDE_ONLY_TOLERANCE,
            ],
        ),
        # Same, but clockwise.
        (
            [1.0] * 2,
            [0.0] * 2,
            0.0,
            True,
            [
                np.pi / 2.0 - 0.99 * SIDE_ONLY_TOLERANCE,
                np.pi / 2.0 + 0.99 * SIDE_ONLY_TOLERANCE,
            ],
        ),
        # Same, but with Rumpy scheme.
        (
            [1.0] * 2,
            [0.0] * 2,
            -np.pi / 2.0,
            True,
            [-0.99 * SIDE_ONLY_TOLERANCE, 0.99 * SIDE_ONLY_TOLERANCE],
        ),
    ]
    + _COMMON_CASES,
)
def test_map_orientation_to_markers(
    expected_front_markers, expected_back_markers, ref_angle, clockwise, orientations
):
    """Test that map_orientation_to_markers translates orientation back to markers correctly."""
    front_markers, back_markers = zip(
        *map(
            lambda x: map_orientation_to_markers(x, ref_angle, clockwise), orientations
        )
    )

    assert np.allclose(front_markers, expected_front_markers)
    assert np.allclose(back_markers, expected_back_markers)
