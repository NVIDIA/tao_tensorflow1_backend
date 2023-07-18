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
"""Tests for RandomGlimpse processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from six.moves import mock
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import RandomGlimpse
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "height, width, crop_location, crop_probability, message",
    [
        (-1, 1, "random", 1, "RandomGlimpse.height (-1) is not positive."),
        (1, -1, "random", 1, "RandomGlimpse.width (-1) is not positive."),
        (
            1,
            -1,
            "random",
            2,
            "RandomGlimpse.crop_probability (2) is not within the range [0, 1].",
        ),
        (
            1,
            -1,
            "none",
            2,
            "RandomGlimpse.crop_location 'none' is not "
            "supported. Valid options: center, random.",
        ),
    ],
)
def test_invalid_random_glimpse_parameters(
    height, width, crop_location, crop_probability, message
):
    """Test RandomGlimpse processor constructor error handling on invalid arguments."""
    with pytest.raises(ValueError) as exc:
        RandomGlimpse(
            height=height,
            width=width,
            crop_location=crop_location,
            crop_probability=crop_probability,
        )
    assert str(exc.value) == message


@pytest.mark.parametrize("batch_size", [None, 5])
def test_random_glimpse_call_with_center_crop(batch_size):
    """Test RandomGlimpse processor call center crop mode."""
    batch_shape = [] if batch_size is None else [batch_size]
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4, batch_shape=batch_shape, dtype=tf.float32),
        spatial_transform_matrix=tf.eye(3, batch_shape=batch_shape, dtype=tf.float32),
    )

    random_glimpse = RandomGlimpse(
        crop_location=RandomGlimpse.CENTER, crop_probability=1.0, height=6, width=5
    )

    expected_stm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 3.0, 1.0]])
    if batch_size is not None:
        expected_stm = np.tile(expected_stm, [batch_size, 1, 1])

    final_transform = random_glimpse(transform)
    stm = tf.compat.v1.Session().run(final_transform.spatial_transform_matrix)
    np.testing.assert_equal(stm, expected_stm)
    assert final_transform.canvas_shape == Canvas2D(6, 5)


@mock.patch("nvidia_tao_tf1.core.processors.augment.random_glimpse.tf.random.uniform")
@pytest.mark.parametrize("batch_size", [None, 5])
def test_random_glimpse_call_with_random_crop(mocked_random_uniform, batch_size):
    """Test RandomGlimpse processor call random crop mode."""
    batch_shape = [] if batch_size is None else [batch_size]

    # Fix random uniform to return 0.5, which will be the value for x and y of a translation
    # matrix.
    mocked_random_uniform.return_value = tf.constant(
        0.5, shape=batch_shape, dtype=tf.float32
    )

    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4, batch_shape=batch_shape, dtype=tf.float32),
        spatial_transform_matrix=tf.eye(3, batch_shape=batch_shape, dtype=tf.float32),
    )

    random_glimpse = RandomGlimpse(
        crop_location=RandomGlimpse.RANDOM, crop_probability=1.0, height=6, width=5
    )

    expected_stm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0]])
    if batch_size is not None:
        expected_stm = np.tile(expected_stm, [batch_size, 1, 1])

    final_transform = random_glimpse(transform)
    stm = tf.compat.v1.Session().run(final_transform.spatial_transform_matrix)
    np.testing.assert_equal(stm, expected_stm)
    assert final_transform.canvas_shape == Canvas2D(6, 5)


@pytest.mark.parametrize(
    "batch_size", [None, 5, tf.compat.v1.placeholder(dtype=tf.int32)]
)
def test_random_glimpse_call_with_scale(batch_size):
    """Test RandomGlimpse processor call scale mode."""
    batch_shape = [] if batch_size is None else [batch_size]
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4, batch_shape=batch_shape, dtype=tf.float32),
        spatial_transform_matrix=tf.eye(3, batch_shape=batch_shape, dtype=tf.float32),
    )

    feed_dict = {}
    expected_batch_size = batch_size
    if type(batch_size) == tf.Tensor:
        expected_batch_size = 7
        feed_dict = {batch_size: expected_batch_size}

    random_glimpse = RandomGlimpse(
        crop_location=RandomGlimpse.CENTER, crop_probability=0.0, height=6, width=5
    )

    expected_stm = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    if batch_size is not None:
        expected_stm = np.tile(expected_stm, [expected_batch_size, 1, 1])

    final_transform = random_glimpse(transform)
    stm = tf.compat.v1.Session().run(
        final_transform.spatial_transform_matrix, feed_dict=feed_dict
    )
    np.testing.assert_equal(stm, expected_stm)
    assert final_transform.canvas_shape == Canvas2D(6, 5)


def test_random_glimpse_call_with_invalid_input():
    """Test RandomGlimpse processor call error handling on invalid input types."""
    # Calling RandomGlimpse with str should throw a TypeError.
    with pytest.raises(TypeError):
        RandomGlimpse(1, 1, RandomGlimpse.CENTER, 1)("Transform")


@pytest.mark.parametrize(
    "crop_location, height, width, message",
    [
        (
            RandomGlimpse.RANDOM,
            5,
            6,
            "Attempted to extract random crop (6) wider than input width (5).",
        ),
        (
            RandomGlimpse.RANDOM,
            6,
            5,
            "Attempted to extract random crop (6) taller than input height (5).",
        ),
        (
            RandomGlimpse.CENTER,
            5,
            6,
            "Attempted to extract center crop (6) wider than input width (5).",
        ),
        (
            RandomGlimpse.CENTER,
            6,
            5,
            "Attempted to extract center crop (6) taller than input height (5).",
        ),
        ("unknown", 5, 5, "Unhandled crop location: 'unknown'."),
    ],
)
def test_random_glimpse_invalid_crop_configurations(
    crop_location, height, width, message
):
    """Test RandomGlimpse processor call error raising for invalid crop configurations."""
    transform = Transform(
        canvas_shape=Canvas2D(5, 5),
        color_transform_matrix=tf.eye(4),
        spatial_transform_matrix=tf.eye(3),
    )
    # Bypass constructor validation.
    if crop_location == "unknown":
        RandomGlimpse.CROP_LOCATIONS.append("unknown")
    random_glimpse = RandomGlimpse(
        crop_location=crop_location, crop_probability=1.0, height=height, width=width
    )

    with pytest.raises(ValueError) as exc:
        random_glimpse(transform)
    assert str(exc.value) == message


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    random_glimpse = RandomGlimpse(
        height=12, width=10, crop_location=RandomGlimpse.CENTER, crop_probability=1.0
    )
    random_glimpse_dict = random_glimpse.serialize()
    deserialized_random_glimpse = deserialize_tao_object(random_glimpse_dict)
    assert random_glimpse._height == deserialized_random_glimpse._height
    assert random_glimpse._width == deserialized_random_glimpse._width
    assert random_glimpse._crop_location == deserialized_random_glimpse._crop_location
    assert (
        random_glimpse._crop_probability
        == deserialized_random_glimpse._crop_probability
    )
