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
"""Tests for RandomFlip processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from six.moves import mock
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import RandomFlip
from nvidia_tao_tf1.core.processors.augment.spatial import flip_matrix
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "horizontal_probability, vertical_probability, message",
    [
        (
            2,
            0,
            "RandomFlip.horizontal_probability (2)"
            " is not within the range [0.0, 1.0].",
        ),
        (
            -1,
            0,
            "RandomFlip.horizontal_probability (-1)"
            " is not within the range [0.0, 1.0].",
        ),
        (
            0,
            2,
            "RandomFlip.vertical_probability (2)"
            " is not within the range [0.0, 1.0].",
        ),
        (
            0,
            -1,
            "RandomFlip.vertical_probability (-1)"
            " is not within the range [0.0, 1.0].",
        ),
    ],
)
def test_invalid_flip_probability(
    horizontal_probability, vertical_probability, message
):
    """Test RandomFlip processor constructor error handling on invalid flip probability."""
    with pytest.raises(ValueError) as exc:
        RandomFlip(
            horizontal_probability=horizontal_probability,
            vertical_probability=vertical_probability,
        )
    assert str(exc.value) == message


@mock.patch("nvidia_tao_tf1.core.processors.augment.random_flip.spatial.random_flip_matrix")
@pytest.mark.parametrize(
    "horizontal_probability, vertical_probability", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
)
def test_random_flip_call(
    mocked_random_flip_matrix, horizontal_probability, vertical_probability
):
    """Test RandomFlip processor call."""
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4),
        spatial_transform_matrix=tf.eye(3),
    )
    mocked_random_flip_matrix.return_value = tf.eye(3)
    processor = RandomFlip(
        horizontal_probability=horizontal_probability,
        vertical_probability=vertical_probability,
    )
    processor(transform)
    mocked_random_flip_matrix.assert_called_with(
        horizontal_probability=horizontal_probability,
        vertical_probability=vertical_probability,
        height=12,
        width=10,
        batch_size=None,
    )


def test_random_flip_call_with_invalid_input():
    """Test RandomTranslation processor call error handling on invalid input types."""
    # Calling RandomTranslation with str should throw a TypeError.
    with pytest.raises(TypeError):
        RandomFlip(0)("Transform")


@mock.patch("nvidia_tao_tf1.core.processors.augment.spatial.tf.random.uniform")
@pytest.mark.parametrize(
    "batch_size", [None, 5, tf.compat.v1.placeholder(dtype=tf.int32)]
)
def test_random_flip(mocked_random_uniform, batch_size):
    """Test RandomFlip processor."""
    batch_shape = [] if batch_size is None else [batch_size]
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4, batch_shape=batch_shape, dtype=tf.float32),
        spatial_transform_matrix=tf.eye(3, batch_shape=batch_shape, dtype=tf.float32),
    )

    feed_dict = {}
    if type(batch_size) == tf.Tensor:
        feed_dict = {batch_size: 7}

    rnd_prob = 0.0
    expected_x = True
    expected_y = True
    if batch_size is not None:
        # Generate a sequence of probabilities [0., 1., 0., 1., ...] so that every second
        # sample gets randomly transformed.
        float_batch_size = tf.cast(batch_size, tf.float32)
        rnd_prob = tf.math.floormod(
            tf.linspace(0.0, float_batch_size - 1.0, batch_size), 2.0
        )
        expected_x = tf.cast(1.0 - rnd_prob, tf.bool)
        expected_y = tf.cast(1.0 - rnd_prob, tf.bool)

    mocked_random_uniform.return_value = rnd_prob

    processor = RandomFlip(horizontal_probability=0.5, vertical_probability=0.5)
    stm = processor(transform)

    expected_stm = flip_matrix(
        horizontal=expected_x, vertical=expected_y, width=10, height=12
    )
    if batch_size is None:
        assert expected_stm.shape.ndims == 2
    else:
        assert expected_stm.shape.ndims == 3
    stm, expected_stm = tf.compat.v1.Session().run(
        [stm.spatial_transform_matrix, expected_stm], feed_dict=feed_dict
    )
    np.testing.assert_equal(stm, expected_stm)


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    processor = RandomFlip(horizontal_probability=0.5)
    processor_dict = processor.serialize()
    deserialized_processor = deserialize_tao_object(processor_dict)
    assert (
        processor._horizontal_probability
        == deserialized_processor._horizontal_probability
    )
