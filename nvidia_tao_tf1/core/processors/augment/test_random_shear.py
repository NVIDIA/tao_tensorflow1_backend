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
"""Tests for RandomRotation processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from six.moves import mock
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import RandomShear
from nvidia_tao_tf1.core.processors.augment.spatial import shear_matrix
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "probability, message",
    [
        (-0.1, "RandomShear.probability (-0.1) is not within the range [0.0, 1.0]."),
        (1.1, "RandomShear.probability (1.1) is not within the range [0.0, 1.0]."),
    ],
)
def test_raises_on_invalid_probability(probability, message):
    with pytest.raises(ValueError) as exc:
        RandomShear(max_ratio_x=0.1, max_ratio_y=0.1, probability=probability)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "max_ratio_x, message", [(-1, "RandomShear.max_ratio_x (-1) is less than 0.")]
)
def test_raises_on_invalid_max_ratio_x(max_ratio_x, message):
    with pytest.raises(ValueError) as exc:
        RandomShear(max_ratio_x=max_ratio_x, max_ratio_y=0.0, probability=0.5)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "max_ratio_y, message", [(-1, "RandomShear.max_ratio_y (-1) is less than 0.")]
)
def test_raises_on_invalid_max_ratio_y(max_ratio_y, message):
    with pytest.raises(ValueError) as exc:
        RandomShear(max_ratio_x=0.0, max_ratio_y=max_ratio_y, probability=0.5)
    assert str(exc.value) == message


@mock.patch("nvidia_tao_tf1.core.processors.augment.random_translation.spatial.random_shear_matrix")
def test_random_shear_call(mocked_random_shear_matrix):
    """Test RandomShear processor call."""
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4),
        spatial_transform_matrix=tf.eye(3),
    )
    mocked_random_shear_matrix.return_value = tf.eye(3)
    processor = RandomShear(max_ratio_x=0.5, max_ratio_y=0.25, probability=1.0)
    processor(transform)
    mocked_random_shear_matrix.assert_called_with(
        max_ratio_x=0.5, max_ratio_y=0.25, height=12, width=10, batch_size=None
    )


def test_random_shear_call_with_invalid_input():
    """Test RandomShear processor call error handling on invalid input types."""
    # Calling RandomShear with str should throw a TypeError.
    with pytest.raises(TypeError):
        RandomShear(0, 0, 0)("Transform")


@mock.patch("nvidia_tao_tf1.core.processors.augment.spatial.tf.random.uniform")
@pytest.mark.parametrize(
    "batch_size", [None, 5, tf.compat.v1.placeholder(dtype=tf.int32)]
)
def test_random_shear(mocked_random_uniform, batch_size):
    """Test RandomShear processor."""
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
    rnd_x = 0.5
    rnd_y = 0.25
    expected_x = rnd_x
    expected_y = rnd_y
    if batch_size is not None:
        # Generate a sequence of probabilities [0., 1., 0., 1., ...] so that every second
        # sample gets randomly tranformed.
        float_batch_size = tf.cast(batch_size, tf.float32)
        rnd_prob = tf.math.floormod(
            tf.linspace(0.0, float_batch_size - 1.0, batch_size), 2.0
        )
        # Generate a linearly interpolated sequences of x and y translation values.
        rnd_x = tf.linspace(-0.5, 0.5, batch_size)
        rnd_y = tf.linspace(1.0, -1.0, batch_size)
        # Zero out the samples that don't get transformed.
        mask = 1.0 - rnd_prob
        expected_x = rnd_x * mask
        expected_y = rnd_y * mask

    # The first tf.random_uniform call is for deciding whether shear is applied,
    # the second is for x shear ratio, the third is for y shear ratio.
    mocked_random_uniform.side_effect = [rnd_prob, rnd_x, rnd_y]

    processor = RandomShear(max_ratio_x=1.0, max_ratio_y=1.0, probability=0.5)
    stm = processor(transform)

    expected_stm = shear_matrix(expected_x, expected_y, 10, 12)
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
    processor = RandomShear(max_ratio_x=1.0, max_ratio_y=1.0, probability=0.5)
    processor_dict = processor.serialize()
    deserialized_processor = deserialize_tao_object(processor_dict)
    assert processor._max_ratio_x == deserialized_processor._max_ratio_x
    assert processor._max_ratio_y == deserialized_processor._max_ratio_y
    assert processor._probability == deserialized_processor._probability
