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
from nvidia_tao_tf1.core.processors import RandomRotation
from nvidia_tao_tf1.core.processors.augment.spatial import rotation_matrix
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "probability, message",
    [
        (-0.1, "RandomRotation.probability (-0.1) is not within the range [0.0, 1.0]."),
        (1.1, "RandomRotation.probability (1.1) is not within the range [0.0, 1.0]."),
    ],
)
def test_raises_on_invalid_probability(probability, message):
    with pytest.raises(ValueError) as exc:
        RandomRotation(min_angle=7, max_angle=7, probability=probability)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "min_angle, message",
    [
        (-600, "RandomRotation.min_angle (-600) is smaller than -360.0 degrees."),
        (
            8,
            "RandomRotation.min_angle (8) is greater than RandomRotation.max_angle (7).",
        ),
    ],
)
def test_raises_on_invalid_min_angle(min_angle, message):
    with pytest.raises(ValueError) as exc:
        RandomRotation(min_angle=min_angle, max_angle=7, probability=0.5)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "max_angle, message",
    [(361, "RandomRotation.max_angle (361) is greater than 360.0 degrees.")],
)
def test_raises_on_invalid_max_angle(max_angle, message):
    with pytest.raises(ValueError) as exc:
        RandomRotation(min_angle=7, max_angle=max_angle, probability=0.5)
    assert str(exc.value) == message


@mock.patch("nvidia_tao_tf1.core.processors.augment.random_rotation.tf.random.uniform")
@mock.patch("nvidia_tao_tf1.core.processors.augment.random_rotation.spatial.rotation_matrix")
def test_delegates_random_angle_to_rotation_matrix(
    mocked_rotation_matrix, mocked_random_uniform
):
    """Test RandomRotation processor call."""
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4),
        spatial_transform_matrix=tf.eye(3),
    )
    mocked_rotation_matrix.return_value = tf.eye(3)
    seven = tf.constant(7.0, dtype=tf.float32)
    mocked_random_uniform.return_value = seven

    processor = RandomRotation(min_angle=40, max_angle=90, probability=1.0)
    processor(transform)
    mocked_rotation_matrix.assert_called_with(seven, height=12, width=10)


@mock.patch("nvidia_tao_tf1.core.processors.augment.random_rotation.tf.random.uniform")
@pytest.mark.parametrize(
    "batch_size", [None, 3, tf.compat.v1.placeholder(dtype=tf.int32)]
)
def test_random_rotation(mocked_random_uniform, batch_size):
    """Test RandomRotation processor."""
    batch_shape = [] if batch_size is None else [batch_size]
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4, batch_shape=batch_shape, dtype=tf.float32),
        spatial_transform_matrix=tf.eye(3, batch_shape=batch_shape, dtype=tf.float32),
    )

    feed_dict = {}
    if type(batch_size) == tf.Tensor:
        feed_dict = {batch_size: 7}

    rnd = tf.fill(dims=batch_shape, value=0.5)
    mocked_random_uniform.return_value = rnd

    processor = RandomRotation(min_angle=40, max_angle=90, probability=1.0)
    stm = processor(transform)

    expected_stm = rotation_matrix(rnd, 10, 12)
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
    processor = RandomRotation(min_angle=40, max_angle=90, probability=1.0)
    processor_dict = processor.serialize()
    deserialized_processor = deserialize_tao_object(processor_dict)
    assert processor._min_angle == deserialized_processor._min_angle
    assert processor._max_angle == deserialized_processor._max_angle
    assert processor._probability == deserialized_processor._probability
