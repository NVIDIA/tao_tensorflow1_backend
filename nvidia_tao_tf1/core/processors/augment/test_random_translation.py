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
"""Tests for RandomTranslation processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from six.moves import mock
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import RandomTranslation
from nvidia_tao_tf1.core.processors.augment.spatial import translation_matrix
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "probability, message",
    [
        (2, "RandomTranslation.probability (2) is not within the range [0.0, 1.0]."),
        (-1, "RandomTranslation.probability (-1) is not within the range [0.0, 1.0]."),
    ],
)
def test_invalid_translation_probability(probability, message):
    """Test RandomTranslation processor constructor error handling on invalid probability."""
    with pytest.raises(ValueError) as exc:
        RandomTranslation(0, 0, probability=probability)
    assert str(exc.value) == message


@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.random_translation.spatial.random_translation_matrix"
)
def test_random_translation_call(mocked_random_translation_matrix):
    """Test RandomTranslation processor call."""
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4),
        spatial_transform_matrix=tf.eye(3),
    )
    mocked_random_translation_matrix.return_value = tf.eye(3)
    processor = RandomTranslation(max_x=90, max_y=45, probability=1.0)
    processor(transform)
    mocked_random_translation_matrix.assert_called_with(
        max_x=90, max_y=45, batch_size=None
    )


def test_random_translation_call_with_invalid_input():
    """Test RandomTranslation processor call error handling on invalid input types."""
    # Calling RandomTranslation with str should throw a TypeError.
    with pytest.raises(TypeError):
        RandomTranslation(0, 0, 0)("Transform")


@mock.patch("nvidia_tao_tf1.core.processors.augment.spatial.tf.random.uniform")
@pytest.mark.parametrize(
    "batch_size", [None, 5, tf.compat.v1.placeholder(dtype=tf.int32)]
)
def test_random_translation(mocked_random_uniform, batch_size):
    """Test RandomTranslation processor."""
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
    rnd_x = 15.0
    rnd_y = 12.0
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
        rnd_x = tf.linspace(1.0, 15.0, batch_size)
        rnd_y = tf.linspace(-15.0, 20.0, batch_size)
        # Zero out the samples that don't get transformed.
        mask = 1.0 - rnd_prob
        expected_x = rnd_x * mask
        expected_y = rnd_y * mask

    # The first tf.random_uniform call is for deciding whether translation is applied,
    # the second is for x translation, the third is for y translation.
    mocked_random_uniform.side_effect = [rnd_prob, rnd_x, rnd_y]

    processor = RandomTranslation(max_x=30, max_y=20, probability=0.5)
    stm = processor(transform)

    expected_stm = translation_matrix(x=expected_x, y=expected_y)
    stm, expected_stm = tf.compat.v1.Session().run(
        [stm.spatial_transform_matrix, expected_stm], feed_dict=feed_dict
    )
    np.testing.assert_equal(stm, expected_stm)


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    processor = RandomTranslation(max_x=30, max_y=20, probability=0.5)
    processor_dict = processor.serialize()
    deserialized_processor = deserialize_tao_object(processor_dict)
    assert processor._max_x == deserialized_processor._max_x
    assert processor._max_y == deserialized_processor._max_y
    assert processor._probability == deserialized_processor._probability
