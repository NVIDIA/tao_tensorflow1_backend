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
"""Tests for Scale processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from six.moves import mock
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import Scale
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "height, width, message",
    [
        (0, 1, "Scale.height (0) is not positive."),
        (1, 0, "Scale.width (0) is not positive."),
    ],
)
def test_invalid_scale_parameters(height, width, message):
    """Test Scale processor constructor error handling on invalid height and width."""
    with pytest.raises(ValueError) as exc:
        Scale(height=height, width=width)
    assert str(exc.value) == message


@mock.patch("nvidia_tao_tf1.core.processors.augment.scale.spatial.zoom_matrix")
def test_scale_call(mocked_zoom_matrix):
    """Test Scale processor call."""
    transform = Transform(
        canvas_shape=Canvas2D(height=12, width=10),
        color_transform_matrix=tf.eye(4),
        spatial_transform_matrix=tf.eye(3),
    )
    mocked_zoom_matrix.return_value = tf.eye(3)
    processor = Scale(height=6, width=5)
    processor(transform)
    mocked_zoom_matrix.assert_called_with(ratio=(2, 2))


@pytest.mark.parametrize(
    "batch_size", [None, 3, tf.compat.v1.placeholder(dtype=tf.int32)]
)
def test_scale(batch_size):
    """Test Scale processor."""
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

    processor = Scale(height=6, width=5)
    transformed = processor(transform)
    res = tf.compat.v1.Session().run(
        transformed.spatial_transform_matrix, feed_dict=feed_dict
    )
    expected = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    if batch_size is not None:
        expected = np.tile(expected, [expected_batch_size, 1, 1])
    np.testing.assert_array_equal(res, expected)


def test_scale_call_with_invalid_input():
    """Test Scale processor call error handling on invalid input types."""
    # Calling Scale with str should throw a TypeError.
    with pytest.raises(TypeError):
        Scale(1, 1)("Transform")


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    processor = Scale(height=6, width=5)
    processor_dict = processor.serialize()
    deserialized_processor = deserialize_tao_object(processor_dict)
    assert processor._height == deserialized_processor._height
    assert processor._width == deserialized_processor._width
