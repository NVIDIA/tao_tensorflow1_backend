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
"""Tests for Crop processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import Crop
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "left, top, right, bottom, message",
    [
        (-1, 0, 1, 1, "Crop.left (-1) is not positive."),
        (0, -1, 1, 1, "Crop.top (-1) is not positive."),
        (0, 0, -1, 1, "Crop.right (-1) is not positive."),
        (0, 0, 1, -1, "Crop.bottom (-1) is not positive."),
        (2, 0, 1, 1, "Crop.right (1) should be greater than Crop.left (2)."),
        (0, 2, 1, 1, "Crop.bottom (1) should be greater than Crop.top (2)."),
    ],
)
def test_invalid_crop_parameters(left, top, right, bottom, message):
    """Test Scale processor constructor error handling on invalid arguments."""
    with pytest.raises(ValueError) as exc:
        Crop(left=left, top=top, right=right, bottom=bottom)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "batch_size", [None, 4, tf.compat.v1.placeholder(dtype=tf.int32)]
)
@pytest.mark.parametrize(
    "left, top, right, bottom", [(0, 0, 1, 1), (2, 2, 5, 5), (3, 3, 10, 10)]
)
def test_crop_call(left, top, right, bottom, batch_size):
    """Test Crop processor call"""
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

    processor = Crop(left=left, top=top, right=right, bottom=bottom)
    expected_stm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [left, top, 1.0]])
    expected_ctm = np.eye(4)
    if batch_size is not None:
        expected_stm = np.tile(expected_stm, [expected_batch_size, 1, 1])
        expected_ctm = np.tile(expected_ctm, [expected_batch_size, 1, 1])
    expected_shape = Canvas2D(width=right - left, height=bottom - top)
    final_transform = processor(transform)
    ctm, stm = tf.compat.v1.Session().run(
        [
            final_transform.color_transform_matrix,
            final_transform.spatial_transform_matrix,
        ],
        feed_dict=feed_dict,
    )
    np.testing.assert_equal(ctm, expected_ctm)
    np.testing.assert_equal(stm, expected_stm)
    assert final_transform.canvas_shape == expected_shape


def test_crop_call_with_invalid_input():
    """Test Crop processor call error handling on invalid input types."""
    # Calling Crop with str should throw a TypeError.
    with pytest.raises(TypeError):
        Crop(0, 0, 1, 1)("Transform")


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    processor = Crop(left=2, top=2, right=5, bottom=5)
    processor_dict = processor.serialize()
    deserialized_processor = deserialize_tao_object(processor_dict)
    assert processor._left == deserialized_processor._left
    assert processor._top == deserialized_processor._top
    assert processor._right == deserialized_processor._right
    assert processor._bottom == deserialized_processor._bottom
