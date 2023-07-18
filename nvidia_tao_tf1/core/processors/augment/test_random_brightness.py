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
"""Tests for RandomBrightness processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from six.moves import mock
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import RandomBrightness
from nvidia_tao_tf1.core.processors.augment.color import brightness_offset_matrix
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "batch_size", [None, 5, tf.compat.v1.placeholder(dtype=tf.int32)]
)
@pytest.mark.parametrize("scale_max", [90, 180])
@pytest.mark.parametrize(
    "brightness_offset", [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, 2.0, -1.0)]
)
@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.random_contrast.color.tf.random.truncated_normal"
)
def test_random_brightness_call(
    mocked_truncated_normal, batch_size, brightness_offset, scale_max
):
    """Test RandomBrightness processor call"""
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

    # Fix brightness offset for testing.
    if batch_size is not None:
        brightness_offset = tf.tile(
            tf.constant(brightness_offset, shape=[1, 3]), [batch_size, 1]
        )
        mocked_truncated_normal.return_value = brightness_offset
    else:
        mocked_truncated_normal.return_value = tf.constant(
            brightness_offset, dtype=tf.float32
        )

    processor = RandomBrightness(scale_max=scale_max, uniform_across_channels=False)
    final_transform = processor(transform)

    expected_ctm = brightness_offset_matrix(offset=brightness_offset)
    if batch_size is None:
        assert expected_ctm.shape.ndims == 2
    else:
        assert expected_ctm.shape.ndims == 3
    ctm, expected_ctm = tf.compat.v1.Session().run(
        [final_transform.color_transform_matrix, expected_ctm], feed_dict=feed_dict
    )

    np.testing.assert_equal(ctm, expected_ctm)

    if batch_size is None:
        mocked_truncated_normal.assert_called_with([3], mean=0.0, stddev=scale_max / 2)
    else:
        mocked_truncated_normal.assert_called_once()
        call_batch_shape = mocked_truncated_normal.call_args[0][0]
        assert len(call_batch_shape) == 2
        assert (
            tf.compat.v1.Session().run(call_batch_shape[0], feed_dict=feed_dict)
            == expected_batch_size
        )
        assert call_batch_shape[1] == 3
        assert mocked_truncated_normal.call_args[1] == {
            "mean": 0.0,
            "stddev": scale_max / 2,
        }


def test_random_brightness_call_with_invalid_input():
    """Test RandomBrightness processor call error handling on invalid input types."""
    # Calling RandomBrightness with str should throw a TypeError.
    with pytest.raises(TypeError):
        RandomBrightness(0, True)("Transform")


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    processor = RandomBrightness(scale_max=90, uniform_across_channels=False)
    processor_dict = processor.serialize()
    deserialized_processor = deserialize_tao_object(processor_dict)
    assert processor._scale_max == deserialized_processor._scale_max
    assert (
        processor._uniform_across_channels
        == deserialized_processor._uniform_across_channels
    )
