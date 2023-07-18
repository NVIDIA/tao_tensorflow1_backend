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
"""Tests for RandomHueSaturation processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from six.moves import mock
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import RandomHueSaturation
from nvidia_tao_tf1.core.processors.augment.color import hue_saturation_matrix
from nvidia_tao_tf1.core.types import Canvas2D, Transform


@pytest.mark.parametrize(
    "hue, saturation, message",
    [
        (
            -1,
            0,
            "RandomHueSaturation.hue_rotation_max (-1) "
            "is not within the range [0.0, 360.0].",
        ),
        (
            361,
            0,
            "RandomHueSaturation.hue_rotation_max (361) "
            "is not within the range [0.0, 360.0].",
        ),
        (
            0,
            -1,
            "RandomHueSaturation.saturation_shift_max (-1) "
            "is not within the range [0.0, 1.0].",
        ),
        (
            0,
            2,
            "RandomHueSaturation.saturation_shift_max (2) "
            "is not within the range [0.0, 1.0].",
        ),
    ],
)
def test_invalid_hue_saturation_values(hue, saturation, message):
    """Test RandomHueSaturation constructor error handling for invalid hue and saturation values."""
    with pytest.raises(ValueError) as exc:
        RandomHueSaturation(hue, saturation)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "batch_size", [None, 3, tf.compat.v1.placeholder(dtype=tf.int32)]
)
@pytest.mark.parametrize("hue", [0, 10, 20, 180, 360])
@pytest.mark.parametrize("saturation", [0.0, 0.2, 0.5, 1.0])
@mock.patch(
    "nvidia_tao_tf1.core.processors.augment.random_contrast.color.tf.random.truncated_normal"
)
@mock.patch("nvidia_tao_tf1.core.processors.augment.random_contrast.color.tf.random.uniform")
def test_random_hue_saturation_call(
    mocked_random_uniform, mocked_truncated_normal, batch_size, hue, saturation
):
    """Test RandomHueSaturation processor call."""
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

    # Fix all random function calls to return deterministic values for testing.
    if batch_size is not None:
        batched_hue = tf.linspace(float(hue), float(hue) + 45.0, batch_size)
        batched_saturation = tf.linspace(saturation, saturation + 0.2, batch_size)
    else:
        batched_hue = tf.constant(hue, dtype=tf.float32)
        batched_saturation = tf.constant(saturation, dtype=tf.float32)

    mocked_truncated_normal.return_value = batched_hue
    mocked_random_uniform.return_value = batched_saturation

    processor = RandomHueSaturation(
        hue_rotation_max=hue, saturation_shift_max=saturation
    )
    final_transform = processor(transform)

    # Add mean saturation.
    final_hue = batched_hue
    final_saturation = 1.0 + batched_saturation
    expected_ctm = hue_saturation_matrix(hue=final_hue, saturation=final_saturation)
    if batch_size is None:
        assert expected_ctm.shape.ndims == 2
    else:
        assert expected_ctm.shape.ndims == 3
    ctm, expected_ctm = tf.compat.v1.Session().run(
        [final_transform.color_transform_matrix, expected_ctm], feed_dict=feed_dict
    )

    np.testing.assert_equal(ctm, expected_ctm)

    if batch_size is None:
        mocked_truncated_normal.assert_called_with([], mean=0.0, stddev=hue / 2.0)
        mocked_random_uniform.assert_called_with(
            [], minval=-saturation, maxval=saturation
        )
    else:
        mocked_truncated_normal.assert_called_once()
        call_batch_shape = mocked_truncated_normal.call_args[0][0]
        assert len(call_batch_shape) == 1
        assert (
            tf.compat.v1.Session().run(call_batch_shape[0], feed_dict=feed_dict)
            == expected_batch_size
        )
        assert mocked_truncated_normal.call_args[1] == {
            "mean": 0.0,
            "stddev": hue / 2.0,
        }

        mocked_random_uniform.assert_called_once()
        call_batch_shape = mocked_random_uniform.call_args[0][0]
        assert len(call_batch_shape) == 1
        assert (
            tf.compat.v1.Session().run(call_batch_shape[0], feed_dict=feed_dict)
            == expected_batch_size
        )
        assert mocked_random_uniform.call_args[1] == {
            "minval": -saturation,
            "maxval": saturation,
        }


def test_random_hue_saturation_call_with_invalid_input():
    """Test RandomHueSaturation processor call error handling on invalid input types."""
    # Calling RandomHueSaturation with str should throw a TypeError.
    with pytest.raises(TypeError):
        RandomHueSaturation(0, 0)("Transform")


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    processor = RandomHueSaturation(hue_rotation_max=10, saturation_shift_max=0.2)
    processor_dict = processor.serialize()
    deserialized_processor = deserialize_tao_object(processor_dict)
    assert processor._hue_rotation_max == deserialized_processor._hue_rotation_max
    assert (
        processor._saturation_shift_max == deserialized_processor._saturation_shift_max
    )
