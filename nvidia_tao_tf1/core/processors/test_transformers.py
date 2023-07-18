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
"""Processor for applying a Modulus Transform to input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.processors import ColorTransform, ColorTransformer
from nvidia_tao_tf1.core.processors import SpatialTransform, SpatialTransformer
from nvidia_tao_tf1.core.types import Canvas2D, DataFormat, Transform


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("max_clip", [10.0, 20.0])
@pytest.mark.parametrize("min_clip", [0.0, 3.0])
@pytest.mark.parametrize("batch_size", [5, 10])
def test_color_transformer_call(batch_size, min_clip, max_clip, data_format):
    """Test color transformer call function."""
    applicant = (
        tf.ones([batch_size, 3, 5, 5])
        if data_format == DataFormat.CHANNELS_FIRST
        else tf.ones([batch_size, 5, 5, 3])
    )

    ctm = tf.random.uniform((4, 4), minval=1, maxval=5)
    batched_ctms = tf.tile(tf.expand_dims(ctm, axis=0), [batch_size, 1, 1])
    transform = Transform(
        canvas_shape=Canvas2D(5, 5),
        color_transform_matrix=ctm,
        spatial_transform_matrix=tf.eye(3),
    )
    transformer = ColorTransformer(
        transform, min_clip=min_clip, max_clip=max_clip, data_format=data_format
    )
    output = transformer(applicant)
    expected_output = ColorTransform(
        data_format=data_format, min_clip=min_clip, max_clip=max_clip
    )(applicant, batched_ctms)
    with tf.compat.v1.Session() as sess:
        output, expected_output = sess.run([output, expected_output])
    np.testing.assert_allclose(output, expected_output)


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("method", ["bilinear", "bicubic"])
@pytest.mark.parametrize("background_value", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("batch_size", [5, 10])
def test_spatial_transformer_call(batch_size, background_value, method, data_format):
    """Test color transformer call function."""
    applicant = (
        tf.ones([batch_size, 3, 5, 5])
        if data_format == DataFormat.CHANNELS_FIRST
        else tf.ones([batch_size, 5, 5, 3])
    )

    stm = tf.random.uniform((3, 3), minval=1, maxval=5)
    batched_stms = tf.tile(tf.expand_dims(stm, axis=0), [batch_size, 1, 1])
    transform = Transform(
        canvas_shape=Canvas2D(5, 5),
        color_transform_matrix=tf.eye(4),
        spatial_transform_matrix=stm,
    )
    transformer = SpatialTransformer(
        transform,
        data_format=data_format,
        method=method,
        background_value=background_value,
    )
    output = transformer(applicant)
    expected_output = SpatialTransform(
        data_format=data_format, method=method, background_value=background_value
    )(applicant, batched_stms)
    expected_shape = (
        [batch_size, 3, 5, 5]
        if data_format == DataFormat.CHANNELS_FIRST
        else [batch_size, 5, 5, 3]
    )

    with tf.compat.v1.Session() as sess:
        assert output.get_shape().is_fully_defined()
        assert expected_shape == output.get_shape().as_list()
        output, expected_output = sess.run([output, expected_output])
    np.testing.assert_allclose(output, expected_output)
