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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.types import Canvas2D
from nvidia_tao_tf1.core.types import data_format
from nvidia_tao_tf1.core.types import DataFormat
from nvidia_tao_tf1.core.types import Example
from nvidia_tao_tf1.core.types import set_data_format
from nvidia_tao_tf1.core.types import Transform


def test_Canvas2D():
    """Test Canvas2D namedtuple."""
    fields = ("height", "width")
    assert getattr(Canvas2D, "_fields") == fields


def test_Example():
    """Test Example namedtuple."""
    fields = ("instances", "labels")
    assert getattr(Example, "_fields") == fields


def test_Transform():
    """Test Transform namedtuple."""
    fields = ("canvas_shape", "color_transform_matrix", "spatial_transform_matrix")
    assert getattr(Transform, "_fields") == fields


def test_default_data_format():
    """Test modulus default data format."""
    assert data_format() == "channels_first"


@pytest.mark.repeat(3)
@pytest.mark.parametrize(
    "data_fmt", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
def test_set_data_format(data_fmt):
    """Test for set data format."""
    set_data_format(data_fmt)
    assert data_format() == data_fmt


@pytest.mark.parametrize(
    "from_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize(
    "to_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("input_shape", [(1, 2, 3, 4), (1, 2, 3), (1, 2)])
def test_data_format_convert(from_format, to_format, input_shape):
    """Test for convert tensor format"""
    t = tf.constant(np.ones(input_shape))
    sess = tf.compat.v1.Session()
    input_dims = len(input_shape)
    if input_dims not in [3, 4]:
        with pytest.raises(NotImplementedError):
            sess.run(DataFormat.convert(t, from_format, to_format))
    else:
        output_np = sess.run(DataFormat.convert(t, from_format, to_format))
        if from_format == to_format:
            expected_shape = input_shape
            assert output_np.shape == expected_shape
        elif to_format == DataFormat.CHANNELS_LAST:
            expected_order = [0, 2, 3, 1] if input_dims == 4 else [1, 2, 0]
            expected_shape = [input_shape[i] for i in expected_order]
            assert list(output_np.shape) == expected_shape
        elif to_format == DataFormat.CHANNELS_FIRST:
            expected_order = [0, 3, 1, 2] if input_dims == 4 else [2, 0, 1]
            expected_shape = [input_shape[i] for i in expected_order]
            assert list(output_np.shape) == expected_shape
