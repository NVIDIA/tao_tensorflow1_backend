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

"""Unit tests DataFormat class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import (
    CHANNELS_FIRST,
    CHANNELS_LAST,
    DataFormat,
)


def test_creation_fails_with_invalid_data_format():
    with pytest.raises(ValueError) as e:
        DataFormat("upside_down")
    assert "Unrecognized data_format 'upside_down'." in str(e.value)


def test_equality():
    assert DataFormat("channels_first") == DataFormat("channels_first")
    assert DataFormat("channels_last") == DataFormat("channels_last")


def test_inequality():
    assert DataFormat("channels_first") != DataFormat("channels_last")
    assert DataFormat("channels_last") != DataFormat("channels_first")


def test_stringify():
    assert "channels_first" == str(DataFormat("channels_first"))


def test_tensor_axis_4d_channels_first():
    axis = DataFormat("channels_first").axis_4d
    assert axis.batch == 0
    assert axis.channel == 1
    assert axis.row == 2
    assert axis.column == 3


def test_tensor_axis_4d_channels_last():
    axis = DataFormat("channels_last").axis_4d
    assert axis.batch == 0
    assert axis.row == 1
    assert axis.column == 2
    assert axis.channel == 3


def test_tensor_axis_5d_channels_first():
    axis = DataFormat("channels_first").axis_5d
    assert axis.batch == 0
    assert axis.time == 1
    assert axis.channel == 2
    assert axis.row == 3
    assert axis.column == 4


def test_tensor_axis_5d_channels_last():
    axis = DataFormat("channels_last").axis_5d
    assert axis.batch == 0
    assert axis.time == 1
    assert axis.row == 2
    assert axis.column == 3
    assert axis.channel == 4


def test_convert_shape_4d():
    shape = [64, 3, 200, 200]

    identity_convert = CHANNELS_FIRST.convert_shape(shape, CHANNELS_FIRST)

    assert shape == identity_convert

    to_channels_last = CHANNELS_FIRST.convert_shape(shape, CHANNELS_LAST)

    assert to_channels_last == [64, 200, 200, 3]


def test_convert_shape_5d():
    shape = [64, 6, 3, 200, 200]

    identity_convert = CHANNELS_FIRST.convert_shape(shape, CHANNELS_FIRST)

    assert shape == identity_convert

    to_channels_last = CHANNELS_FIRST.convert_shape(shape, CHANNELS_LAST)

    assert to_channels_last == [64, 6, 200, 200, 3]


def test_convert_last_to_first():
    shape = (None, None, 300, 300, 64)

    identity_convert = CHANNELS_LAST.convert_shape(shape, CHANNELS_LAST)

    assert shape == identity_convert

    to_channels_first = CHANNELS_LAST.convert_shape(shape, CHANNELS_FIRST)

    assert to_channels_first == (None, None, 64, 300, 300)
