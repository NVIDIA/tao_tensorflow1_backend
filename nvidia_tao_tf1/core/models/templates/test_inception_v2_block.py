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

import keras
from keras.models import Model

from nvidia_tao_tf1.core.models.templates.inception_v2_block import InceptionV2Block
from nvidia_tao_tf1.core.models.templates.utils import count_layers_by_class_name
import pytest


topologies = [
    # Test with/without bias
    ([32, [32, 32], [32, 32, 32], 32], True, True, "channels_first", 29152),
    ([32, [32, 32], [32, 32, 32], 32], True, False, "channels_first", 28928),
    # Test with/without BN
    ([32, [32, 32], [32, 32, 32], 32], True, None, "channels_first", 28928),
    ([32, [32, 32], [32, 32, 32], 32], False, None, "channels_first", 28256),
    # Test channel_last
    ([32, [32, 32], [32, 32, 32], 32], True, False, "channels_last", 28928),
    # Test network structure
    ([1, [2, 3], [4, 5, 6], 118], True, True, "channels_first", 1574),
    ([1, [2, 3], [4, 5, 6], 118], True, False, "channels_last", 1435),
    ([1, [2, 3], [4, 5, 6], 118], False, False, "channels_first", 879),
    ([1, [2, 3], [4, 5, 6], 118], False, None, "channels_last", 1018),
]


@pytest.mark.parametrize(
    "subblocks,use_batch_norm,use_bias," "data_format,expected_params", topologies
)
def test_inception_v2_block(
    subblocks, use_batch_norm, use_bias, data_format, expected_params
):
    """Test Inception_v2_block for a variety of topologies and parameters."""
    w, h = 48, 16
    expected_num_channel = 128
    if data_format == "channels_last":
        shape = (w, h, 3)
    elif data_format == "channels_first":
        shape = (3, w, h)
    inputs = keras.layers.Input(shape=shape)
    outputs = InceptionV2Block(
        use_batch_norm=use_batch_norm,
        use_bias=use_bias,
        data_format=data_format,
        subblocks=subblocks,
    )(inputs)

    # Check output shape
    output_shape = outputs.get_shape()[1:4]
    if data_format == "channels_last":
        expected_spatial_shape = (w, h, expected_num_channel)
        assert tuple(s.value for s in output_shape) == expected_spatial_shape
    elif data_format == "channels_first":
        expected_spatial_shape = (expected_num_channel, w, h)
        assert tuple(s.value for s in output_shape) == expected_spatial_shape

    model = Model(inputs=inputs, outputs=outputs, name="inceptionv2")

    # Batchnorm check
    n_batchnorms = count_layers_by_class_name(model, ["BatchNormalization"])
    if use_batch_norm:
        assert n_batchnorms > 0
    else:
        assert n_batchnorms == 0

    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            # There should be no bias if use_bias is None, and batch norm is on.
            if use_bias is None:
                assert layer.get_config()["use_bias"] == (not use_batch_norm)
            else:
                assert layer.get_config()["use_bias"] == use_bias

    # Layer count check (just check the number of conv layers, not able to check
    # the number of branches.)
    n_layers_counted = count_layers_by_class_name(model, ["Conv2D"])

    expected_nlayers = 7  # each inception v2 block should have 7 convolutional layers.
    assert n_layers_counted == expected_nlayers

    # Total model parameters check.
    assert model.count_params() == expected_params
