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
import pytest
from nvidia_tao_tf1.core.models.templates.utils import count_layers_by_class_name
from nvidia_tao_tf1.cv.bpnet.models.templates.helnet import HelNet

mtype_stride_mapping = {
    "default": 16,
    "s8_3rdblock": 8,
    "s8_3rdblock_wdilation": 8,
    "s8_1stlayer": 8,
    "s8_1stlayer_wdilation": 8
}

topologies = [
    # Test the different nlayers
    (6, False, True, "channels_first", "default"),
    (10, False, True, "channels_first", "default"),
    (12, False, True, "channels_first", "default"),
    (18, False, True, "channels_first", "default"),
    # Without BN
    (18, False, True, "channels_first", "default"),
    # With pooling
    (18, False, True, "channels_first", "default"),
    # channels_last:
    # With BN, with pooling
    (18, True, True, "channels_last", "default"),
    # No BN, no pooling
    (18, False, False, "channels_last", "default"),
    # test cases for model types
    (18, False, False, "channels_last", "s8_3rdblock"),
    (18, True, False, "channels_first", "s8_3rdblock_wdilation"),
    (18, True, True, "channels_first", "s8_1stlayer"),
    (18, False, True, "channels_last", "s8_1stlayer_wdilation"),
]


@pytest.mark.parametrize("nlayers,pooling,use_batch_norm,data_format, mtype",
                         topologies)
def test_helnet(nlayers, pooling, use_batch_norm, data_format, mtype):
    """Test headless Helnets for a variety of topologies and parameters."""
    w, h = 960, 480
    expected_stride = mtype_stride_mapping[mtype]

    if data_format == "channels_last":
        shape = (w, h, 3)
    elif data_format == "channels_first":
        shape = (3, w, h)

    inputs = keras.layers.Input(shape=shape)

    model = HelNet(
        nlayers,
        inputs,
        mtype=mtype,
        pooling=pooling,
        use_batch_norm=use_batch_norm,
        data_format=data_format,
    )

    # Batchnorm check
    n_batchnorms = count_layers_by_class_name(model, ["BatchNormalization"])
    if use_batch_norm:
        assert n_batchnorms > 0
    else:
        assert n_batchnorms == 0

    # There should be no bias if batch norm is on (~5% gain in training speed with DGX-1 Volta).
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            assert layer.get_config()["use_bias"] == (not use_batch_norm)

    # Layer count check
    n_layers_counted = count_layers_by_class_name(model, ["Conv2D", "Dense"])

    expected_nlayers = nlayers - 1  # subtract one because it's headless
    assert n_layers_counted == expected_nlayers

    # Check model output shape
    output_shape = model.outputs[0].get_shape()
    expected_spatial_shape = (int(w / expected_stride),
                              int(h / expected_stride))
    if data_format == "channels_last":
        assert output_shape[1:3] == expected_spatial_shape
    elif data_format == "channels_first":
        assert output_shape[2:4] == expected_spatial_shape


def test_helnet_variable_feature_maps():
    """Test headless Helnets for a change in number of feature maps."""
    shape = (3, 960, 480)
    block_widths = (64, 128, 64, 128)

    inputs = keras.layers.Input(shape=shape)

    # create a HelNet34 network and change width of blocks 3 and 4
    model = HelNet(34,
                   inputs,
                   pooling=False,
                   use_batch_norm=True,
                   block_widths=block_widths)

    for layer in model.layers:
        config = layer.get_config()
        # check if layer is a convolutional layer
        if type(layer) == keras.layers.convolutional.Conv2D:
            layer_name = config["name"]
            # check if layer is a block layer
            if layer_name.split("_")[0] == "block":
                block_num = int(layer_name.split("_")[1][0])
                assert config["filters"] == block_widths[block_num - 1]
