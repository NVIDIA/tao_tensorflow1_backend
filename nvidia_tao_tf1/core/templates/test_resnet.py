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

import pytest

from nvidia_tao_tf1.core.models.import_keras import keras as keras_fn
from nvidia_tao_tf1.core.models.templates.utils import count_layers_by_class_name
from nvidia_tao_tf1.core.templates.resnet import ResNet

keras = keras_fn()

topologies = [
    # Test the different nlayers
    (18, True, 'channels_first', True, False, True),
    # (34, True, 'channels_first', True), # Commented because takes ages
    # (50, True, 'channels_first', True), # Commented because takes ages
    # (101, True, 'channels_first', True), # Commented because takes ages
    # (152, True, 'channels_first', True), # Commented because takes ages

    # Without BN
    (18, False, 'channels_first', True, True, False),
    # Without head
    (18, False, 'channels_first', False, False, True),

    # channels_last:
    # With BN, with head
    (18, True, 'channels_last', True, False, False),
    # Without BN, with head
    (18, False, 'channels_last', True, True, True),
    # Without BN, without head
    (18, False, 'channels_last', False, False, True),
]


@pytest.mark.parametrize("nlayers, use_batch_norm, data_format, add_head,"
                         "all_projections, use_pooling", topologies)
def test_resnet(nlayers,
                use_batch_norm,
                data_format,
                add_head,
                all_projections,
                use_pooling,
                nclasses=None):
    """Test Resnets for a variety of topologies and parameters."""
    if data_format == 'channels_last':
        shape = (256, 256, 3)
    elif data_format == 'channels_first':
        shape = (3, 256, 256)

    inputs = keras.layers.Input(shape=shape)

    if add_head:
        nclasses = 10

    model = ResNet(
        nlayers,
        inputs,
        use_batch_norm=use_batch_norm,
        data_format=data_format,
        add_head=add_head,
        all_projections=all_projections,
        use_pooling=use_pooling,
        nclasses=nclasses)

    # Batchnorm check
    n_batchnorms = count_layers_by_class_name(model, ["BatchNormalization"])
    if use_batch_norm:
        assert n_batchnorms > 0
    else:
        assert n_batchnorms == 0

    # Layer count check
    n_layers_counted = count_layers_by_class_name(model, ["Conv2D", "Dense"])

    expected_nlayers = nlayers if add_head else nlayers - 1
    bridge_factor = 8 if all_projections else 4

    # Account for bridging dimensional gap with the extra 4 Conv2D blocks.
    expected_nlayers += bridge_factor
    assert n_layers_counted == expected_nlayers

    # Check model output shape
    output_shape = model.outputs[0].get_shape()

    # Set expected shape depending on whther or not pruning is set.
    expected_shape = (2, 2) if use_pooling else (16, 16)
    if add_head:
        assert output_shape[1:] == (nclasses)
    else:
        if data_format == 'channels_last':
            assert output_shape[1:3] == expected_shape
        elif data_format == 'channels_first':
            assert output_shape[2:4] == expected_shape
