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
from nvidia_tao_tf1.core.templates.vgg import VggNet

topologies = [
    # Test the different nlayers
    (16, True, 'channels_first', True, True),
    # Without BN
    (19, False, 'channels_first', True, False),
    # Without head
    (16, False, 'channels_first', False, True),
    # channels_last:
    # With BN, with head
    (16, True, 'channels_last', True, False),
    # Without BN, with head
    (19, False, 'channels_last', True, True),
    # Without BN, without head
    (19, False, 'channels_last', False, True),
]


@pytest.mark.parametrize("nlayers, use_batch_norm, data_format, add_head,"
                         "use_pooling", topologies)
def test_vggnet(nlayers, use_batch_norm, data_format, add_head, use_pooling,
                nclasses=None):
    """Test Resnets for a variety of topologies and parameters."""
    # Set channel format.
    if data_format == 'channels_last':
        shape = (256, 256, 3)
    elif data_format == 'channels_first':
        shape = (3, 256, 256)

    # Define supported counts.
    supported_counts = [16, 19]
    inputs = keras.layers.Input(shape=shape)

    # Add 10 class dense head if needed.
    if add_head:
        nclasses = 10

    # Instantiate model.
    if nlayers not in supported_counts:
        with pytest.raises(NotImplementedError):
            model = VggNet(nlayers, inputs,
                           use_batch_norm=use_batch_norm,
                           data_format=data_format,
                           add_head=add_head,
                           use_pooling=use_pooling,
                           activation_type='relu',
                           nclasses=nclasses)
    else:
        model = VggNet(nlayers, inputs,
                       use_batch_norm=use_batch_norm,
                       data_format=data_format,
                       add_head=add_head,
                       use_pooling=use_pooling,
                       activation_type='relu',
                       nclasses=nclasses)

    # Batchnorm check.
    n_batchnorms = count_layers_by_class_name(model, ["BatchNormalization"])
    if use_batch_norm:
        assert n_batchnorms > 0
    else:
        assert n_batchnorms == 0

    # Layer count check.
    n_conv_layers_counted = count_layers_by_class_name(model, ["Conv2D"])
    n_dense_layers_counted = count_layers_by_class_name(model, ["Dense"])

    expected_conv_layers = nlayers-3

    expected_dense_layers = 0
    if add_head:
        expected_dense_layers = 3
    assert n_dense_layers_counted == expected_dense_layers
    assert n_conv_layers_counted == expected_conv_layers

    # Check model output shape.
    output_shape = model.outputs[0].get_shape()

    # Set expected shape depending on whether or not pruning is set.
    expected_shape = (16, 16)
    if add_head:
        assert output_shape[1:] == (nclasses)
    else:
        if data_format == 'channels_last':
            assert output_shape[1:3] == expected_shape
        elif data_format == 'channels_first':
            assert output_shape[2:4] == expected_shape
