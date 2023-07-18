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
"""Tests for FpeNet BaseModel model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from nvidia_tao_tf1.core.models.templates.utils import count_layers_by_class_name
from nvidia_tao_tf1.cv.fpenet.models.rcnet import RecombinatorNet


def build_sample_images(name='inputs',
                        data_format='channels_first',
                        channels=1,
                        height=80,
                        width=80):
    """Construct FpeNet model for testing.

    Args:
        name (str): Name of the input tensor. Default value is 'inputs'
        data_format (str): Expected tensor format, either `channels_first` or `channels_last`.
            Default value is `channels_first`.
        channels, height, width (all int): Input image dimentions.
    """

    # Set sample inputs.
    if data_format == 'channels_first':
        shape = (channels, height, width)
    elif data_format == 'channels_last':
        shape = (height, width, channels)
    else:
        raise ValueError(
            'Provide either `channels_first` or `channels_last` for `data_format`.'
        )
    image_face = Input(shape=shape, name=name)
    return image_face


def test_fpenet_model_builder():
    """Test FpeNetBaseModel constructor."""
    image_face = build_sample_images(name='input_face')

    # Test: 'FpeNet_base'
    block_trunk = [(3, 64), (3, 64), (1, 64)]
    blocks_encoder = [[(3, 64)]] * 4
    blocks_decoder = [[(3, 64), (1, 64)]] * 4
    model = RecombinatorNet(inputs=image_face,
                            pooling=True,
                            use_batch_norm=False,
                            data_format='channels_first',
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activation_type='relu',
                            activation_kwargs=None,
                            blocks_encoder=blocks_encoder,
                            block_trunk=block_trunk,
                            blocks_decoder=blocks_decoder,
                            use_upsampling_layer=False)

    assert count_layers_by_class_name(model, ['InputLayer']) == 1
    assert count_layers_by_class_name(model, ['Conv2D']) == 15
    assert count_layers_by_class_name(model, ['Dense']) == 0
    assert count_layers_by_class_name(model, ['Reshape']) == 0
    assert count_layers_by_class_name(model, ['Dropout']) == 0
    assert count_layers_by_class_name(model, ['MaxPooling2D']) == 4
    assert count_layers_by_class_name(model, ['Concatenate']) == 4
    assert count_layers_by_class_name(model, ['Softargmax']) == 0
    assert model.count_params() == 566784

    # Test: channels_last
    data_format = 'channels_last'
    image_face = build_sample_images(name='input_face', data_format=data_format)

    block_trunk = [(3, 64), (3, 64), (1, 64)]
    blocks_encoder = [[(3, 64)]] * 4
    blocks_decoder = [[(3, 64), (1, 64)]] * 4
    model = RecombinatorNet(inputs=image_face,
                            pooling=True,
                            use_batch_norm=False,
                            data_format=data_format,
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activation_type='relu',
                            activation_kwargs=None,
                            blocks_encoder=blocks_encoder,
                            block_trunk=block_trunk,
                            blocks_decoder=blocks_decoder,
                            use_upsampling_layer=False)

    assert count_layers_by_class_name(model, ['InputLayer']) == 1
    assert count_layers_by_class_name(model, ['Conv2D']) == 15
    assert count_layers_by_class_name(model, ['Dense']) == 0
    assert count_layers_by_class_name(model, ['Reshape']) == 0
    assert count_layers_by_class_name(model, ['Dropout']) == 0
    assert count_layers_by_class_name(model, ['MaxPooling2D']) == 4
    assert count_layers_by_class_name(model, ['Concatenate']) == 4
    assert count_layers_by_class_name(model, ['Softargmax']) == 0
    assert model.count_params() == 566784
