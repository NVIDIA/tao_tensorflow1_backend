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
"""Model template for backbone of YOLOv4-tiny."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Input
)
from keras.models import Model

from nvidia_tao_tf1.core.templates.utils import _mish_conv
from nvidia_tao_tf1.core.templates.utils import arg_scope, csp_tiny_block


def CSPDarkNetTiny(
    input_tensor=None,
    input_shape=None,
    add_head=False,
    data_format='channels_first',
    kernel_regularizer=None,
    bias_regularizer=None,
    nclasses=1000,
    use_batch_norm=True,
    freeze_bn=False,
    freeze_blocks=None,
    use_bias=False,
    force_relu=False,
    activation="leaky_relu"
):
    """
    The DarkNet-tiny model architecture in YOLOv4-tiny.

    Reference: https://arxiv.org/abs/2011.08036

    """
    if freeze_blocks is None:
        freeze_blocks = []
    if input_shape is None:
        if data_format == 'channels_first':
            input_shape = (3, None, None)
        else:
            input_shape = (None, None, 3)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    with arg_scope(
        [_mish_conv],
        use_batch_norm=use_batch_norm,
        data_format=data_format,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        padding='same',
        freeze_bn=freeze_bn,
        use_bias=use_bias,
        force_relu=force_relu,
        activation=activation
    ):
        x = _mish_conv(img_input, 32, kernel=(3, 3), strides=(2, 2), name="conv_0",
                       trainable=not(0 in freeze_blocks))
        x = _mish_conv(x, 64, kernel=(3, 3), strides=(2, 2), name="conv_1",
                       trainable=not(1 in freeze_blocks))
        x = csp_tiny_block(x, num_filters=64, name="conv_2",
                           trainable=not(2 in freeze_blocks),
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           data_format=data_format, freeze_bn=freeze_bn,
                           force_relu=force_relu, use_bias=use_bias,
                           activation=activation,
                           use_batch_norm=use_batch_norm)
        x = csp_tiny_block(x, num_filters=128, name="conv_3",
                           trainable=not(3 in freeze_blocks),
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           data_format=data_format, freeze_bn=freeze_bn,
                           force_relu=force_relu, use_bias=use_bias,
                           activation=activation,
                           use_batch_norm=use_batch_norm)
        x = csp_tiny_block(x, num_filters=256, name="conv_4",
                           trainable=not(4 in freeze_blocks),
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           data_format=data_format, freeze_bn=freeze_bn,
                           force_relu=force_relu, use_bias=use_bias,
                           activation=activation,
                           use_batch_norm=use_batch_norm)
        x = _mish_conv(x, 512, kernel=(3, 3), name="conv_5",
                       trainable=not(5 in freeze_blocks),
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       data_format=data_format, freeze_bn=freeze_bn,
                       force_relu=force_relu, use_bias=use_bias)
    if add_head:
        x = GlobalAveragePooling2D(data_format=data_format, name='avgpool')(x)
        x = Dense(nclasses, activation='softmax', name='predictions',
                  kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer)(x)
    # Create model.
    model_name = 'cspdarknet_tiny'
    if use_batch_norm:
        model_name += '_bn'
    if add_head:
        model_name += '_add_head'
    model = Model(img_input, x, name=model_name)
    return model
