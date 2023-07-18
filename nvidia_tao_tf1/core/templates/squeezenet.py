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

"""SqueezeNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend
from keras import layers
from keras import models

from nvidia_tao_tf1.core.templates.utils import arg_scope
from nvidia_tao_tf1.core.templates.utils import fire_module


def SqueezeNet(inputs=None,
               input_shape=None,
               dropout=1e-3,
               add_head=False,
               data_format='channels_first',
               kernel_regularizer=None,
               bias_regularizer=None,
               nclasses=1000,
               freeze_blocks=None,
               skip=False):
    """
    The squeeze net architecture.

    For details, see https://arxiv.org/pdf/1602.07360.pdf


    Args:
        inputs(tensor): Input tensor.
        input_shape(tuple, None): Shape of the input tensor, can be None.
        dropout(float): Dropout ratio.
        add_head(bool): Whether or not to add the ImageNet head. If not, will add dense head.
        data_format(str): Data format, can be channels_first or channels_last.
        kernel_regularizer: Kernel regularizer applied to the model.
        bias_regularizer: Bias regularizer applied to the model.
        nclasses(int): Number of classes the output will be classified into.
        freeze_blocks(list): the list of blocks to be frozen in the model.

    Returns:
        The output tensor.
    """
    if freeze_blocks is None:
        freeze_blocks = []
    if input_shape is None:
        if data_format == 'channels_first':
            input_shape = (3, 224, 224)
        else:
            input_shape = (224, 224, 3)

    if inputs is None:
        img_input = layers.Input(shape=input_shape, name="Input")
    else:
        if not backend.is_keras_tensor(inputs):
            img_input = layers.Input(tensor=inputs, shape=input_shape, name="Input")
        else:
            img_input = inputs

    x = layers.Conv2D(96,
                      kernel_size=(7, 7),
                      strides=(2, 2),
                      padding='same',
                      name='conv1',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      data_format=data_format,
                      trainable=not(0 in freeze_blocks))(img_input)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1',
                            data_format=data_format, padding='same')(x)
    with arg_scope([fire_module],
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   data_format=data_format):
        x = fire_module(x, 2, 16, 64, trainable=not(1 in freeze_blocks))
        if skip:
            x = layers.add([x, fire_module(x, 3, 16, 64,
                                           trainable=not(2 in freeze_blocks))])
        else:
            x = fire_module(x, 3, 16, 64, trainable=not(2 in freeze_blocks))
        x = fire_module(x, 4, 32, 128, trainable=not(3 in freeze_blocks))
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4',
                                data_format=data_format,
                                padding='same')(x)
        if skip:
            x = layers.add([x, fire_module(x, 5, 32, 128, trainable=not(4 in freeze_blocks))])
        else:
            x = fire_module(x, 5, 32, 128, trainable=not(4 in freeze_blocks))
        x = fire_module(x, 6, 48, 192, trainable=not(5 in freeze_blocks))
        if skip:
            x = layers.add([x, fire_module(x, 7, 48, 192, trainable=not(6 in freeze_blocks))])
        else:
            x = fire_module(x, 7, 48, 192, trainable=not(6 in freeze_blocks))

        x = fire_module(x, 8, 64, 256, trainable=not(7 in freeze_blocks))
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8',
                                data_format=data_format,
                                padding='same')(x)
        if skip:
            x = layers.add([x, fire_module(x, 9, 64, 256, trainable=not(8 in freeze_blocks))])
        else:
            x = fire_module(x, 9, 64, 256, trainable=not(8 in freeze_blocks))
        if add_head:
            x = layers.Dropout(rate=dropout, name='fire9_dropout')(x)
            x = layers.Conv2D(nclasses,
                              kernel_size=(1, 1),
                              padding='same',
                              name='conv10',
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              data_format=data_format)(x)
            x = layers.Activation('relu', name='conv10_relu')(x)
            x = layers.GlobalAveragePooling2D(data_format=data_format, name='pool10')(x)
            x = layers.Activation("softmax", name='output')(x)

    return models.Model(img_input, x)
