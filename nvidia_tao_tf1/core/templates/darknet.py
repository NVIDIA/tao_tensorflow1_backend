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

"""DarkNet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import layers
from keras import models

from nvidia_tao_tf1.core.templates.utils import _leaky_conv
from nvidia_tao_tf1.core.templates.utils import arg_scope


def DarkNet(nlayers,
            input_tensor=None,
            input_shape=None,
            alpha=0.1,
            add_head=False,
            data_format='channels_first',
            kernel_regularizer=None,
            bias_regularizer=None,
            nclasses=1000,
            use_batch_norm=True,
            freeze_bn=False,
            freeze_blocks=None,
            use_bias=False,
            force_relu=False):
    """
    The DarkNet model architecture.

    Args:
        nlayers(int): 19 or 53.
        input_tensor(tensor): Input tensor.
        input_shape(tuple, None): Shape of the input tensor, can be None.
        alpha(float): The leaky rate for Leaky ReLU.
        add_head(bool): Whether or not to add the ImageNet head. If not, will add dense head.
        data_format(str): Data format, can be channels_first or channels_last.
        kernel_regularizer: Kernel regularizer applied to the model.
        bias_regularizer: Bias regularizer applied to the model.
        nclasses(int): Number of classes the output will be classified into.
        use_batch_norm(bool): Whether or not to use the BN layer.
        activation_type(str): Activation type, can be relu or relu6.
        freeze_bn(bool): Whether or not to freeze the BN layers.
        freeze_blocks(list): the list of blocks in the model to be frozen.
        use_bias(bool): Whether or not use bias for the conv layer
                        that is immediately before the BN layers.
        force_relu(bool): Whether to use ReLU instead of LeakyReLU

    Returns:
        The output tensor.

    """
    if freeze_blocks is None:
        freeze_blocks = []
    if input_shape is None:
        if data_format == 'channels_first':
            input_shape = (3, None, None)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    with arg_scope([_leaky_conv],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   alpha=alpha,
                   padding='same',
                   freeze_bn=freeze_bn,
                   use_bias=use_bias,
                   force_relu=force_relu):
        x = _leaky_conv(img_input, filters=32, kernel=3, strides=1,
                        name='conv1', trainable=not(0 in freeze_blocks))
        if nlayers == 53:
            x = _leaky_conv(x, filters=64, kernel=3, strides=2, name='conv2',
                            trainable=not(1 in freeze_blocks))
            y = _leaky_conv(x, filters=32, kernel=1, strides=1, name='b1_conv1_1',
                            trainable=not(1 in freeze_blocks))
            y = _leaky_conv(y, filters=64, kernel=3, strides=1, name='b1_conv1_2',
                            trainable=not(1 in freeze_blocks))
            x = layers.Add(name='b1_add1')([x, y])
            x = _leaky_conv(x, filters=128, kernel=3, strides=2, name='conv3',
                            trainable=not(2 in freeze_blocks))
            for i in range(2):
                y = _leaky_conv(x, filters=64, kernel=1, strides=1,
                                name='b2_conv{}_1'.format(i+1), trainable=not(2 in freeze_blocks))
                y = _leaky_conv(y, filters=128, kernel=3, strides=1,
                                name='b2_conv{}_2'.format(i+1), trainable=not(2 in freeze_blocks))
                x = layers.Add(name='b2_add{}'.format(i+1))([x, y])
            x = _leaky_conv(x, filters=256, kernel=3, strides=2, name='conv4',
                            trainable=not(3 in freeze_blocks))
            for i in range(8):
                y = _leaky_conv(x, filters=128, kernel=1, strides=1,
                                name='b3_conv{}_1'.format(i+1), trainable=not(3 in freeze_blocks))
                y = _leaky_conv(y, filters=256, kernel=3, strides=1,
                                name='b3_conv{}_2'.format(i+1), trainable=not(3 in freeze_blocks))
                x = layers.Add(name='b3_add{}'.format(i+1))([x, y])
            x = _leaky_conv(x, filters=512, kernel=3, strides=2, name='conv5',
                            trainable=not(4 in freeze_blocks))
            for i in range(8):
                y = _leaky_conv(x, filters=256, kernel=1, strides=1,
                                name='b4_conv{}_1'.format(i+1), trainable=not(4 in freeze_blocks))
                y = _leaky_conv(y, filters=512, kernel=3, strides=1,
                                name='b4_conv{}_2'.format(i+1), trainable=not(4 in freeze_blocks))
                x = layers.Add(name='b4_add{}'.format(i+1))([x, y])
            x = _leaky_conv(x, filters=1024, kernel=3, strides=2, name='conv6',
                            trainable=not(5 in freeze_blocks))
            for i in range(4):
                y = _leaky_conv(x, filters=512, kernel=1, strides=1,
                                name='b5_conv{}_1'.format(i+1), trainable=not(5 in freeze_blocks))
                y = _leaky_conv(y, filters=1024, kernel=3, strides=1,
                                name='b5_conv{}_2'.format(i+1), trainable=not(5 in freeze_blocks))
                x = layers.Add(name='b5_add{}'.format(i+1))([x, y])
        elif nlayers == 19:
            x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                    padding='same', name='maxpool_1')(x)
            x = _leaky_conv(x, filters=64, kernel=3, strides=1, name='b1_conv1',
                            trainable=not(1 in freeze_blocks))
            x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                    padding='same', name='maxpool_2')(x)
            x = _leaky_conv(x, filters=128, kernel=3, strides=1,
                            name='b2_conv1', trainable=not(2 in freeze_blocks))
            x = _leaky_conv(x, filters=64, kernel=1, strides=1,
                            name='b2_conv2', trainable=not(2 in freeze_blocks))
            x = _leaky_conv(x, filters=128, kernel=3, strides=1,
                            name='b2_conv3', trainable=not(2 in freeze_blocks))
            x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                    padding='same', name='maxpool_3')(x)
            x = _leaky_conv(x, filters=256, kernel=3, strides=1,
                            name='b3_conv1', trainable=not(3 in freeze_blocks))
            x = _leaky_conv(x, filters=128, kernel=1, strides=1,
                            name='b3_conv2', trainable=not(3 in freeze_blocks))
            x = _leaky_conv(x, filters=256, kernel=3, strides=1,
                            name='b3_conv3', trainable=not(3 in freeze_blocks))
            x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                    padding='same', name='maxpool_4')(x)
            x = _leaky_conv(x, filters=512, kernel=3, strides=1,
                            name='b4_conv1', trainable=not(4 in freeze_blocks))
            x = _leaky_conv(x, filters=256, kernel=1, strides=1,
                            name='b4_conv2', trainable=not(4 in freeze_blocks))
            x = _leaky_conv(x, filters=512, kernel=3, strides=1,
                            name='b4_conv3', trainable=not(4 in freeze_blocks))
            x = _leaky_conv(x, filters=256, kernel=1, strides=1,
                            name='b4_conv4', trainable=not(4 in freeze_blocks))
            x = _leaky_conv(x, filters=512, kernel=3, strides=1,
                            name='b4_conv5', trainable=not(4 in freeze_blocks))
            x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                    padding='same', name='maxpool_5')(x)
            x = _leaky_conv(x, filters=1024, kernel=3, strides=1,
                            name='b5_conv1', trainable=not(5 in freeze_blocks))
            x = _leaky_conv(x, filters=512, kernel=1, strides=1,
                            name='b5_conv2', trainable=not(5 in freeze_blocks))
            x = _leaky_conv(x, filters=1024, kernel=3, strides=1,
                            name='b5_conv3', trainable=not(5 in freeze_blocks))
            x = _leaky_conv(x, filters=512, kernel=1, strides=1,
                            name='b5_conv4', trainable=not(5 in freeze_blocks))
            x = _leaky_conv(x, filters=1024, kernel=3, strides=1,
                            name='b5_conv5', trainable=not(5 in freeze_blocks))
        else:
            raise NotImplementedError('A DarkNet with nlayers=%d is not implemented.' % nlayers)

        # if add_head, make it a network with a stride of 32, otherwise, the stride is 16.
        if add_head:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name='avgpool')(x)
            x = layers.Dense(nclasses, activation='softmax', name='predictions',
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer)(x)

    # Create model.
    model_name = 'darknet%d' % nlayers
    if use_batch_norm:
        model_name += '_bn'
    if add_head:
        model_name += '_add_head'

    model = models.Model(img_input, x, name=model_name)

    return model
