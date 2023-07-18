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

"""MobileNet V1 and V2 models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend
from keras import layers
from keras import models

from nvidia_tao_tf1.core.templates.utils import _conv_block, _depthwise_conv_block, \
                                _inverted_res_block, _make_divisible
from nvidia_tao_tf1.core.templates.utils import arg_scope


def MobileNet(inputs,
              input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              stride=32,
              add_head=True,
              data_format='channels_first',
              kernel_regularizer=None,
              bias_regularizer=None,
              nclasses=1000,
              use_batch_norm=True,
              activation_type='relu',
              freeze_bn=False,
              freeze_blocks=None,
              use_bias=False):
    """
    The MobileNet model architecture.

    Args:
        inputs(tensor): Input tensor.
        input_shape(tuple, None): Shape of the input tensor, can be None.
        alpha(float): The alpha parameter, defaults to 1.0.
        depth_multiplier(int): Depth multiplier for Depthwise Conv, defaults to 1.
        dropout(float): Dropout ratio.
        stride(int): The total stride of this model.
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

    Returns:
        The output tensor.

    """
    # Determine proper input shape and default size.
    assert stride in [16, 32], (
        "Only stride 16 and 32 are supported, got {}".format(stride)
    )
    old_data_format = backend.image_data_format()
    backend.set_image_data_format(data_format)
    if freeze_blocks is None:
        freeze_blocks = []
    if input_shape is None:
        if backend.image_data_format() == 'channels_first':
            input_shape = (3, 224, 224)
        else:
            input_shape = (224, 224, 3)

    if inputs is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(inputs):
            img_input = layers.Input(tensor=inputs, shape=input_shape)
        else:
            img_input = inputs

    with arg_scope([_conv_block, _depthwise_conv_block],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   activation_type=activation_type,
                   freeze_bn=freeze_bn,
                   use_bias=use_bias):
        x = _conv_block(img_input, 32, alpha, strides=(2, 2),
                        trainable=not(0 in freeze_blocks))
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1,
                                  trainable=not(1 in freeze_blocks))

        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=2,
                                  trainable=not(2 in freeze_blocks))
        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,
                                  trainable=not(3 in freeze_blocks))

        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=4,
                                  trainable=not(4 in freeze_blocks))
        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,
                                  trainable=not(5 in freeze_blocks))

        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=6,
                                  trainable=not(6 in freeze_blocks))
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7,
                                  trainable=not(7 in freeze_blocks))
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8,
                                  trainable=not(8 in freeze_blocks))
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9,
                                  trainable=not(9 in freeze_blocks))
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10,
                                  trainable=not(10 in freeze_blocks))
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11,
                                  trainable=not(11 in freeze_blocks))

        # make it a network with a stride of 32, otherwise, the stride is 16.
        if stride == 32:
            x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                                      strides=(2, 2), block_id=12,
                                      trainable=not(12 in freeze_blocks))
            x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13,
                                      trainable=not(13 in freeze_blocks))
        if add_head:
            x = layers.AveragePooling2D(pool_size=(7, 7),
                                        data_format=data_format, padding='valid')(x)
            x = layers.Flatten(name='flatten_1')(x)

            x = layers.Dropout(dropout, name='dropout')(x)
            x = layers.Dense(nclasses, activation='softmax', name='predictions',
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer)(x)

    # Create model.
    model_name = 'mobilenet'
    if use_batch_norm:
        model_name += '_bn'
    if add_head:
        model_name += '_add_head'

    model = models.Model(img_input, x, name=model_name)

    backend.set_image_data_format(old_data_format)
    return model


def MobileNetV2(inputs,
                input_shape=None,
                alpha=1.0,
                depth_multiplier=1,
                stride=32,
                add_head=True,
                data_format='channels_first',
                kernel_regularizer=None,
                bias_regularizer=None,
                use_batch_norm=True,
                activation_type='relu',
                all_projections=False,
                nclasses=1000,
                freeze_bn=False,
                freeze_blocks=None,
                use_bias=False):
    """
    The MobileNet V2 model architecture.

    Args:
        inputs(tensor): Input tensor.
        input_shape(tuple, None): Shape of the input tensor, can be None.
        alpha(float): The alpha parameter, defaults to 1.0.
        depth_multiplier(int): Depth multiplier for Depthwise Conv, defaults to 1.
        stride(int): The total stride of this model.
        add_head(bool): Whether or not to add the ImageNet head. If not, will add dense head.
        data_format(str): Data format, can be channels_first or channels_last.
        kernel_regularizer: Kernel regularizer applied to the model.
        bias_regularizer: Bias regularizer applied to the model.
        nclasses(int): Number of classes the output will be classified into.
        use_batch_norm(bool): Whether or not to use the BN layer.
        activation_type(str): Activation type, can be relu or relu6.
        freeze_bn(bool): Whether or not to freeze the BN layers.
        freeze_blocks(list): the list of blocks in the model to be frozen.

    Returns:
        The output tensor.

    """
    assert stride in [16, 32], (
        "Only stride 16 and 32 are supported, got {}".format(stride)
    )
    old_data_format = backend.image_data_format()
    backend.set_image_data_format(data_format)
    if freeze_blocks is None:
        freeze_blocks = []
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    if input_shape is None:
        if backend.image_data_format() == 'channels_first':
            input_shape = (3, 224, 224)
        else:
            input_shape = (224, 224, 3)

    if inputs is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(inputs):
            img_input = layers.Input(tensor=inputs, shape=input_shape)
        else:
            img_input = inputs

    first_block_filters = _make_divisible(32 * alpha, 8)
    # Use explicit padding.
    x = layers.ZeroPadding2D((1, 1), name='conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=use_bias,
                      name='conv1',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      trainable=not(0 in freeze_blocks))(x)

    if use_batch_norm:
        if freeze_bn:
            x = layers.BatchNormalization(axis=channel_axis,
                                          epsilon=1e-3,
                                          momentum=0.999,
                                          name='bn_conv1')(x, training=False)
        else:
            x = layers.BatchNormalization(axis=channel_axis,
                                          epsilon=1e-3,
                                          momentum=0.999,
                                          name='bn_conv1')(x)

    if activation_type == 'relu6':
        x = layers.ReLU(6., name='re_lu_0')(x)
    else:
        x = layers.ReLU(name='re_lu_0')(x)

    with arg_scope([_inverted_res_block],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   activation_type=activation_type,
                   all_projections=all_projections,
                   use_bias=use_bias,
                   freeze_bn=freeze_bn):
        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0,
                                trainable=not(1 in freeze_blocks))

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1,
                                trainable=not(2 in freeze_blocks))
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2,
                                trainable=not(3 in freeze_blocks))

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3,
                                trainable=not(4 in freeze_blocks))
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4,
                                trainable=not(5 in freeze_blocks))
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5,
                                trainable=not(6 in freeze_blocks))

        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                                expansion=6, block_id=6,
                                trainable=not(7 in freeze_blocks))
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=7,
                                trainable=not(8 in freeze_blocks))
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=8,
                                trainable=not(9 in freeze_blocks))
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=9,
                                trainable=not(10 in freeze_blocks))

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=10,
                                trainable=not(11 in freeze_blocks))
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=11,
                                trainable=not(12 in freeze_blocks))
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=12,
                                trainable=not(13 in freeze_blocks))

        if stride == 32:
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                                    expansion=6, block_id=13,
                                    trainable=not(14 in freeze_blocks))
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                                    expansion=6, block_id=14,
                                    trainable=not(15 in freeze_blocks))
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                                    expansion=6, block_id=15,
                                    trainable=not(16 in freeze_blocks))
            x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                                    expansion=6, block_id=16,
                                    trainable=not(17 in freeze_blocks))

            # no alpha applied to last conv as stated in the paper:
            # if the width multiplier is greater than 1 we
            # increase the number of output channels
            if alpha > 1.0:
                last_block_filters = _make_divisible(1280 * alpha, 8)
            else:
                last_block_filters = 1280

            x = layers.Conv2D(last_block_filters,
                              kernel_size=1,
                              use_bias=use_bias,
                              name='conv_1',
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              trainable=not(18 in freeze_blocks))(x)

            if use_batch_norm:
                if freeze_bn:
                    x = layers.BatchNormalization(epsilon=1e-3,
                                                  axis=channel_axis,
                                                  momentum=0.999,
                                                  name='conv_1_bn')(x, training=False)
                else:
                    x = layers.BatchNormalization(epsilon=1e-3,
                                                  axis=channel_axis,
                                                  momentum=0.999,
                                                  name='conv_1_bn')(x)
            if activation_type == 'relu6':
                x = layers.ReLU(6., name='re_lu_head')(x)
            else:
                x = layers.ReLU(name='re_lu_head')(x)
        if add_head:
            x = layers.AveragePooling2D(pool_size=(7, 7),
                                        data_format=data_format,
                                        padding='valid')(x)
            x = layers.Flatten(name='flatten_1')(x)
            x = layers.Dense(nclasses,
                             activation='softmax',
                             name='predictions',
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer)(x)

    # Create model.
    model_name = 'mobilenet_v2'
    if use_batch_norm:
        model_name += '_bn'
    if add_head:
        model_name += '_add_head'

    model = models.Model(img_input, x, name=model_name)

    backend.set_image_data_format(old_data_format)
    return model
