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

"""TLT LPRNet baseline model."""

import tensorflow as tf

# @TODO(tylerz): Shall we use fixed padding as caffe?
# def Conv2DFixedPadding(input, filters, kernel_size, strides):
#     return net

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4


def BNReLULayer(inputs, name, trainable, relu=True,
                init_zero=False, data_format='channels_last'):
    '''Fusion block of bn and relu.'''

    if init_zero:
        gamma_initializer = tf.keras.initializers.Zeros()
    else:
        gamma_initializer = tf.keras.initializers.Ones()

    if data_format == 'channels_first':
        axis = 1
    else:
        axis = -1

    net = tf.keras.layers.BatchNormalization(
                            axis=axis,
                            momentum=_BATCH_NORM_DECAY,
                            epsilon=_BATCH_NORM_EPSILON,
                            center=True,
                            scale=True,
                            trainable=trainable,
                            gamma_initializer=gamma_initializer,
                            fused=True,
                            name=name
                            )(inputs)

    if relu:
        net = tf.keras.layers.ReLU()(net)

    return net


def ResidualBlock(inputs, block_idx, filters, kernel_size,
                  strides, use_projection=False, finetune_bn=True,
                  trainable=True, data_format="channels_last",
                  kernel_regularizer=None, bias_regularizer=None):
    '''Fusion layer of Residual block.'''

    # # @TODO(tylerz): shall we init conv kernels with glorot_normal as caffe ????
    # kernel_initializer = "glorot_normal"
    # # @TODO(tylerz): shall we init conv bias with 0.2 as caffe???
    # bias_initializer = tf.constant_initializer(0.2)

    # branch1
    if use_projection:
        shortcut = tf.keras.layers.Conv2D(
                   filters=filters,
                   kernel_size=1,
                   strides=strides,
                   padding="valid",
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   name="res%s_branch1" % block_idx,
                   data_format=data_format,
                   trainable=trainable
                   )(inputs)

        shortcut = BNReLULayer(inputs=shortcut, name="bn%s_branch1" % block_idx,
                               trainable=trainable and finetune_bn,
                               relu=False, init_zero=False,
                               data_format=data_format)
    else:
        shortcut = inputs

    # branch2a
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        name="res%s_branch2a" % block_idx,
        data_format=data_format,
        trainable=trainable
        )(inputs)

    x = BNReLULayer(inputs=x, name="bn%s_branch2a" % block_idx,
                    trainable=trainable and finetune_bn,
                    relu=True, init_zero=False,
                    data_format=data_format)
    # branch2b
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        name="res%s_branch2b" % block_idx,
        data_format=data_format,
        trainable=trainable
        )(x)

    # @TODO(tylerz): Shall we init gamma zero here ??????
    x = BNReLULayer(inputs=x, name="bn%s_branch2b" % block_idx,
                    trainable=trainable and finetune_bn,
                    relu=False, init_zero=False,
                    data_format=data_format)

    net = tf.keras.layers.ReLU()(shortcut+x)

    return net


class LPRNetbaseline:
    '''Tuned ResNet18 and ResNet10 as the baseline for LPRNet.'''

    def __init__(self, nlayers=18, freeze_bn=False,
                 kernel_regularizer=None, bias_regularizer=None):
        '''Initialize the parameter for LPRNet baseline model.'''

        self.finetune_bn = (not freeze_bn)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        assert nlayers in (10, 18), print("LPRNet baseline model only supports 18 and 10 layers")
        self.nlayers = nlayers

    def __call__(self, input_tensor, trainable):
        '''Generate LPRNet baseline model.'''

        finetune_bn = self.finetune_bn
        kernel_regularizer = self.kernel_regularizer
        bias_regularizer = self.bias_regularizer
        data_format = "channels_first"

        # block1:
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   name="conv1",
                                   data_format=data_format,
                                   trainable=trainable
                                   )(input_tensor)

        x = BNReLULayer(inputs=x, name="bn_conv1",
                        trainable=trainable and finetune_bn,
                        relu=True, init_zero=False,
                        data_format=data_format)

        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=(1, 1),
                                      padding="same", data_format=data_format)(x)

        # block 2a
        x = ResidualBlock(inputs=x, block_idx="2a", filters=64,
                          kernel_size=3, strides=(1, 1), use_projection=True,
                          finetune_bn=finetune_bn, trainable=trainable,
                          data_format=data_format,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer)

        if self.nlayers == 18:
            # block 2b
            x = ResidualBlock(inputs=x, block_idx="2b", filters=64,
                              kernel_size=3, strides=(1, 1), use_projection=False,
                              finetune_bn=finetune_bn, trainable=trainable,
                              data_format=data_format,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer)
        # block 3a
        x = ResidualBlock(inputs=x, block_idx="3a", filters=128,
                          kernel_size=3, strides=(2, 2), use_projection=True,
                          finetune_bn=finetune_bn, trainable=trainable,
                          data_format=data_format,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer)

        if self.nlayers == 18:
            # block 3b
            x = ResidualBlock(inputs=x, block_idx="3b", filters=128,
                              kernel_size=3, strides=(1, 1), use_projection=False,
                              finetune_bn=finetune_bn, trainable=trainable,
                              data_format=data_format,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer)
        # block 4a
        x = ResidualBlock(inputs=x, block_idx="4a", filters=256,
                          kernel_size=3, strides=(2, 2), use_projection=True,
                          finetune_bn=finetune_bn, trainable=trainable,
                          data_format=data_format,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer)

        if self.nlayers == 18:
            # block 4b
            x = ResidualBlock(inputs=x, block_idx="4b", filters=256,
                              kernel_size=3, strides=(1, 1), use_projection=False,
                              finetune_bn=finetune_bn, trainable=trainable,
                              data_format=data_format,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer)

        # block 5a
        x = ResidualBlock(inputs=x, block_idx="5a", filters=300,
                          kernel_size=3, strides=(1, 1), use_projection=True,
                          finetune_bn=finetune_bn, trainable=trainable,
                          data_format=data_format,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer)

        if self.nlayers == 18:
            # block 5b
            x = ResidualBlock(inputs=x, block_idx="5b", filters=300,
                              kernel_size=3, strides=(1, 1), use_projection=False,
                              finetune_bn=finetune_bn, trainable=trainable,
                              data_format=data_format,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer)

        return x
