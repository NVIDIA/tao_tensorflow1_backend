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
"""MobileNet V1 model for FasterRCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import AveragePooling2D, BatchNormalization, \
                         Conv2D, DepthwiseConv2D, \
                         Flatten, ReLU, TimeDistributed, \
                         ZeroPadding2D

from nvidia_tao_tf1.core.templates.utils import add_arg_scope, arg_scope
from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


class MobileNetV1(FrcnnModel):
    '''MobileNet V1 as backbones for FasterRCNN model.

    This is MobileNet V1 class that use FrcnnModel class as base class and do some customization
    specific to MobileNet V1 backbone. Methods here will override those functions in FrcnnModel
    class.
    '''

    def backbone(self, input_images):
        '''backbone for MobileNet V1 FasterRCNN.'''
        with arg_scope([_conv_block, _depthwise_conv_block],
                       use_batch_norm=True,
                       data_format='channels_first',
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       activation_type='relu',
                       freeze_bn=self.freeze_bn,
                       use_bias=not self.conv_bn_share_bias):
            x = _conv_block(input_images, 32, 1, strides=(2, 2),
                            trainable=not (0 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 64, 1,
                                      block_id=1,
                                      trainable=not (1 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 128, 1,
                                      strides=(2, 2), block_id=2,
                                      trainable=not (2 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 128, 1, block_id=3,
                                      trainable=not (3 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 256, 1,
                                      strides=(2, 2), block_id=4,
                                      trainable=not (4 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 256, 1, block_id=5,
                                      trainable=not (5 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 512, 1,
                                      strides=(2, 2), block_id=6,
                                      trainable=not (6 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 512, 1, block_id=7,
                                      trainable=not (7 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 512, 1, block_id=8,
                                      trainable=not (8 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 512, 1, block_id=9,
                                      trainable=not (9 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 512, 1, block_id=10,
                                      trainable=not (10 in self.freeze_blocks))
            x = _depthwise_conv_block(x, 512, 1, block_id=11,
                                      trainable=not (11 in self.freeze_blocks))
        return x

    def rcnn_body(self, x):
        '''RCNN body for MobileNet V1.'''
        _stride = 2 if self.roi_pool_2x else 1
        with arg_scope([_depthwise_conv_block],
                       use_batch_norm=True,
                       data_format='channels_first',
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       activation_type='relu',
                       freeze_bn=self.freeze_bn,
                       use_bias=not self.conv_bn_share_bias,
                       use_td=True):
            x = _depthwise_conv_block(x, 1024, 1,
                                      strides=(_stride, _stride), block_id=12)
            x = _depthwise_conv_block(x, 1024, 1, block_id=13)
        x = TimeDistributed(AveragePooling2D(pool_size=(self.roi_pool_size, self.roi_pool_size),
                                             data_format='channels_first', padding='valid'))(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        x = TimeDistributed(Flatten(name='classifier_flatten'), name='time_distributed_flatten')(x)
        return x


@add_arg_scope
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1),
                kernel_regularizer=None, bias_regularizer=None,
                use_batch_norm=True, activation_type='relu',
                data_format='channels_first', freeze_bn=False, use_bias=True,
                trainable=True):
    """Adds an initial convolution layer (with batch normalization and relu).

    Args:
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    Input shape:
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    Returns:
        Output tensor of block.
    """
    channel_axis = 1 if data_format == 'channels_first' else -1
    filters = int(filters * alpha)
    if kernel[0] // 2 > 0:
        x = ZeroPadding2D(padding=(kernel[0]//2, kernel[0]//2),
                          name='conv1_pad')(inputs)
    x = Conv2D(filters, kernel,
               padding='valid',
               use_bias=use_bias,
               strides=strides,
               name='conv1',
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               trainable=trainable)(x)
    if use_batch_norm:
        if freeze_bn:
            x = BatchNormalization(axis=channel_axis,
                                   name='conv1_bn')(x, training=False)
        else:
            x = BatchNormalization(axis=channel_axis,
                                   name='conv1_bn')(x)
    if activation_type == 'relu6':
        x = ReLU(6.)(x)
    else:
        x = ReLU()(x)
    return x


@add_arg_scope
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1,
                          kernel_regularizer=None, bias_regularizer=None,
                          use_batch_norm=True, activation_type='relu',
                          data_format='channels_first', freeze_bn=False,
                          use_bias=True,
                          trainable=True,
                          use_td=False):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu, pointwise convolution,
    batch normalization and relu activation.
    Args:
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    Input shape:
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    Returns:
        Output tensor of block.
    """
    channel_axis = 1 if data_format == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    layer = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)
    if use_td:
        layer = TimeDistributed(layer)
    x = layer(inputs)

    layer = DepthwiseConv2D((3, 3),
                            padding='valid',
                            depth_multiplier=depth_multiplier,
                            strides=strides,
                            use_bias=use_bias,
                            name='conv_dw_%d' % block_id,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            trainable=trainable)
    if use_td:
        layer = TimeDistributed(layer)
    x = layer(x)

    if use_batch_norm:
        layer = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)
        if use_td:
            layer = TimeDistributed(layer)
        if freeze_bn:
            x = layer(x, training=False)
        else:
            x = layer(x)

    if activation_type == 'relu6':
        x = ReLU(6.)(x)
    else:
        x = ReLU()(x)

    layer = Conv2D(pointwise_conv_filters,
                   (1, 1),
                   padding='same',
                   use_bias=use_bias,
                   strides=(1, 1),
                   name='conv_pw_%d' % block_id,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   trainable=trainable)
    if use_td:
        layer = TimeDistributed(layer)
    x = layer(x)
    if use_batch_norm:
        layer = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)
        if use_td:
            layer = TimeDistributed(layer)
        if freeze_bn:
            x = layer(x, training=False)
        else:
            x = layer(x)

    if activation_type == 'relu6':
        x = ReLU(6.)(x)
    else:
        x = ReLU()(x)

    return x
