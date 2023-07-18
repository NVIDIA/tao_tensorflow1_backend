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
"""MobileNet V2 model for FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Add, AveragePooling2D, BatchNormalization, Conv2D, \
                         DepthwiseConv2D, Flatten, ReLU, \
                         TimeDistributed, ZeroPadding2D

from nvidia_tao_tf1.core.templates.utils import add_arg_scope, arg_scope
from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


class MobileNetV2(FrcnnModel):
    '''MobileNet V2 as backbones for FasterRCNN model.

    This is MobileNet V2 class that use FrcnnModel class as base class and do some customization
    specific to MobileNet V2 backbone. Methods here will override those functions in FrcnnModel
    class.
    '''

    def backbone(self, input_images):
        '''backbone for MobileNet V2 FasterRCNN.'''
        channel_axis = 1
        first_block_filters = _make_divisible(32, 8)
        x = ZeroPadding2D((1, 1), name='conv1_pad')(input_images)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2),
                   padding='valid',
                   use_bias=not self.conv_bn_share_bias,
                   name='conv1',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not (0 in self.freeze_blocks))(x)
        if self.freeze_bn:
            x = BatchNormalization(axis=channel_axis,
                                   epsilon=1e-3,
                                   momentum=0.999,
                                   name='bn_conv1')(x, training=False)
        else:
            x = BatchNormalization(axis=channel_axis,
                                   epsilon=1e-3,
                                   momentum=0.999,
                                   name='bn_conv1')(x)
        x = ReLU()(x)
        with arg_scope([_inverted_res_block],
                       use_batch_norm=True,
                       data_format='channels_first',
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       activation_type='relu',
                       all_projections=self.all_projections,
                       freeze_bn=self.freeze_bn,
                       use_bias=not self.conv_bn_share_bias):
            x = _inverted_res_block(x, filters=16, alpha=1, stride=1,
                                    expansion=1, block_id=0,
                                    trainable=not (1 in self.freeze_blocks))

            x = _inverted_res_block(x, filters=24, alpha=1, stride=2,
                                    expansion=6, block_id=1,
                                    trainable=not(2 in self.freeze_blocks))
            x = _inverted_res_block(x, filters=24, alpha=1, stride=1,
                                    expansion=6, block_id=2,
                                    trainable=not (3 in self.freeze_blocks))

            x = _inverted_res_block(x, filters=32, alpha=1, stride=2,
                                    expansion=6, block_id=3,
                                    trainable=not (4 in self.freeze_blocks))
            x = _inverted_res_block(x, filters=32, alpha=1, stride=1,
                                    expansion=6, block_id=4,
                                    trainable=not (5 in self.freeze_blocks))
            x = _inverted_res_block(x, filters=32, alpha=1, stride=1,
                                    expansion=6, block_id=5,
                                    trainable=not (6 in self.freeze_blocks))

            x = _inverted_res_block(x, filters=64, alpha=1, stride=2,
                                    expansion=6, block_id=6,
                                    trainable=not (7 in self.freeze_blocks))
            x = _inverted_res_block(x, filters=64, alpha=1, stride=1,
                                    expansion=6, block_id=7,
                                    trainable=not (8 in self.freeze_blocks))
            x = _inverted_res_block(x, filters=64, alpha=1, stride=1,
                                    expansion=6, block_id=8,
                                    trainable=not (9 in self.freeze_blocks))
            x = _inverted_res_block(x, filters=64, alpha=1, stride=1,
                                    expansion=6, block_id=9,
                                    trainable=not (10 in self.freeze_blocks))

            x = _inverted_res_block(x, filters=96, alpha=1, stride=1,
                                    expansion=6, block_id=10,
                                    trainable=not (11 in self.freeze_blocks))
            x = _inverted_res_block(x, filters=96, alpha=1, stride=1,
                                    expansion=6, block_id=11,
                                    trainable=not (12 in self.freeze_blocks))
            x = _inverted_res_block(x, filters=96, alpha=1, stride=1,
                                    expansion=6, block_id=12,
                                    trainable=not (13 in self.freeze_blocks))
        return x

    def rcnn_body(self, x):
        '''RCNN body for MobileNet V2 FasterRCNN model.'''
        _stride = 2 if self.roi_pool_2x else 1
        channel_axis = 1
        with arg_scope([_inverted_res_block],
                       use_batch_norm=True,
                       data_format='channels_first',
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       activation_type='relu',
                       all_projections=self.all_projections,
                       freeze_bn=self.freeze_bn,
                       use_bias=not self.conv_bn_share_bias,
                       use_td=True):
            x = _inverted_res_block(x, filters=160, alpha=1, stride=_stride,
                                    expansion=6, block_id=13)
            x = _inverted_res_block(x, filters=160, alpha=1, stride=1,
                                    expansion=6, block_id=14)
            x = _inverted_res_block(x, filters=160, alpha=1, stride=1,
                                    expansion=6, block_id=15)
            x = _inverted_res_block(x, filters=320, alpha=1, stride=1,
                                    expansion=6, block_id=16)

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        last_block_filters = 1280
        layer = Conv2D(last_block_filters,
                       kernel_size=1,
                       use_bias=not self.conv_bn_share_bias,
                       name='conv_1',
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None)
        layer = TimeDistributed(layer)
        x = layer(x)

        layer = BatchNormalization(epsilon=1e-3,
                                   axis=channel_axis,
                                   momentum=0.999,
                                   name='conv_1_bn')
        layer = TimeDistributed(layer)
        if self.freeze_bn:
            x = layer(x, training=False)
        else:
            x = layer(x)
        x = ReLU()(x)
        x = TimeDistributed(AveragePooling2D(pool_size=(self.roi_pool_size, self.roi_pool_size),
                                             data_format='channels_first',
                                             padding='valid'))(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        x = TimeDistributed(Flatten(name='classifier_flatten'), name='time_distributed_flatten')(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@add_arg_scope
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id,
                        kernel_regularizer=None, bias_regularizer=None,
                        use_batch_norm=True, activation_type='relu',
                        data_format='channels_first', all_projections=False,
                        freeze_bn=False, use_bias=True, trainable=True,
                        use_td=False):
    '''Inverted residual block as building blocks for MobileNet V2.'''
    channel_axis = 1 if data_format == 'channels_first' else -1
    # if use TD layer, then channel axis should + 1 since input is now a 5D tensor
    if use_td and channel_axis == 1:
        in_channels = inputs._keras_shape[channel_axis+1]
    else:
        in_channels = inputs._keras_shape[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        layer = Conv2D(expansion * in_channels,
                       kernel_size=1,
                       padding='valid',
                       use_bias=use_bias,
                       activation=None,
                       name=prefix + 'expand',
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       trainable=trainable)
        if use_td:
            layer = TimeDistributed(layer)
        x = layer(x)
        if use_batch_norm:
            layer = BatchNormalization(epsilon=1e-3, axis=channel_axis,
                                       momentum=0.999,
                                       name=prefix + 'expand_bn')
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
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    layer = ZeroPadding2D((1, 1), name=prefix + 'depthwise_pad')
    if use_td:
        layer = TimeDistributed(layer)
    x = layer(x)

    layer = DepthwiseConv2D(kernel_size=3,
                            strides=stride,
                            activation=None,
                            use_bias=use_bias,
                            padding='valid',
                            name=prefix + 'depthwise',
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            trainable=trainable)
    if use_td:
        layer = TimeDistributed(layer)
    x = layer(x)
    if use_batch_norm:
        layer = BatchNormalization(epsilon=1e-3,
                                   axis=channel_axis,
                                   momentum=0.999,
                                   name=prefix + 'depthwise_bn')
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
    # Project
    layer = Conv2D(pointwise_filters,
                   kernel_size=1,
                   padding='valid',
                   use_bias=use_bias,
                   activation=None,
                   name=prefix + 'project',
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   trainable=trainable)
    if use_td:
        layer = TimeDistributed(layer)
    x = layer(x)
    if use_batch_norm:
        layer = BatchNormalization(axis=channel_axis,
                                   epsilon=1e-3,
                                   momentum=0.999,
                                   name=prefix + 'project_bn')
        if use_td:
            layer = TimeDistributed(layer)
        if freeze_bn:
            x = layer(x, training=False)
        else:
            x = layer(x)

    if in_channels == pointwise_filters and stride == 1:
        if all_projections:
            layer = Conv2D(in_channels,
                           kernel_size=1,
                           padding='valid',
                           use_bias=use_bias,
                           activation=None,
                           name=prefix + 'projected_inputs',
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           trainable=trainable)
            if use_td:
                layer = TimeDistributed(layer)
            inputs_projected = layer(inputs)
            return Add(name=prefix + 'add')([inputs_projected, x])
        return Add(name=prefix + 'add')([inputs, x])
    return x
