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
"""GoogleNet as backbone of Faster-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, AveragePooling2D, BatchNormalization, \
                         Conv2D, Flatten, MaxPooling2D, \
                         TimeDistributed

from nvidia_tao_tf1.core.templates.utils import arg_scope, InceptionV1Block
from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


class GoogleNet(FrcnnModel):
    '''GoogleNet as backbone of FasterRCNN model.

    This is GoogleNet class that use FrcnnModel class as base class and do some customization
    specific to GoogleNet backbone. Methods here will override those functions in FrcnnModel class.
    '''

    def backbone(self, input_images):
        '''GoogleNet backbone implementation.'''
        bn_axis = 1
        data_format = 'channels_first'
        x = Conv2D(64,
                   (7, 7),
                   strides=(2, 2),
                   padding='same',
                   data_format=data_format,
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   name='conv1',
                   trainable=not(0 in self.freeze_blocks),
                   use_bias=not self.conv_bn_share_bias)(input_images)
        if self.freeze_bn:
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                         data_format=data_format, name='pool1')(x)
        x = Conv2D(64,
                   (1, 1),
                   strides=(1, 1),
                   padding='same',
                   data_format=data_format,
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   name='conv2_reduce',
                   use_bias=not self.conv_bn_share_bias,
                   trainable=not(0 in self.freeze_blocks))(x)
        if self.freeze_bn:
            x = BatchNormalization(axis=bn_axis, name='bn_conv2_reduce')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, name='bn_conv2_reduce')(x)
        x = Activation('relu')(x)
        x = Conv2D(192,
                   (3, 3),
                   strides=(1, 1),
                   padding='same',
                   data_format=data_format,
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   name='conv2',
                   use_bias=not self.conv_bn_share_bias,
                   trainable=not(0 in self.freeze_blocks))(x)
        if self.freeze_bn:
            x = BatchNormalization(axis=bn_axis, name='bn_conv2')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, name='bn_conv2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                         data_format=data_format, name='pool2')(x)
        # Define a block functor which can create blocks.
        with arg_scope([InceptionV1Block],
                       use_batch_norm=True,
                       data_format=data_format,
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       freeze_bn=self.freeze_bn,
                       activation_type='relu',
                       use_bias=not self.conv_bn_share_bias):
            # Inception_3a
            x = InceptionV1Block(subblocks=(64, 96, 128, 16, 32, 32),
                                 index='3a', trainable=not(1 in self.freeze_blocks))(x)
            # Inception_3b
            x = InceptionV1Block(subblocks=(128, 128, 192, 32, 96, 64),
                                 index='3b', trainable=not(2 in self.freeze_blocks))(x)
            # Max Pooling
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                             data_format=data_format, name='pool3')(x)
            # Inception_4a
            x = InceptionV1Block(subblocks=(192, 96, 208, 16, 48, 64),
                                 index='4a', trainable=not(3 in self.freeze_blocks))(x)
            # Inception_4b
            x = InceptionV1Block(subblocks=(160, 112, 224, 24, 64, 64),
                                 index='4b', trainable=not(4 in self.freeze_blocks))(x)
            # Inception_4c
            x = InceptionV1Block(subblocks=(128, 128, 256, 24, 64, 64),
                                 index='4c', trainable=not(5 in self.freeze_blocks))(x)
            # Inception_4d
            x = InceptionV1Block(subblocks=(112, 144, 288, 32, 64, 64),
                                 index='4d', trainable=not(6 in self.freeze_blocks))(x)
            # Inception_4e
            x = InceptionV1Block(subblocks=(256, 160, 320, 32, 128, 128),
                                 index='4e', trainable=not(7 in self.freeze_blocks))(x)
        return x

    def rcnn_body(self, x):
        '''GoogleNet RCNN body.'''
        if self.roi_pool_2x:
            x = TimeDistributed(MaxPooling2D(pool_size=(2, 2),
                                             strides=(2, 2), padding='same',
                                             data_format='channels_first',
                                             name='pool4'))(x)
        with arg_scope([InceptionV1Block],
                       use_batch_norm=True,
                       data_format='channels_first',
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       freeze_bn=self.freeze_bn,
                       activation_type='relu',
                       use_bias=not self.conv_bn_share_bias,
                       use_td=True):
            # Inception_5a
            x = InceptionV1Block(subblocks=(256, 160, 320, 32, 128, 128), index='5a')(x)
            # Inception_5b
            x = InceptionV1Block(subblocks=(384, 192, 384, 48, 128, 128), index='5b')(x)

        x = TimeDistributed(AveragePooling2D(pool_size=(self.roi_pool_size, self.roi_pool_size),
                            strides=(1, 1), padding='valid',
                            data_format='channels_first', name='avg_pool'))(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        x = TimeDistributed(Flatten(name='classifier_flatten'), name='time_distributed_flatten')(x)
        return x
