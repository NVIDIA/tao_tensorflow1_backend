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
"""ResNet101 model for FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, \
                         Conv2D, Flatten, \
                         MaxPooling2D, TimeDistributed, ZeroPadding2D

from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


class ResNet101(FrcnnModel):
    '''ResNet101 as backbones for FasterRCNN model.

    This is ResNet101 class that uses FrcnnModel class as base class and do some customization
    specific to ResNet101 backbone. Methods here will override those functions in FrcnnModel class.
    '''

    def backbone(self, input_images):
        '''backbone of the ResNet FasterRCNN model.'''
        bn_axis = 1
        freeze0 = bool(0 in self.freeze_blocks)
        freeze1 = bool(1 in self.freeze_blocks)
        freeze2 = bool(2 in self.freeze_blocks)
        freeze3 = bool(3 in self.freeze_blocks)

        x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input_images)
        x = Conv2D(
            64, 7, strides=2,
            use_bias=not self.conv_bn_share_bias,
            kernel_regularizer=self.kernel_reg,
            name='conv1_conv',
            trainable=not freeze0
        )(x)
        if self.freeze_bn:
            x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name='conv1_bn')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        x = self.stack1(x, 64, 3, stride1=1, name='conv2',
                        freeze_bn=self.freeze_bn, freeze=freeze1)
        x = self.stack1(x, 128, 4, name='conv3',
                        freeze_bn=self.freeze_bn, freeze=freeze2)
        x = self.stack1(x, 256, 23, name='conv4',
                        freeze_bn=self.freeze_bn, freeze=freeze3)
        return x

    def rcnn_body(self, x):
        '''RCNN body.'''
        _stride = 2 if self.roi_pool_2x else 1
        x = self.stack1(x, 512, 3, name='conv5', stride1=_stride,
                        freeze_bn=self.freeze_bn, use_td=True)
        x = TimeDistributed(AveragePooling2D((self.roi_pool_size, self.roi_pool_size),
                                             name='avg_pool'))(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        x = TimeDistributed(Flatten(name='classifier_flatten'),
                            name='time_distributed_flatten')(x)
        return x

    def block1(self, x, filters, kernel_size=3, stride=1,
               conv_shortcut=True, name=None, freeze_bn=False,
               freeze=False, use_td=False):
        """A residual block."""
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        if conv_shortcut is True:
            layer = Conv2D(
                4 * filters,
                1,
                strides=stride,
                name=name + '_0_conv',
                trainable=not freeze,
                kernel_regularizer=self.kernel_reg
            )
            if use_td:
                layer = TimeDistributed(layer)
            shortcut = layer(x)
            layer = BatchNormalization(
                axis=bn_axis,
                epsilon=1.001e-5,
                name=name + '_0_bn',
            )
            if use_td:
                layer = TimeDistributed(layer)
            if freeze_bn:
                shortcut = layer(shortcut, training=False)
            else:
                shortcut = layer(shortcut)
        else:
            shortcut = x

        layer = Conv2D(
            filters, 1, strides=stride,
            name=name + '_1_conv',
            trainable=not freeze,
            kernel_regularizer=self.kernel_reg
        )
        if use_td:
            layer = TimeDistributed(layer)
        x = layer(x)

        layer = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5,
            name=name + '_1_bn'
        )
        if use_td:
            layer = TimeDistributed(layer)
        if freeze_bn:
            x = layer(x, training=False)
        else:
            x = layer(x)
        x = Activation('relu', name=name + '_1_relu')(x)

        layer = Conv2D(
            filters, kernel_size, padding='SAME',
            name=name + '_2_conv',
            trainable=not freeze,
            kernel_regularizer=self.kernel_reg
        )
        if use_td:
            layer = TimeDistributed(layer)
        x = layer(x)

        layer = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5,
            name=name + '_2_bn'
        )
        if use_td:
            layer = TimeDistributed(layer)
        if freeze_bn:
            x = layer(x, training=False)
        else:
            x = layer(x)
        x = Activation('relu', name=name + '_2_relu')(x)

        layer = Conv2D(
            4 * filters, 1,
            name=name + '_3_conv',
            trainable=not freeze,
            kernel_regularizer=self.kernel_reg
        )
        if use_td:
            layer = TimeDistributed(layer)
        x = layer(x)

        layer = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5,
            name=name + '_3_bn'
        )
        if use_td:
            layer = TimeDistributed(layer)
        if freeze_bn:
            x = layer(x, training=False)
        else:
            x = layer(x)

        x = Add(name=name + '_add')([shortcut, x])
        x = Activation('relu', name=name + '_out')(x)
        return x

    def stack1(self, x, filters, blocks, stride1=2, name=None,
               freeze_bn=False, freeze=False, use_td=False):
        """A set of stacked residual blocks."""
        x = self.block1(x, filters, stride=stride1, name=name + '_block1',
                        freeze=freeze, freeze_bn=freeze_bn, use_td=use_td)
        for i in range(2, blocks + 1):
            x = self.block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i),
                            freeze=freeze, freeze_bn=freeze_bn, use_td=use_td)
        return x
