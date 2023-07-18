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
"""VGG16 models for FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Conv2D, Dense, Dropout, \
                         Flatten, MaxPooling2D, \
                         TimeDistributed

from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


class VGG16(FrcnnModel):
    '''VGG16 as backbones for FasterRCNN model.

    This is VGG16 class that use FrcnnModel class as base class and do some customization
    specific to VGG16 backbone. Methods here will override those functions in FrcnnModel class.
    '''

    def backbone(self, input_images):
        '''backbone.'''
        # Block 1
        freeze1 = bool(1 in self.freeze_blocks)
        x = Conv2D(64, (3, 3), activation='relu',
                   padding='same', name='block1_conv1',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze1)(input_images)
        x = Conv2D(64, (3, 3), activation='relu',
                   padding='same', name='block1_conv2',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        freeze2 = bool(2 in self.freeze_blocks)
        x = Conv2D(128, (3, 3), activation='relu',
                   padding='same', name='block2_conv1',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze2)(x)
        x = Conv2D(128, (3, 3), activation='relu',
                   padding='same', name='block2_conv2',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        freeze3 = bool(3 in self.freeze_blocks)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='block3_conv1',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze3)(x)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='block3_conv2',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze3)(x)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='block3_conv3',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze3)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        freeze4 = bool(4 in self.freeze_blocks)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block4_conv1',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze4)(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block4_conv2',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze4)(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block4_conv3',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze4)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        freeze5 = bool(5 in self.freeze_blocks)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block5_conv1',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze5)(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block5_conv2',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze5)(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block5_conv3',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None,
                   trainable=not freeze5)(x)
        return x

    def rcnn_body(self, x):
        '''RCNN body.'''
        if self.roi_pool_2x:
            x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2),
                                name='classifier_pool'))(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        out = TimeDistributed(Flatten(name='classifier_flatten'),
                              name='time_distributed_flatten')(x)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1',
                              kernel_regularizer=self.kernel_reg,
                              bias_regularizer=None))(out)
        if self.dropout_rate > 0:
            out = TimeDistributed(Dropout(self.dropout_rate))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2',
                              kernel_regularizer=self.kernel_reg,
                              bias_regularizer=None))(out)
        if self.dropout_rate > 0:
            out = TimeDistributed(Dropout(self.dropout_rate))(out)
        return out
