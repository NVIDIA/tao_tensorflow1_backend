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

"""helper layers for model export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.layers import Layer


class BoxLayer(Layer):
    '''
    Helper layer to export model - Get box.

    Input:
        Encoded detection (last layer output of training model).
    Output:
        Boxes in corner format (x_min, y_min, x_max, y_max)
    '''

    def compute_output_shape(self, input_shape):
        '''Define output shape.'''
        return (input_shape[0], input_shape[1], 1, 4)

    def call(self, x):
        '''See class doc.'''
        x_shape = K.shape(x)
        x = K.reshape(x, [x_shape[0], x_shape[1], 1, x_shape[2]])
        by = x[:, :, :, 0:1] + K.sigmoid(x[:, :, :, 6:7]) * x[:, :, :, 4:5]  # shape [..., 1]
        bx = x[:, :, :, 1:2] + K.sigmoid(x[:, :, :, 7:8]) * x[:, :, :, 5:6]  # shape [..., 1]
        bh = x[:, :, :, 2:3] * K.exp(x[:, :, :, 8:9])  # shape [..., 1]
        bw = x[:, :, :, 3:4] * K.exp(x[:, :, :, 9:10])  # shape [..., 1]
        x_min = bx - 0.5 * bw
        x_max = bx + 0.5 * bw
        y_min = by - 0.5 * bh
        y_max = by + 0.5 * bh
        loc = K.concatenate([x_min, y_min, x_max, y_max], -1)

        return K.identity(loc, name="out_box")


class ClsLayer(Layer):
    '''
    Helper layer to export model - Get class score.

    Input:
        Encoded detection (last layer output of training model).
    Output:
        (Sigmoid) confidence scores for each class.
    '''

    def compute_output_shape(self, input_shape):
        '''Define output shape.'''
        return (input_shape[0], input_shape[1], input_shape[2]-11, 1)

    def call(self, x):
        '''See class doc.'''

        # shape [..., num_cls]
        x_shape = K.shape(x)
        x = K.reshape(x, [x_shape[0], x_shape[1], x_shape[2], 1])
        cls_score = K.sigmoid(x[:, :, 11:, :]) * K.sigmoid(x[:, :, 10:11, :])
        return K.identity(cls_score, name="out_cls")
