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

"""IVA YOLO Decode Layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class YOLOv4DecodeLayer(Layer):
    '''Decodes model output to corner-formatted boxes.'''

    def call(self, x):
        '''
        Decode output.

        Args:
            x: 3-D tensor. Last dimension is
                (cy, cx, ph, pw, step_y, step_x, pred_y, pred_x, pred_h, pred_w, object, cls...)
        Returns:
            boxes: 3-D tensor. Last dimension is (x_min, y_min, x_max, y_max, cls_score)
        '''

        # shape [..., num_cls]
        # !!! DO NOT replace `:, :, :,` with `...,` as this fails TensorRT export

        cls_score = tf.sigmoid(x[:, :, 11:]) * tf.sigmoid(x[:, :, 10:11])
        by = x[:, :, 0:1] + x[:, :, 6:7] * x[:, :, 4:5]  # shape [..., 1]
        bx = x[:, :, 1:2] + x[:, :, 7:8] * x[:, :, 5:6]  # shape [..., 1]
        bh = x[:, :, 2:3] * x[:, :, 8:9]  # shape [..., 1]
        bw = x[:, :, 3:4] * x[:, :, 9:10]  # shape [..., 1]
        x_min = bx - 0.5 * bw
        x_max = bx + 0.5 * bw
        y_min = by - 0.5 * bh
        y_max = by + 0.5 * bh

        # tf.concat(axis=-1) can't be processed correctly by uff converter.
        return K.concatenate([x_min, y_min, x_max, y_max, cls_score], -1)

    def compute_output_shape(self, input_shape):
        '''Layer output shape function.'''
        return (input_shape[0], input_shape[1], input_shape[2] - 7)
