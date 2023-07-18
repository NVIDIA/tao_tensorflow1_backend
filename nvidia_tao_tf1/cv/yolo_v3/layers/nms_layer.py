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

"""IVA YOLO NMS Layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.topology import Layer
import tensorflow as tf


class NMSLayer(Layer):
    '''
    NMS layer to get final outputs from raw pred boxes.

    Args:
        output_size: how many boxes you want for final outputs (padded by zeros)
        iou_threshold: boxes with iou > threshold will be NMSed
        score_threshold: Remove boxes with confidence less than threshold before NMS.
    '''

    def __init__(self,
                 output_size=200,
                 iou_threshold=0.5,
                 score_threshold=0.01,
                 force_on_cpu=False,
                 **kwargs):
        '''Init function.'''

        self.output_size = output_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.force_on_cpu = force_on_cpu
        super(NMSLayer, self).__init__(**kwargs)

    def call(self, x):
        '''
        Perform NMS on output.

        Args:
            x: 3-D tensor. Last dimension is (x_min, y_min, x_max, y_max, cls_confidence[0, 1, ...])
        Returns:
            results: 3-D Tensor, [num_batch, output_size, 6].
                The last dim is (cls_inds, cls_score, xmin, ymin, xmax, ymax)
        '''
        if not self.force_on_cpu:
            nmsed_box, nmsed_score, nmsed_class, _ = tf.image.combined_non_max_suppression(
                tf.expand_dims(x[..., :4], axis=2),
                x[..., 4:],
                max_output_size_per_class=self.output_size,
                max_total_size=self.output_size,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                pad_per_class=False,
                clip_boxes=True,
                name='batched_nms'
            )
        else:
            with tf.device("cpu:0"):
                nmsed_box, nmsed_score, nmsed_class, _ = tf.image.combined_non_max_suppression(
                    tf.expand_dims(x[..., :4], axis=2),
                    x[..., 4:],
                    max_output_size_per_class=self.output_size,
                    max_total_size=self.output_size,
                    iou_threshold=self.iou_threshold,
                    score_threshold=self.score_threshold,
                    pad_per_class=False,
                    clip_boxes=True,
                    name='batched_nms'
                )
        nmsed_score = tf.expand_dims(nmsed_score, axis=-1)
        nmsed_class = tf.expand_dims(nmsed_class, axis=-1)
        outputs = tf.concat([nmsed_class, nmsed_score, nmsed_box], axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        '''Layer output shape function.'''
        return (input_shape[0], self.output_size, 6)

    def get_config(self):
        '''Layer get_config function.'''
        config = {
            'output_size': self.output_size,
            'iou_threshold': self.iou_threshold,
            'score_threshold': self.score_threshold,
            'force_on_cpu': self.force_on_cpu,
        }
        base_config = super(NMSLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
