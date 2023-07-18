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

"""TF implementation of SSD output decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.topology import InputSpec, Layer
import tensorflow as tf


class DecodeDetections(Layer):
    '''
    A Keras layer to decode the raw SSD prediction output.

    Input shape:
        3D tensor of shape `(batch_size, n_boxes, n_classes + 12)`.

    Output shape:
        3D tensor of shape `(batch_size, top_k, 6)`.
    '''

    def __init__(self,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=200,
                 nms_max_output_size=400,
                 img_height=None,
                 img_width=None,
                 **kwargs):
        '''Init function.'''
        if (img_height is None) or (img_width is None):
            raise ValueError("If relative box coordinates are supposed to be converted to absolute \
coordinates, the decoder needs the image size in order to decode the predictions, but \
`img_height == {}` and `img_width == {}`".format(img_height, img_width))

        # We need these members for the config.
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.img_height = img_height
        self.img_width = img_width
        self.nms_max_output_size = nms_max_output_size

        super(DecodeDetections, self).__init__(**kwargs)

    def build(self, input_shape):
        '''Layer build function.'''
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetections, self).build(input_shape)

    def call(self, y_pred, mask=None):
        '''
        Layer call function.

        Input shape:
            3D tensor of shape `(batch_size, n_boxes, n_classes + 12)`.

        Returns:
            3D tensor of shape `(batch_size, top_k, 6)`. The second axis is zero-padded
            to always yield `top_k` predictions per batch item. The last axis contains
            the coordinates for each predicted box in the format
            `[class_id, confidence, xmin, ymin, xmax, ymax]`.
        '''

        # 1. calculate boxes location

        scores = y_pred[..., 1:-12]

        cx_pred = y_pred[..., -12]
        cy_pred = y_pred[..., -11]
        w_pred = y_pred[..., -10]
        h_pred = y_pred[..., -9]
        w_anchor = y_pred[..., -6] - y_pred[..., -8]
        h_anchor = y_pred[..., -5] - y_pred[..., -7]
        cx_anchor = tf.truediv(y_pred[..., -6] + y_pred[..., -8], 2.0)
        cy_anchor = tf.truediv(y_pred[..., -5] + y_pred[..., -7], 2.0)
        cx_variance = y_pred[..., -4]
        cy_variance = y_pred[..., -3]
        variance_w = y_pred[..., -2]
        variance_h = y_pred[..., -1]

        # Convert anchor box offsets to image offsets.
        cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cy = cy_pred * cy_variance * h_anchor + cy_anchor
        w = tf.exp(w_pred * variance_w) * w_anchor
        h = tf.exp(h_pred * variance_h) * h_anchor

        # Convert 'centroids' to 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        xmin = tf.expand_dims(xmin * self.img_width, axis=-1)
        ymin = tf.expand_dims(ymin * self.img_height, axis=-1)
        xmax = tf.expand_dims(xmax * self.img_width, axis=-1)
        ymax = tf.expand_dims(ymax * self.img_height, axis=-1)

        # [batch_size, num_boxes, 1, 4]
        boxes = tf.stack(values=[xmin, ymin, xmax, ymax], axis=-1)

        # 2. apply NMS

        nmsed_box, nmsed_score, nmsed_class, _ = tf.image.combined_non_max_suppression(
            boxes,
            scores,
            max_output_size_per_class=self.nms_max_output_size,
            max_total_size=self.top_k,
            iou_threshold=self.iou_threshold,
            score_threshold=self.confidence_thresh,
            pad_per_class=False,
            clip_boxes=False,
            name='batched_nms')

        nmsed_class += 1
        nmsed_score = tf.expand_dims(nmsed_score, axis=-1)
        nmsed_class = tf.expand_dims(nmsed_class, axis=-1)
        outputs = tf.concat([nmsed_class, nmsed_score, nmsed_box], axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        '''Keras layer compute_output_shape.'''
        batch_size, _, _ = input_shape
        return (batch_size, self.top_k, 6)  # Last axis: (cls_ID, confidence, 4 box coordinates)

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'img_height': self.img_height,
            'img_width': self.img_width,
        }
        base_config = super(DecodeDetections, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
