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
"""OutputParser layer in FasterRCNN for post-processing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer
import tensorflow as tf


class OutputParser(Layer):
    '''OutputParser layer for post-processing in FasterRCNN.'''

    def __init__(self, max_box_num, regr_std_scaling, iou_thres, score_thres, **kwargs):
        """Initialize the OutputParser layer.

        Args:
            max_box_num(int): maximum number of total boxes for output.
        """
        self.max_box_num = max_box_num
        self.regr_std_scaling = regr_std_scaling
        self.iou_thres = iou_thres
        self.score_thres = score_thres
        super(OutputParser, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """compute_output_shape.

        Args:
            input_shape(tuple): the shape of the input tensor.
        Returns:
            The output 3D tensor shape: (None, B, 6). where 6 is bbox + class + confidence
        """
        return [
            (None, self.max_box_num, 4),
            (None, self.max_box_num),
            (None, self.max_box_num),
            (None,),
            input_shape[0],
        ]

    def call(self, x, mask=None):
        """Call this layer with inputs.

        Args:
            x(list): The list of input tensors.
                x[0]: the input ROIs in shape (N, B, 4), absolute coordinates.
                x[1]: RCNN confidence in the shape (N, B, C+1),including the background.
                x[2]: RCNN deltas in the shape (N, B, C*4), for valid classes.
                x[3]: Input image for clipping boxes.
        Returns:
            the output tensor of the layer.
        """
        assert(len(x) == 4)
        # (N, B, 4) to (N, B, 1, 4) for ease of broadcasting
        rois = tf.expand_dims(x[0], axis=2)
        # strip the groundtruth class in confidence
        rcnn_conf = x[1]
        rcnn_conf_valid = rcnn_conf[:, :, :-1]
        # (N, B, C*4) to (N, B, C, 4)
        rcnn_deltas = x[2]
        rcnn_deltas = tf.reshape(
            rcnn_deltas,
            (tf.shape(rcnn_deltas)[0], tf.shape(rcnn_deltas)[1], -1, 4)
        )
        input_image = x[3]
        image_h = tf.cast(tf.shape(input_image)[2], tf.float32)
        image_w = tf.cast(tf.shape(input_image)[3], tf.float32)
        # apply deltas to RoIs
        y1 = rois[:, :, :, 0]
        x1 = rois[:, :, :, 1]
        y2 = rois[:, :, :, 2]
        x2 = rois[:, :, :, 3]
        w0 = x2 - x1 + 1.0
        h0 = y2 - y1 + 1.0
        x0 = x1 + w0 / 2.0
        y0 = y1 + h0 / 2.0
        tx = rcnn_deltas[:, :, :, 0] / self.regr_std_scaling[0]
        ty = rcnn_deltas[:, :, :, 1] / self.regr_std_scaling[1]
        tw = rcnn_deltas[:, :, :, 2] / self.regr_std_scaling[2]
        th = rcnn_deltas[:, :, :, 3] / self.regr_std_scaling[3]
        cx = tx * w0 + x0
        cy = ty * h0 + y0
        ww = tf.exp(tw) * w0
        hh = tf.exp(th) * h0
        xx1 = cx - 0.5 * ww
        yy1 = cy - 0.5 * hh
        xx2 = cx + 0.5 * ww
        yy2 = cy + 0.5 * hh
        xx1 = tf.clip_by_value(xx1, 0.0, image_w-1.0)
        yy1 = tf.clip_by_value(yy1, 0.0, image_h-1.0)
        xx2 = tf.clip_by_value(xx2, 0.0, image_w-1.0)
        yy2 = tf.clip_by_value(yy2, 0.0, image_h-1.0)
        boxes = tf.stack([yy1, xx1, yy2, xx2], axis=-1)
        tf_nms = tf.image.combined_non_max_suppression
        # walk around of a bug for tf.image.combined_non_max_suppression
        # force it to run on CPU since GPU version is flaky
        with tf.device("cpu:0"):
            nmsed_boxes, nmsed_scores, nmsed_classes, num_dets = tf_nms(
                boxes,
                rcnn_conf_valid,
                self.max_box_num,
                self.max_box_num,
                self.iou_thres,
                self.score_thres,
                clip_boxes=False,
            )
        return [nmsed_boxes, nmsed_scores, nmsed_classes, num_dets, x[0]]

    def get_config(self):
        """Get config for this layer."""
        config = {
            'max_box_num': self.max_box_num,
            'regr_std_scaling': self.regr_std_scaling,
            'iou_thres': self.iou_thres,
            'score_thres': self.score_thres,
        }
        base_config = super(OutputParser, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
