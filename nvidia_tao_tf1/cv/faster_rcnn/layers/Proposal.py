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
"""Proposal layer in FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import (
    apply_deltas_to_anchors,
    batch_op,
    clip_bbox_tf,
    make_anchors,
    nms_tf
)


class Proposal(Layer):
    '''Proposal layer to convert RPN output to RoIs.

    In FasterRCNN, Proposal layer is applied on top of RPN to convert the
    dense anchors into a smaller sparse subset of proposals(RoIs). This
    conversion roughly includes below steps:
    1. Apply deltas to anchors.
    2. Take pre NMS top N boxes.
    3. Do NMS against the dense boxes and finally take post NMS top N boxes.
    '''

    def __init__(self, anchor_sizes, anchor_ratios,  std_scaling,
                 rpn_stride, pre_nms_top_N, post_nms_top_N,
                 nms_iou_thres, activation_type, bs_per_gpu, **kwargs):
        '''Initialize the Proposal layer.

        Args:
            anchor_sizes(list): the list of anchor box sizes, at input image scale.
            anchor_ratios(list): the list of anchor box ratios.
            std_scaling(float): a constant to do scaling for the RPN deltas output.
            rpn_stride(int): the total stirde of RPN relative to input image, always
                16 in current implementation.
            pre_nms_top_N(int): the number of boxes to retain before doing NMS.
            post_nms_top_N(int): the number of boxes to retain after doing NMS.
            nms_iou_thres(float): the NMS IoU threshold.
            activation_type(str): the activation type for RPN confidence output. Currently
                only sigmoid is supported.
            bs_per_gpu(int): the batch size for each GPU.
        '''
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = [np.sqrt(ar) for ar in anchor_ratios]
        self.std_scaling = std_scaling
        self.rpn_stride = rpn_stride
        self.pre_nms_top_N = pre_nms_top_N
        self.post_nms_top_N = post_nms_top_N
        self.nms_iou_thres = nms_iou_thres
        self.activation_type = activation_type
        self.bs_per_gpu = bs_per_gpu
        super(Proposal, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        '''Compute the output shape.'''
        batch_size = input_shape[0][0]
        return tuple([batch_size, self.post_nms_top_N, 4])

    def _build_anchors_tf(self, rpn_h, rpn_w):
        """Build the anchors in tensorflow ops."""
        anc_x, anc_y = tf.meshgrid(
            tf.range(tf.cast(rpn_w, tf.float32)),
            tf.range(tf.cast(rpn_h, tf.float32))
        )
        # this is a simple numpy function to generate the base anchors
        ancs = tf.constant(
            make_anchors(self.anchor_sizes, self.anchor_ratios).reshape(-1, 2),
            dtype=tf.float32
        )
        anc_pos = self.rpn_stride*(tf.stack((anc_x, anc_y), axis=-1) + 0.5)
        anc_pos = tf.reshape(anc_pos, (rpn_h, rpn_w, 1, 2))
        anc_pos = tf.broadcast_to(anc_pos, (rpn_h, rpn_w, ancs.shape[0], 2))
        anc_left_top = anc_pos - ancs/2.0
        full_anc_xywh = tf.concat(
            (
                anc_left_top,
                tf.broadcast_to(ancs,
                                tf.shape(anc_left_top))
            ),
            axis=-1
        )
        # broadcast to batch dim: (H, W, A, 4) -> (N, H, W, A, 4)
        full_anc_xywh = tf.broadcast_to(
            full_anc_xywh,
            tf.concat([[self.bs_per_gpu], tf.shape(full_anc_xywh)], axis=-1)
        )
        return tf.reshape(full_anc_xywh, (-1, 4))

    def call(self, x, mask=None):
        """Call Proposal layer with RPN outputs as inputs.

        Args:
            x(list): the list of input tensors.
                x[0]: RPN confidence.
                x[1]: RPN deltas.
                x[2]: input image of the entire model, for clipping bboxes.
        Returns:
            The output bbox coordinates(RoIs).
        """
        rpn_scores = x[0]
        rpn_deltas = x[1] * (1.0 / self.std_scaling)
        input_image = x[2]
        # get dynamic shapes
        rpn_h = tf.shape(rpn_scores)[2]
        rpn_w = tf.shape(rpn_scores)[3]
        image_h = tf.cast(tf.shape(input_image)[2], tf.float32)
        image_w = tf.cast(tf.shape(input_image)[3], tf.float32)
        # RPN deltas: (N, A4, H, W) -> (N, H, W, A4) -> (-1, 4)
        rpn_deltas = tf.reshape(tf.transpose(rpn_deltas, perm=[0, 2, 3, 1]), (-1, 4))
        # Anchors: (N, H, W, A, 4) -> (-1, 4)
        full_anc_tf = self._build_anchors_tf(rpn_h, rpn_w)
        # for testing, remember it then we can retrieve it in testing.
        self.full_anc_tf = full_anc_tf
        self.rpn_deltas = rpn_deltas
        all_boxes = apply_deltas_to_anchors(full_anc_tf, rpn_deltas)
        num_ancs = len(self.anchor_sizes) * len(self.anchor_ratios)
        NHWA4 = (self.bs_per_gpu, rpn_h, rpn_w, num_ancs, 4)
        # (N, H, W, A, 4) -> (NAHW, 4)
        all_boxes = tf.reshape(tf.transpose(tf.reshape(all_boxes, NHWA4), (0, 3, 1, 2, 4)), (-1, 4))
        # for testing
        self.all_boxes_tf = all_boxes
        all_boxes = clip_bbox_tf(all_boxes, image_w, image_h)
        # (N, AHW, 4)
        all_boxes = tf.reshape(all_boxes, (self.bs_per_gpu, -1, 4))
        # (N, A, H, W) -> (N, AHW)
        all_probs = tf.reshape(rpn_scores, (self.bs_per_gpu, -1))
        # NMS for each image
        result = batch_op([all_boxes, all_probs],
                          lambda x: nms_tf(*x,
                                           pre_nms_top_N=self.pre_nms_top_N,
                                           post_nms_top_N=self.post_nms_top_N,
                                           nms_iou_thres=self.nms_iou_thres),
                          self.bs_per_gpu)
        return result

    def get_config(self):
        """Get config for this layer."""
        config = {'anchor_sizes': self.anchor_sizes,
                  'anchor_ratios': self.anchor_ratios,
                  'std_scaling': self.std_scaling,
                  'rpn_stride': self.rpn_stride,
                  'pre_nms_top_N': self.pre_nms_top_N,
                  'post_nms_top_N': self.post_nms_top_N,
                  'nms_iou_thres': self.nms_iou_thres,
                  'activation_type': self.activation_type,
                  'bs_per_gpu': self.bs_per_gpu}
        base_config = super(Proposal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
