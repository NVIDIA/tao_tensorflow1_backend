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
"""Unit tests for FasterRCNN proposal layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import keras
import keras.backend as K
import mock
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.faster_rcnn.layers.custom_layers import Proposal
from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import (
    compute_rpn_target_np, nms_core_py_func,
    rpn_to_roi
)


class TestProposal(unittest.TestCase):
    '''Main class for testing the proposal layer.'''

    def init(self):
        '''Initialize.'''
        self.anchor_sizes = [64.0, 128.0, 256.0]
        self.anchor_ratios = [1.0, 0.5, 2.0]
        self.std_scaling = 1.0
        self.rpn_stride = 16.0
        self.pre_nms_top_N = 12000
        self.post_nms_top_N = 2000
        self.nms_iou_thres = 0.7
        self.activation_type = 'sigmoid'
        self.image_h = 384
        self.image_w = 1280
        self.rpn_h = self.image_h // 16
        self.rpn_w = self.image_w // 16
        self.bs_per_gpu = 1
        self.num_anchors = len(self.anchor_sizes) * len(self.anchor_ratios)
        self.iou_high_thres = 0.5
        self.iou_low_thres = 0.0
        self.rpn_train_bs = 256
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))
        self.session = K.get_session()

    def build_proposal_graph(self):
        '''Build the model with only a proposal layer.'''
        input_scores = keras.layers.Input(shape=(self.num_anchors,
                                                 self.rpn_h, self.rpn_w),
                                          name='rpn_scores')
        input_deltas = keras.layers.Input(shape=(4*self.num_anchors,
                                                 self.rpn_h, self.rpn_w),
                                          name='rpn_deltas')
        input_image = keras.layers.Input(
            shape=(3, self.image_h, self.image_w),
            name="input_image"
        )
        proposal_out = Proposal(self.anchor_sizes,
                                self.anchor_ratios,
                                self.std_scaling,
                                self.rpn_stride,
                                self.pre_nms_top_N,
                                self.post_nms_top_N,
                                self.nms_iou_thres,
                                self.activation_type,
                                self.bs_per_gpu)(
                                    [input_scores, input_deltas, input_image]
                                )
        if not isinstance(proposal_out, list):
            proposal_out = [proposal_out]
        self.model = keras.models.Model(inputs=[input_scores, input_deltas, input_image],
                                        outputs=proposal_out)

    def generate_test_vectors(self):
        '''generate the RPN scores and deltas as test vectors.'''
        gt_boxes = np.array([[141., 1244.,  382., 1278.],
                             [151.,  849.,  305.,  896.],
                             [163., 1191.,  326., 1270.],
                             [175.,  133.,  284.,  180.],
                             [170.,  213.,  288.,  255.],
                             [208.,   86.,  261.,  140.],
                             [170.,  458.,  235.,  497.],
                             [158.,  534.,  215.,  607.],
                             [156.,  856.,  282.,  901.],
                             [176.,  608.,  219., 652.]], dtype=np.float32)
        ar_sqrt = [np.sqrt(ar) for ar in self.anchor_ratios]
        rpn_scores, rpn_deltas = compute_rpn_target_np(gt_boxes, self.anchor_sizes,
                                                       ar_sqrt, self.rpn_stride,
                                                       self.rpn_h, self.rpn_w,
                                                       self.image_h, self.image_w,
                                                       self.rpn_train_bs, self.iou_high_thres,
                                                       self.iou_low_thres)
        images = np.zeros((1, 3, self.image_h, self.image_w), dtype=np.float32)
        return (rpn_scores[:, self.num_anchors:, :, :],
                rpn_deltas[:, self.num_anchors*4:, :, :],
                images)

    def proposal_np(self, scores, deltas):
        '''Proposal layer in numpy.'''
        return rpn_to_roi(scores, deltas,
                          self.image_w, self.image_h,
                          self.anchor_sizes, self.anchor_ratios,
                          self.std_scaling, self.rpn_stride,
                          use_regr=True, max_boxes=self.post_nms_top_N,
                          overlap_thresh=self.nms_iou_thres,
                          rpn_pre_nms_top_N=self.pre_nms_top_N)

    def test_proposal_layer(self):
        '''Check the proposal layer output.'''
        self.init()
        # monkey patch the tf.image.non_max_suppression since we found
        # its result cannot match the numpy NMS exactly anyway.
        with mock.patch('tensorflow.image.non_max_suppression', side_effect=nms_core_py_func) \
                as _non_max_suppression_function:  # noqa pylint: disable=F841, W0612
            self.build_proposal_graph()
        scores, deltas, images = self.generate_test_vectors()
        proposal_out_keras = self.model.predict([scores, deltas, images])
        proposal_out_np = self.proposal_np(scores, deltas)
        assert np.allclose(proposal_out_keras, proposal_out_np[:, :, (1, 0, 3, 2)], atol=1e-4)
