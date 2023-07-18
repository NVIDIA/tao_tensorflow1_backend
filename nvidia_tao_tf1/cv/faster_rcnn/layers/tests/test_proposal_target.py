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
"""Unit tests for FasterRCNN proposal target layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import keras.backend as K
import mock
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import (
    calc_iou_np,
    generate_proposal_target_v1,
    sample_proposals,
    unpad_np
)
from nvidia_tao_tf1.cv.faster_rcnn.tests.utils import _fake_choice, _fake_uniform


class TestProposalTarget(unittest.TestCase):
    '''Main class that checks the proposal_target generation graph.'''

    def init(self):
        '''Initialize the necessary data.'''
        self.rois = tf.placeholder(shape=(None, None, 4), dtype=tf.float32, name='rois')
        self.gt_boxes = tf.placeholder(shape=(None, None, 4), dtype=tf.float32, name='gt_boxes')
        self.gt_class_ids = tf.placeholder(shape=(None, None),
                                           dtype=tf.float32,
                                           name='gt_class_ids')
        self.iou_high_thres = 0.5
        self.iou_low_thres = 0.0
        self.roi_train_bs = 256
        self.roi_positive_ratio = 0.25
        self.deltas_scaling = [10., 10., 5., 5.]
        self.bg_class_id = 3
        self.image_w = 1248
        self.image_h = 384

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))
        self.session = K.get_session()

    def compute_proposal_target_tf(self):
        '''construct the proposal target graph.'''
        self.rois_tf, self.class_ids_tf, self.deltas_tf, _ = \
            generate_proposal_target_v1(self.rois,
                                        self.gt_boxes,
                                        self.gt_class_ids,
                                        self.iou_high_thres,
                                        self.iou_low_thres,
                                        self.roi_train_bs,
                                        self.roi_positive_ratio,
                                        np.array(self.deltas_scaling),
                                        self.bg_class_id)

    def proposal_target_np(self, gt_bboxes, gt_class_ids, rois):
        '''compute proposal target in numpy.'''
        rois, _ = unpad_np(rois)
        gt_bboxes, nz = unpad_np(gt_bboxes)
        gt_class_ids = gt_class_ids[nz]
        rois_np, cls_np, deltas_np = calc_iou_np(rois, gt_bboxes, gt_class_ids,
                                                 self.iou_high_thres, self.iou_low_thres,
                                                 self.deltas_scaling,
                                                 num_classes=4)
        roi_idxs_np = sample_proposals(cls_np, self.roi_train_bs, self.roi_positive_ratio)
        return rois_np[roi_idxs_np, ...], cls_np[roi_idxs_np, ...], deltas_np[roi_idxs_np, ...]

    def proposal_target_tf(self, gt_bboxes, gt_class_ids, rois):
        '''compute proposal target in tf.'''
        rois_tf, cls_tf, deltas_tf = self.session.run([self.rois_tf,
                                                       self.class_ids_tf,
                                                       self.deltas_tf],
                                                      feed_dict={'rois:0': rois,
                                                                 'gt_boxes:0': gt_bboxes,
                                                                 'gt_class_ids: 0': gt_class_ids})
        return rois_tf, cls_tf, deltas_tf

    def gen_test_vectors(self):
        '''generate test vectors.'''
        gt_boxes = np.array([[[726, 148, 826, 319],
                             [389, 185, 425, 207],
                             [679, 167, 692, 198]]], dtype=np.float32)
        gt_cls = np.array([[1, 0, 2]], dtype=np.int32)
        rois = np.array([[[730, 150, 830, 320],
                         [390, 190, 430, 210],
                         [700, 200, 750, 250]]], dtype=np.float32)
        # convert to y1, x1, y2, x2 format
        gt_boxes = np.copy(gt_boxes[:, :, (1, 0, 3, 2)])
        rois = np.copy(rois[:, :, (1, 0, 3, 2)])
        # pad gt_boxes to 100
        gt_boxes_ = gt_boxes.shape[1]
        gt_boxes = np.concatenate([gt_boxes, np.zeros(shape=(1, 100-gt_boxes_, 4))], axis=1)
        gt_cls_ = gt_cls.shape[1]
        gt_cls = np.concatenate([gt_cls, -1 * np.ones(shape=(1, 100-gt_cls_))], axis=1)
        rois_ = rois.shape[1]
        rois = np.concatenate([rois, np.zeros(shape=(1, 300-rois_, 4))], axis=1)
        return gt_boxes, gt_cls, rois

    def test_proposal_target(self):
        '''check the output of tensorflow and numpy.'''
        self.init()
        with mock.patch('tensorflow.random.uniform', side_effect=_fake_uniform) \
                as uniform_function:  # noqa pylint: disable=F841, W0612
            with mock.patch('tensorflow.random_shuffle', side_effect=tf.identity) \
                    as random_shuffle_function:  # noqa pylint: disable=F841, W0612
                self.compute_proposal_target_tf()
        gt_bboxes, gt_class_ids, rois = self.gen_test_vectors()
        with mock.patch('numpy.random.choice', side_effect=_fake_choice) \
            as choice_function:  # noqa pylint: disable=F841, W0612
            rois_np, cls_np, deltas_np = \
                self.proposal_target_np(gt_bboxes[0, ...], # noqa pylint: disable=E1126
                                        gt_class_ids[0, ...], # noqa pylint: disable=E1126
                                        rois[0, ...]) # noqa pylint: disable=E1126
        rois_tf, cls_tf, deltas_tf = self.proposal_target_tf(gt_bboxes, gt_class_ids, rois)
        # convert tf RoIs to (x1, y1, x2, y2) format
        rois_tf = rois_tf[:, (1, 0, 3, 2)]
        deltas_tf = np.reshape(deltas_tf, (deltas_tf.shape[0], -1, 8))
        assert np.allclose(rois_np, rois_tf)
        assert np.allclose(cls_np, cls_tf)
        assert np.allclose(deltas_np, deltas_tf)
