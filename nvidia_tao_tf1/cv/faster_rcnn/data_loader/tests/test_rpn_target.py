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

'''Unit test for RPN target generator.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import keras.backend as K
import mock
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.faster_rcnn.data_loader.inputs_loader import RPNTargetGenerator
from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import _compute_rpn_target_np
from nvidia_tao_tf1.cv.faster_rcnn.tests.utils import _take_first_k
from nvidia_tao_tf1.cv.faster_rcnn.utils.utils import get_init_ops


np.random.seed(42)
tf.set_random_seed(42)


class TestRPNTarget(unittest.TestCase):
    '''Main class for testing the RPN target tensor generator.'''

    def init(self):
        '''Initialize.'''
        self.image_w = 1280
        self.image_h = 384
        self.rpn_w = self.image_w // 16
        self.rpn_h = self.image_h // 16
        self.rpn_stride = 16
        self.anchor_sizes = [64., 128., 256.]
        self.anchor_ratios = [1., .5, 2.]
        self.num_anchors = len(self.anchor_sizes) * len(self.anchor_ratios)
        self.bs_per_gpu = 1
        self.iou_high_thres = 0.7
        self.iou_low_thres = 0.3
        self.rpn_train_bs = 256
        self.max_objs_per_image = 100
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))
        self.session = K.get_session()
        self.rpn_target_gen = RPNTargetGenerator(self.image_w,
                                                 self.image_h,
                                                 self.rpn_w,
                                                 self.rpn_h,
                                                 self.rpn_stride,
                                                 self.anchor_sizes,
                                                 self.anchor_ratios,
                                                 self.bs_per_gpu,
                                                 self.iou_high_thres,
                                                 self.iou_low_thres,
                                                 self.rpn_train_bs,
                                                 self.max_objs_per_image)
        self.input_bboxes = tf.placeholder(tf.float32, shape=(None, 4), name='input_bboxes')

    def compute_rpn_target_tf(self):
        '''compute RPN target via tf Ops.'''
        (self.rpn_target_tf_class, self.rpn_target_tf_bbox,
         self.ious_tf, self.pos_idx, self.best_anc, self.anc_tf,
         self.unmapped_box_tf, self.unmapped_anc_tf,
         self.y_is_bbox_valid_save_tf, self.y_rpn_overlap_save_tf) = \
            self.rpn_target_gen._rpn_target_tf(self.input_bboxes)

    def get_boxes(self):
        """generate boxes as test vectors."""
        boxes = np.array(
            [
                [599.41, 156.40, 629.75, 189.25],
                [387.63, 181.54, 423.81, 203.12],
                [676.60, 163.95, 688.98, 193.93],
                [503.89, 169.71, 590.61, 190.13],
                [511.35, 174.96, 527.81, 187.45],
                [532.37, 176.35, 542.68, 185.27],
                [559.62, 175.83, 575.40, 183.15]
            ]
        )
        num_boxes = boxes.shape[0]
        all_boxes = np.pad(boxes, ((0, self.max_objs_per_image - num_boxes), (0, 0)))
        return np.ascontiguousarray(all_boxes.astype(np.float32))

    def test_rpn_target(self):
        '''Compare the outputs from tf and numpy.'''
        self.init()
        # build tf graph
        with mock.patch('tensorflow.random_shuffle', side_effect=tf.identity) \
                as _:  # noqa pylint: disable = W0612
            self.compute_rpn_target_tf()
        self.session.run(get_init_ops())
        # get a batch with GT
        gt_bbox = self.get_boxes()
        # compuate in numpy
        with mock.patch('numpy.random.choice', side_effect=_take_first_k) \
                as _:  # noqa pylint: disable=F841, W0612
            (rpn_class_np, rpn_bbox_np, ious_np, pos_idx_np, best_anc_np, anc_np,
                unmapped_box_np, unmapped_anc_np, y_is_bbox_valid_save_np,
                y_rpn_overlap_save_np) = \
                _compute_rpn_target_np(
                    gt_bbox,
                    self.anchor_sizes,
                    self.rpn_target_gen.anchor_ratios,
                    self.rpn_stride,
                    self.rpn_h,
                    self.rpn_w,
                    self.image_h,
                    self.image_w,
                    self.rpn_train_bs,
                    self.iou_high_thres,
                    self.iou_low_thres
                )
        # compute in tf
        (rpn_class_tf, rpn_bbox_tf, ious_tf, pos_idx_tf, best_anc_tf, anc_tf,
            unmapped_box_tf, unmapped_anc_tf, y_is_bbox_valid_save_tf,
            y_rpn_overlap_save_tf) = \
            self.session.run(
                [
                    self.rpn_target_tf_class,
                    self.rpn_target_tf_bbox,
                    self.ious_tf,
                    self.pos_idx,
                    self.best_anc,
                    self.anc_tf,
                    self.unmapped_box_tf,
                    self.unmapped_anc_tf,
                    self.y_is_bbox_valid_save_tf,
                    self.y_rpn_overlap_save_tf
                ],
                feed_dict={'input_bboxes:0': gt_bbox}
            )
        # # check
        assert np.equal(anc_np, anc_tf[:, (1, 0, 3, 2)]).all()
        assert np.equal(ious_np, ious_tf).all()
        assert np.equal(pos_idx_tf, pos_idx_np).all()
        assert np.equal(best_anc_tf, best_anc_np).all(), \
            print(np.amax(ious_tf, axis=0), np.amax(ious_np, axis=0))
        assert np.equal(unmapped_box_tf, unmapped_box_np).all()
        assert np.equal(unmapped_anc_tf, unmapped_anc_np).all()
        bbox_valid_diff = np.where(
            y_is_bbox_valid_save_np.reshape(-1) -
            y_is_bbox_valid_save_tf.reshape(-1))[0]
        assert np.equal(y_is_bbox_valid_save_np, y_is_bbox_valid_save_tf).all(), \
            print('bbox_valid_diff: {}, {}'.format(
                y_is_bbox_valid_save_np.reshape(-1)[bbox_valid_diff],
                y_is_bbox_valid_save_tf.reshape(-1)[bbox_valid_diff]))
        assert np.equal(y_rpn_overlap_save_np, y_rpn_overlap_save_tf).all()
        rpn_class_np_0 = rpn_class_np[0, 0:self.num_anchors, :, :]
        rpn_class_tf_0 = rpn_class_tf[0:self.num_anchors, :, :]
        class_diff_0 = np.where(rpn_class_np_0.reshape(-1)-rpn_class_tf_0.reshape(-1))[0]
        assert np.allclose(rpn_class_np_0, rpn_class_tf_0, atol=1e-6), \
            print('class first half diff: {}, {}'.format(
                    rpn_class_np_0.reshape(-1)[class_diff_0],
                    rpn_class_tf_0.reshape(-1)[class_diff_0]))
        assert np.equal(rpn_class_np[0, ...], rpn_class_tf).all()
        assert np.logical_or(np.equal(rpn_class_np, 0.), np.equal(rpn_class_np, 1.0)).all()
        assert np.logical_or(np.equal(rpn_class_tf, 0.), np.equal(rpn_class_tf, 1.0)).all()
        assert np.logical_or(np.equal(rpn_bbox_np[0, 0:self.num_anchors*4, :, :], 0),
                             np.equal(rpn_bbox_np[0, 0:self.num_anchors*4, :, :], 1)).all()
        assert np.logical_or(np.equal(rpn_bbox_tf[:self.num_anchors*4, :, :], 0),
                             np.equal(rpn_bbox_tf[:self.num_anchors*4, :, :], 1)).all()
        assert np.equal(rpn_bbox_tf[0:self.num_anchors*4, :, :],
                        rpn_bbox_np[0, 0:self.num_anchors*4, :, :]).all()
        assert np.allclose(rpn_bbox_np[0, ...], rpn_bbox_tf, atol=1e-6)
