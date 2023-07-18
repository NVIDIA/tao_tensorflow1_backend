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
'''Loss functions for FasterRCNN.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import safe_gather


def _smooth_l1_loss(bbox_pred, bbox_targets, sigma=1.0):
    """Smooth L1 loss function."""
    sigma_2 = sigma * sigma
    box_diff = bbox_pred - bbox_targets
    in_box_diff = box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(K.cast(K.less_equal(abs_in_box_diff, 1.0/sigma_2),
                                            tf.float32))
    x1 = (in_box_diff * in_box_diff) * (sigma_2 / 2.) * smoothL1_sign
    x2 = (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    in_loss_box = x1 + x2
    return in_loss_box


def _build_rpn_class_loss(num_anchors, lambda_rpn_class, rpn_train_bs):
    '''build RPN classification loss.'''
    def rpn_loss_cls(y_true, y_pred):
        y_true = tf.stop_gradient(y_true)
        ce_loss = K.binary_crossentropy(y_true[:, num_anchors:, :, :],
                                        y_pred[:, :, :, :])
        loss = lambda_rpn_class * \
            K.sum(y_true[:, :num_anchors, :, :] * ce_loss, axis=[1, 2, 3]) / rpn_train_bs
        return K.mean(loss)
    return rpn_loss_cls


def _build_rpn_bbox_loss(num_anchors, lambda_rpn_regr, rpn_train_bs):
    '''build RPN bbox loss.'''
    def rpn_loss_regr(y_true, y_pred):
        y_true = tf.stop_gradient(y_true)
        l1_loss = _smooth_l1_loss(y_pred,
                                  y_true[:, 4 * num_anchors:, :, :],
                                  sigma=3.0)
        loss = lambda_rpn_regr * \
            K.sum(y_true[:, :4 * num_anchors, :, :] * l1_loss, axis=[1, 2, 3]) / rpn_train_bs
        return K.mean(loss)
    return rpn_loss_regr


def _build_rcnn_class_loss(lambda_rcnn_class, rcnn_train_bs):
    '''build RCNN classification loss.'''
    def rcnn_loss_cls(y_true, y_pred):
        # y_true: (N, R, C+1), all zero label indicates padded numbers.
        # y_pred: (N, R, C+1)
        # mask for positive + negative ROIs, ignore padded ROIs.
        batch = tf.cast(tf.shape(y_pred)[0], tf.float32)
        y_true = tf.stop_gradient(y_true)
        y_true_mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)
        ce_loss = categorical_crossentropy(y_true, y_pred)
        loss = lambda_rcnn_class * \
            K.sum(ce_loss*tf.stop_gradient(y_true_mask)) / rcnn_train_bs
        # average over batch dim
        return loss / batch
    return rcnn_loss_cls


def _build_rcnn_bbox_loss(num_classes, lambda_rcnn_regr, rcnn_train_bs):
    '''build RCNN bbox loss.'''
    def rcnn_loss_regr(y_true, y_pred):
        # y_true: (N, R, C8)
        # y_pred: (N, R, C4)
        batch = tf.cast(tf.shape(y_pred)[0], tf.float32)
        y_true = tf.stop_gradient(y_true)
        y_true = tf.reshape(y_true,
                            (tf.shape(y_true)[0], tf.shape(y_true)[1], num_classes-1, 8))
        y_true_positive = tf.reshape(y_true[:, :, :, 0:4], (-1, 4))
        y_true_deltas = tf.reshape(y_true[:, :, :, 4:], (-1, 4))
        y_pred = tf.reshape(y_pred, (-1, 4))
        y_true_pos_sel = tf.math.equal(tf.reduce_sum(y_true_positive, axis=1), 4.0)
        positive_idxs = tf.where(y_true_pos_sel)[:, 0]
        l1_loss = _smooth_l1_loss(safe_gather(y_pred, positive_idxs),
                                  safe_gather(y_true_deltas, positive_idxs))
        loss = K.switch(tf.size(positive_idxs) > 0,
                        l1_loss,
                        tf.constant(0.0))
        loss = lambda_rcnn_regr * K.sum(loss) / rcnn_train_bs
        # average over batch dim
        return loss / batch
    return rcnn_loss_regr
