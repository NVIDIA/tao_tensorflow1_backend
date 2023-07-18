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
'''Unit test for FasterRCNN loss functions.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.faster_rcnn.losses.losses import (
    _build_rcnn_bbox_loss,
    _build_rcnn_class_loss,
    _build_rpn_bbox_loss,
    _build_rpn_class_loss
)


NUM_ANCHORS = 9
LAMBDA_RPN_CLASS = 1.0
LAMBDA_RPN_DELTAS = 1.0
LAMBDA_RCNN_CLASS = 1.0
LAMBDA_RCNN_DELTAS = 1.0
RPN_TRAIN_BS = 256
RCNN_TRAIN_BS = 256
BS = 2
NUM_CLASSES = 4
RPN_H = 20
RPN_W = 30


def test_rpn_class_loss():
    '''Check the RPN classification loss.'''
    # loss should be non-negative
    rpn_class_loss = _build_rpn_class_loss(NUM_ANCHORS, LAMBDA_RPN_CLASS, RPN_TRAIN_BS)
    shape = (BS, NUM_ANCHORS, RPN_H, RPN_W)
    shape2 = (BS, 2*NUM_ANCHORS, RPN_H, RPN_W)
    y_pred = tf.constant(np.random.random(shape), dtype=tf.float32)
    y_true_np = np.random.randint(0,
                                  high=2,
                                  size=(BS, 1, NUM_ANCHORS, RPN_H, RPN_W)).astype(np.float32)
    y_true_np = np.broadcast_to(y_true_np, (BS, 2, NUM_ANCHORS, RPN_H, RPN_W)).reshape(shape2)
    y_true = tf.constant(y_true_np,
                         dtype=tf.float32)
    loss = rpn_class_loss(y_true, y_pred)
    with tf.Session() as sess:
        assert sess.run(loss) >= 0.0

    # if providing inputs with 3 dims, it should raise error.
    y_true_wrong = \
        tf.constant(np.random.randint(0,
                                      high=2,
                                      size=(2*NUM_ANCHORS, RPN_H, RPN_W)).astype(np.float32),
                    dtype=tf.float32)
    with pytest.raises(ValueError):
        rpn_class_loss(y_true_wrong, y_pred)


def test_rpn_bbox_loss():
    '''Check the RPN boundingbox loss.'''
    # loss should be non-negative.
    rpn_bbox_loss = _build_rpn_bbox_loss(NUM_ANCHORS, LAMBDA_RPN_DELTAS, RPN_TRAIN_BS)
    shape = (BS, 4*NUM_ANCHORS, RPN_H, RPN_W)
    y_pred = tf.constant(np.random.random(shape), dtype=tf.float32)
    y_true_np = np.broadcast_to(np.random.randint(0,
                                                  high=2,
                                                  size=(BS, 1, NUM_ANCHORS, RPN_H, RPN_W)),
                                (BS, 4, NUM_ANCHORS, RPN_H, RPN_W)).reshape(shape)
    y_true_mask = tf.constant(y_true_np, dtype=tf.float32)
    y_true_deltas = tf.constant(np.random.random(shape), dtype=tf.float32)
    y_true = tf.concat((y_true_mask, y_true_deltas), axis=1)
    loss = rpn_bbox_loss(y_true, y_pred)
    with tf.Session() as sess:
        assert sess.run(loss) >= 0.0

    y_true_wrong = y_true[0, ...]
    with pytest.raises(ValueError):
        rpn_bbox_loss(y_true_wrong, y_pred)


def test_rcnn_class_loss():
    '''Check the RCNN classification loss.'''
    # loss should be non-negative.
    rcnn_class_loss = _build_rcnn_class_loss(LAMBDA_RCNN_CLASS, RCNN_TRAIN_BS)
    shape = (BS, RCNN_TRAIN_BS, NUM_CLASSES)
    y_pred = tf.constant(np.random.random(shape), dtype=tf.float32)
    y_true = tf.constant(np.random.randint(0, high=2, size=shape), dtype=tf.float32)
    loss = rcnn_class_loss(y_true, y_pred)
    with tf.Session() as sess:
        assert sess.run(loss) >= 0.0

    # raise error when with wrong input dims
    y_true_error = y_true[..., 0:2]
    with pytest.raises(ValueError):
        rcnn_class_loss(y_true_error, y_pred)


def test_rcnn_bbox_loss():
    '''Check the RCNN bbox loss.'''
    # loss should be non-negative.
    rcnn_bbox_loss = _build_rcnn_bbox_loss(NUM_CLASSES, LAMBDA_RCNN_DELTAS, RCNN_TRAIN_BS)
    shape = (BS, RCNN_TRAIN_BS, (NUM_CLASSES-1)*4)
    shape2 = (BS, RCNN_TRAIN_BS, (NUM_CLASSES-1)*8)
    y_pred = tf.constant(np.random.random(shape), dtype=tf.float32)
    y_true_np = np.broadcast_to(np.random.randint(0,
                                                  high=2,
                                                  size=(BS, RCNN_TRAIN_BS, NUM_CLASSES-1, 1)),
                                (BS, RCNN_TRAIN_BS, (NUM_CLASSES-1), 4))
    y_true_mask = tf.constant(y_true_np, dtype=tf.float32)
    y_true_deltas_np = np.random.random((BS, RCNN_TRAIN_BS, (NUM_CLASSES-1), 4))
    y_true_deltas = tf.constant(y_true_deltas_np, dtype=tf.float32)
    y_true = tf.reshape(tf.concat((y_true_mask, y_true_deltas), axis=-1), shape2)
    loss = rcnn_bbox_loss(y_true, y_pred)
    with tf.Session() as sess:
        assert sess.run(loss) >= 0.0

    y_true_wrong = y_true[0, ...]
    with pytest.raises(ValueError):
        rcnn_bbox_loss(y_true_wrong, y_pred)
