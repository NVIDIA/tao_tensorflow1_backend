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
"""test yolo loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

# Forcing the default GPU to be index 0
# because TensorFlow tries to set idx to 1
# with an XLA error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from nvidia_tao_tf1.cv.yolo_v3.losses.yolo_loss import YOLOv3Loss


def convert_pred_to_true_match(y_pred):
    # "construct" GT that matches pred
    y_true_02 = tf.sigmoid(y_pred[:, :, 6:8]) * y_pred[:, :, 4:6] + y_pred[:, :, 0:2]
    y_true_24 = tf.exp(y_pred[:, :, 8:10]) * y_pred[:, :, 2:4]
    y_true_45 = tf.sigmoid(y_pred[:, :, 10:11])
    y_true_6_ = tf.sigmoid(y_pred[:, :, 11:])

    # return constructed GT
    return tf.concat([y_true_02, y_true_24, y_true_45, y_true_6_], -1)


def test_loss_zero():
    # let's give a large coef on loss
    yolo_loss = YOLOv3Loss(10, 1000, 10, 1.)
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_x():
    yolo_loss = YOLOv3Loss(10, 1000, 10, 1.)
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_x
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -6, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])

    # eliminate coord loss
    yolo_loss1 = YOLOv3Loss(0, 1000, 10, 1.)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) + 0.0025146198) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_y():
    yolo_loss = YOLOv3Loss(10, 1000, 10, 1.)
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_y
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1.0,
                            -5.0, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])

    # eliminate coord loss
    yolo_loss1 = YOLOv3Loss(0, 1000, 10, 1.)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) + 10.215902) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_wh():
    yolo_loss = YOLOv3Loss(10, 1000, 10, 1.)
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_wh
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.9, 1.8, 1e99, 1e99, -1e99, -1e99]]])

    # eliminate coord loss
    yolo_loss1 = YOLOv3Loss(0, 1000, 10, 1.)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) + 63.55852) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_obj():
    yolo_loss = YOLOv3Loss(10, 1000, 10, 1.)
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_obj
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, -1e99, 1e99, -1e99, -1e99]]])

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 41.446533) < 1e-5


def test_loss_nonzero_cls():
    yolo_loss = YOLOv3Loss(10, 1000, 10, 1.)
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_cls
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, 1e99, -1e99, 1e99, -1e99]]])

    # eliminate cls loss
    yolo_loss1 = YOLOv3Loss(10, 1000, 0, 1.)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 828.9306640625) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_noobj():
    yolo_loss = YOLOv3Loss(10, 1, 10, 1.)
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, -1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_y
    y_pred = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1e10,
                            -5.0, 0.3, 1.5, 1e99, 1e99, -1e99, -1e99]]])

    # eliminate noobj loss
    yolo_loss1 = YOLOv3Loss(10, 0, 10, 1.)

    # turbulate everything other than obj
    y_pred1 = tf.constant([[[100.0, 130.0, 2.0, 2.5, 3.0, 3.5, 1,
                             5e10, 1.5, 0.1, -1e99, 0, 0, 0]]])

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 41.446533) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred1))) < 1e-5
