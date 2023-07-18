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

# Replicating this from Yolov3 findings.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from nvidia_tao_tf1.cv.yolo_v4.losses.yolo_loss import YOLOv4Loss


def convert_pred_to_true_match(y_pred):
    # "construct" GT that matches pred
    y_true_02 = y_pred[:, :, 6:8] * y_pred[:, :, 4:6] + y_pred[:, :, 0:2]
    y_true_24 = y_pred[:, :, 8:10] * y_pred[:, :, 2:4]
    y_true_45 = tf.sigmoid(y_pred[:, :, 10:11])
    y_true_56 = 1.0 - tf.sigmoid(y_pred[:, :, 10:11])
    y_true_6_ = tf.sigmoid(y_pred[:, :, 11:])
    y_true_last = tf.ones_like(y_pred[:, :, 0:1])

    # return constructed GT
    return tf.concat([y_true_02, y_true_24, y_true_45, y_true_56, y_true_6_, y_true_last], -1)


def test_loss_zero():
    # let's give a large coef on loss
    yolo_loss = YOLOv4Loss(10, 1000, 10)
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_x():
    yolo_loss = YOLOv4Loss(10, 1000, 10)
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_x
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.001, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])

    # eliminate coord loss
    yolo_loss1 = YOLOv4Loss(0, 1000, 10)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 0.7832673) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_y():
    yolo_loss = YOLOv4Loss(10, 1000, 10)
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_y
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 0.5,
                            0.01, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])

    # eliminate coord loss
    yolo_loss1 = YOLOv4Loss(0, 1000, 10)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 25.969965) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_wh():
    yolo_loss = YOLOv4Loss(10, 1000, 10)
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_wh
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 2, 5, 1e99, 1e99, -1e99, -1e99]]])

    # eliminate coord loss
    yolo_loss1 = YOLOv4Loss(0, 1000, 10)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 7.7667146) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_obj():
    yolo_loss = YOLOv4Loss(10, 1000, 10)
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_obj
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, -1e99, 1e99, -1e99, -1e99]]])

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 10.361637115478516) < 1e-5


def test_loss_nonzero_cls():
    yolo_loss = YOLOv4Loss(10, 1000, 10)
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_cls
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, 1e99, -1e99, 1e99, -1e99]]])

    # eliminate cls loss
    yolo_loss1 = YOLOv4Loss(10, 1000, 0)

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 414.46533203125) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5


def test_loss_nonzero_noobj():
    yolo_loss = YOLOv4Loss(10, 1, 10)
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, -1e99, 1e99, -1e99, -1e99]]])
    y_true = convert_pred_to_true_match(y_pred)

    # turbulate pred_y
    y_pred = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 1.0,
                            0.01, 1.35, 4.5, 1e99, 1e99, -1e99, -1e99]]])

    # eliminate noobj loss
    yolo_loss1 = YOLOv4Loss(10, 0, 10)

    # turbulate everything other than obj
    y_pred1 = tf.constant([[[0.1, 0.3, 0.05, 0.07, 0.5, 0.7, 0.1,
                             0.7, 1.5, 0.1, -1e99, 0, 0, 0]]])

    with tf.Session() as sess:
        assert abs(sess.run(yolo_loss.compute_loss(y_true, y_pred)) - 31.08489990234375) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred))) < 1e-5
        assert abs(sess.run(yolo_loss1.compute_loss(y_true, y_pred1))) < 1e-5
