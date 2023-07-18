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
"""test ssd loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.cv.ssd.architecture.ssd_loss import SSDLoss


def test_loss_zero():
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    y_true = [[[1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.2]]]
    y_pred = y_true
    with tf.Session() as sess:
        assert abs(sess.run(ssd_loss.compute_loss(tf.constant(y_true),
                                                  tf.constant(y_pred)))[0]) < 1e-5


def test_loss_non_zero_loc():
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    y_true = [[[1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.2]]]
    y_pred = [[[1, 0, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.2]]]
    with tf.Session() as sess:
        log_loss = sess.run(ssd_loss.log_loss(tf.constant(y_true)[:, :, :-12],
                                              tf.constant(y_pred)[:, :, :-12]))
        loc_loss = sess.run(ssd_loss.smooth_L1_loss(tf.constant(y_true)[:, :, -12:-8],
                                                    tf.constant(y_pred)[:, :, -12:-8]))
        total_loss = sess.run(ssd_loss.compute_loss(tf.constant(y_true), tf.constant(y_pred)))

        assert abs(log_loss[0]) < 1e-5
        assert abs(loc_loss[0] - 0.00125) < 1e-5
        assert abs(total_loss[0]) < 1e-5


def test_loss_non_zero():
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    y_true = [[[1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.2]]]
    y_pred = [[[0.3, 0, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.2]]]
    with tf.Session() as sess:
        log_loss = sess.run(ssd_loss.log_loss(tf.constant(y_true)[:, :, :-12],
                                              tf.constant(y_pred)[:, :, :-12]))
        loc_loss = sess.run(ssd_loss.smooth_L1_loss(tf.constant(y_true)[:, :, -12:-8],
                                                    tf.constant(y_pred)[:, :, -12:-8]))
        total_loss = sess.run(ssd_loss.compute_loss(tf.constant(y_true), tf.constant(y_pred)))

        assert abs(log_loss[0] - 1.2039728) < 1e-5
        assert abs(loc_loss[0] - 0.00125) < 1e-5
        assert abs(total_loss[0]) < 1e-5
