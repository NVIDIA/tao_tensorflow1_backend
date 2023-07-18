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
"""Tests for FpeNet loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf
from nvidia_tao_tf1.cv.fpenet.losses.fpenet_loss import FpeLoss


def test_fpenet_loss_call():
    """
    Test call function for loss computation and exception on unsupported loss_type.
    """
    # create dummy data
    y_true = tf.ones((4, 80, 2))  # batch_size, num_keypoints, num_dim (x, y)
    y_pred = tf.zeros((4, 80, 2))  # batch_size, num_keypoints, num_dim (x, y)
    occ_true = tf.ones((4, 80))  # batch_size, num_keypoints
    occ_masking_info = tf.zeros((4))  # batch_size

    # Test 1: 'unknown' loss_type
    with pytest.raises(ValueError):
        FpeLoss('unknown')

    # Test 2: 'l1' loss_type
    loss, _, __ = FpeLoss('l1')(y_true=y_true,
                                y_pred=y_pred,
                                occ_true=occ_true,
                                occ_masking_info=occ_masking_info)
    with tf.Session() as sess:
        loss_np = sess.run(loss)
    expected_loss_l1 = 4.0
    np.testing.assert_almost_equal(loss_np, expected_loss_l1, decimal=6)

    # Test 3: 'square_euclidean'
    loss, _, __ = FpeLoss('square_euclidean',
                          kpts_coeff=1.0)(y_true=y_true,
                                          y_pred=y_pred,
                                          occ_true=occ_true,
                                          occ_masking_info=occ_masking_info)
    with tf.Session() as sess:
        loss_np = sess.run(loss)
    expected_loss_squared_euc = 4.0
    np.testing.assert_almost_equal(loss_np, expected_loss_squared_euc, decimal=6)

    # Test 4: 'wing_loss'
    loss, _, __ = FpeLoss('wing_loss',
                          kpts_coeff=0.01)(y_true=y_true,
                                           y_pred=y_pred,
                                           occ_true=occ_true,
                                           occ_masking_info=occ_masking_info)
    with tf.Session() as sess:
        loss_np = sess.run(loss)
    expected_loss_wing = 6.48744
    np.testing.assert_almost_equal(loss_np, expected_loss_wing, decimal=6)

    # Test 5: occlusion masking
    occ_true = tf.zeros((4, 80))  # batch_size, num_keypoints
    mask_occ = True
    loss, _, __ = FpeLoss('l1',
                          mask_occ=mask_occ)(y_true=y_true,
                                             y_pred=y_pred,
                                             occ_true=occ_true,
                                             occ_masking_info=occ_masking_info)
    with tf.Session() as sess:
        loss_np = sess.run(loss)
    expected_loss_l1 = 0.0
    np.testing.assert_almost_equal(loss_np, expected_loss_l1, decimal=6)

    # Test 6: face region losses test
    y_true = tf.concat([tf.ones((4, 40, 2)), tf.zeros((4, 40, 2))], axis=1)
    occ_true = tf.ones((4, 80))  # batch_size, num_keypoints
    face_loss, mouth_loss, eyes_loss = FpeLoss('l1')(y_true=y_true,
                                                     y_pred=y_pred,
                                                     occ_true=occ_true,
                                                     occ_masking_info=occ_masking_info)
    with tf.Session() as sess:
        loss_np_face, loss_np_eyes, loss_np_mouth = sess.run([face_loss, eyes_loss, mouth_loss])
    expected_loss_l1_face = 2.0
    expected_loss_l1_mouth = 0.0
    expected_loss_l1_eyes = 1.3333334
    np.testing.assert_almost_equal(loss_np_face, expected_loss_l1_face, decimal=6)
    np.testing.assert_almost_equal(loss_np_eyes, expected_loss_l1_eyes, decimal=6)
    np.testing.assert_almost_equal(loss_np_mouth, expected_loss_l1_mouth, decimal=6)

    # Test 7: face dictionary weights test
    y_true = tf.concat([tf.ones((4, 30, 2)), tf.zeros((4, 30, 2)), tf.ones((4, 20, 2))], axis=1)
    weights_dict = {'face': 0.3, 'mouth': 0.3, 'eyes': 0.4}
    face_loss, mouth_loss, eyes_loss = FpeLoss('l1',
                                               weights_dict=weights_dict)(
                                               y_true=y_true,
                                               y_pred=y_pred,
                                               occ_true=occ_true,
                                               occ_masking_info=occ_masking_info)
    with tf.Session() as sess:
        loss_np_face, loss_np_eyes, loss_np_mouth = sess.run([face_loss, eyes_loss, mouth_loss])
    expected_loss_l1_face = 4.28
    expected_loss_l1_eyes = 4.0
    expected_loss_l1_mouth = 1.6
    np.testing.assert_almost_equal(loss_np_face, expected_loss_l1_face, decimal=6)
    np.testing.assert_almost_equal(loss_np_eyes, expected_loss_l1_eyes, decimal=6)
    np.testing.assert_almost_equal(loss_np_mouth, expected_loss_l1_mouth, decimal=6)

    # Test 8: 68 points landmarks
    y_true = tf.ones((3, 68, 2))  # batch_size, num_keypoints, num_dim (x, y)
    y_pred = tf.zeros((3, 68, 2))  # batch_size, num_keypoints, num_dim (x, y)
    occ_true = tf.ones((3, 68))  # batch_size, num_keypoints
    occ_masking_info = tf.zeros((3))  # batch_size
    loss, _, __ = FpeLoss('l1',
                          mask_occ=mask_occ)(y_true=y_true,
                                             y_pred=y_pred,
                                             occ_true=occ_true,
                                             occ_masking_info=occ_masking_info,
                                             num_keypoints=68)
    with tf.Session() as sess:
        loss_np = sess.run(loss)
    expected_loss_l1 = 3.0
    np.testing.assert_almost_equal(loss_np, expected_loss_l1, decimal=6)
