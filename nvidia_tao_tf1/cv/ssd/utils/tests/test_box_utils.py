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
"""test box utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.ssd.utils import box_utils


def test_np_iou():
    a = np.array([0, 0, 2, 2])
    b = np.array([[0, 0, 2, 2], [0, 0, 2, 1], [1, 0, 2, 2], [9, 10, 11, 12], [1, 1, 3, 3]])
    assert max(abs(box_utils.np_elem_iou(a, b) - np.array([1.0, 0.5, 0.5, 0.0, 1.0 / 7.0]))) < 1e-10


def test_np_iou_binary():
    a = np.array([[0, 0, 2, 2], [0, 0, 2, 1], [1, 0, 2, 2]])
    b = np.array([[0, 0, 2, 2], [0, 0, 2, 1], [1, 0, 2, 2]])
    assert max(abs(box_utils.np_elem_iou(a, b) - np.array([1.0, 1.0, 1.0]))) < 1e-10


def test_tf_iou():
    a = tf.constant(np.array([[0, 0, 2, 2]]), dtype=tf.float32)
    b = tf.constant(np.array([[0, 0, 2, 2], [0, 0, 2, 1], [1, 0, 2, 2], [9, 10, 11, 12],
                              [1, 1, 3, 3]]), dtype=tf.float32)
    with tf.Session() as sess:
        result = sess.run(box_utils.iou(a, b))
        assert np.max(abs(result - np.array([[1.0, 0.5, 0.5, 0.0, 1.0 / 7.0]]))) < 1e-5


def test_bipartite_match_row():
    sim_matrix = [[0.9, 0.8, 0.3, 0.2, 0.1, 0.7, 0.5],
                  [0.6, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                  [0.0, 0.0, 0.0, 0.0, 0.9, 0.7, 0.8],
                  [0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0]]
    a = tf.constant(np.array(sim_matrix), dtype=tf.float32)
    b = a - 1.0
    with tf.Session() as sess:
        result = sess.run(box_utils.bipartite_match_row(a))
        assert max(abs(result - np.array([0, 1, 4, 2]))) < 1e-5
        result = sess.run(box_utils.bipartite_match_row(b))
        assert max(abs(result - np.array([0, 1, 4, 2]))) < 1e-5


def test_multi_match():
    sim_matrix = [[0.9, 0.8, 0.3, 0.2, 0.1, 0.7, 0.5],
                  [0.6, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                  [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.8],
                  [0.0, 0.0, 0.4, 0.3, 0.0, 0.0, 0.0]]
    a = tf.constant(np.array(sim_matrix), dtype=tf.float32)
    with tf.Session() as sess:
        gt, anchor = sess.run(box_utils.multi_match(a, 0.2))
        assert set(zip(anchor, gt)) == set([(0, 0), (1, 0), (2, 3), (3, 3), (5, 2), (6, 2)])


def test_corners_to_centroids():
    corner_box = [[1.0, 1.0, 2.0, 2.0],
                  [0.0, 0.0, 2.0, 3.0]]
    tf_box = tf.constant(np.array(corner_box), dtype=tf.float32)
    expected = [[1.5, 1.5, 1.0, 1.0],
                [1.0, 1.5, 2.0, 3.0]]
    with tf.Session() as sess:
        result = sess.run(box_utils.corners_to_centroids(tf_box, 0))
        assert np.max(abs(result - np.array(expected))) < 1e-5
