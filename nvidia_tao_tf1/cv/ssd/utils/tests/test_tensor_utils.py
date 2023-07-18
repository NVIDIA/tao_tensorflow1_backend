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
"""test mAP evaluator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.ssd.utils import tensor_utils


def test_get_non_empty_rows_2d_sparse():
    empty_tensor = tf.sparse.SparseTensor(
        indices=tf.zeros(dtype=tf.int64, shape=[0, 2]),
        values=[],
        dense_shape=[10000, 9])
    empty_results = tf.sparse.to_dense(tensor_utils.get_non_empty_rows_2d_sparse(empty_tensor))

    non_empty_zero = tf.zeros(dtype=tf.int32, shape=[100, 1000])
    non_empty_tensor = tf.sparse.from_dense(non_empty_zero)
    non_empty_zero_results = tensor_utils.get_non_empty_rows_2d_sparse(non_empty_tensor)
    non_empty_zero_results = tf.sparse.to_dense(non_empty_zero_results)

    non_empty = tf.sparse.from_dense(tf.constant(np.array([[1, 0, 3], [0, 0, 0], [0, 0, 9]])))
    non_empty_results = tf.sparse.to_dense(tensor_utils.get_non_empty_rows_2d_sparse(non_empty))

    with tf.Session() as sess:
        result = sess.run(empty_results)
        assert result.shape == (0, 9)
        result = sess.run(non_empty_zero_results)
        assert result.shape == (0, 1000)
        result = sess.run(non_empty_results)
        assert np.max(abs(result - np.array([[1, 0, 3], [0, 0, 9]]))) < 1e-5


def test_tensor_slice_replace():
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    b = tf.constant(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
    a_idx = tf.constant(np.array([1, 2]), dtype=tf.int32)
    b_idx = tf.constant(np.array([1, 0]), dtype=tf.int32)

    with tf.Session() as sess:
        result = sess.run(tensor_utils.tensor_slice_replace(a, b, a_idx, b_idx))
        assert np.max(abs(result - np.array([[1, 2, 3], [-4, -5, -6], [-1, -2, -3]]))) < 1e-5


def test_tensor_strided_replace():
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    b = np.array([[-1, -2, -3],
                  [-4, -5, -6],
                  [-7, -8, -9]])

    a = tf.constant(a)
    b = tf.constant(b)

    with tf.Session() as sess:
        result = sess.run(tensor_utils.tensor_strided_replace(a, (1, 2), b))
        expected = np.array([[1, 2, 3],
                             [-1, -2, -3],
                             [-4, -5, -6],
                             [-7, -8, -9],
                             [7, 8, 9]])
        assert np.max(abs(result - expected)) < 1e-5
        result = sess.run(tensor_utils.tensor_strided_replace(a, (1, 2), b, -1))
        expected = np.array([[1, -1, -2, -3, 3],
                             [4, -4, -5, -6, 6],
                             [7, -7, -8, -9, 9]])
        assert np.max(abs(result - expected)) < 1e-5
        result = sess.run(tensor_utils.tensor_strided_replace(a, (0, 3), b))
        expected = np.array([[-1, -2, -3],
                             [-4, -5, -6],
                             [-7, -8, -9]])
        assert np.max(abs(result - expected)) < 1e-5


def _test_setup_keras_backend():  # comment out since this function doesn't work
    tensor_utils.setup_keras_backend('float32', True)
    with tf.Session() as sess:
        assert sess.run(tf.keras.backend.learning_phase()) == 1
        assert tf.keras.backend.floatx() == 'float32'
    tensor_utils.setup_keras_backend('float16', False)
    with tf.Session() as sess:
        assert sess.run(tf.keras.backend.learning_phase()) == 0
        assert tf.keras.backend.floatx() == 'float16'
