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
"""Test conversion of values and counts to sparse tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from parameterized import parameterized
import pytest
from six import binary_type
import tensorflow as tf

from nvidia_tao_tf1.core.processors import values_and_count_to_sparse_tensor


class ValuesAndCountToSparseTensorTest(tf.test.TestCase):
    def assertSparseTensorShapeEqual(self, st, indices, values, dense_shape):
        """Assert the shapes of the tensors in a SparseTensor conform to the expected shapes."""
        self.assertAllEqual(indices, st.indices.get_shape().as_list())
        self.assertAllEqual(values, st.values.get_shape().as_list())
        self.assertAllEqual(dense_shape, st.dense_shape.get_shape().as_list())

    def assertSparseTensorValueEqual(self, st, values, indices, dense_shape):
        """Assert two SparseTensor values are the same."""
        self.assertAllEqual(indices, st.indices)
        self.assertAllEqual(values, st.values)
        self.assertAllEqual(dense_shape, st.dense_shape)

    @parameterized.expand(
        [
            (np.array([1, 1, 1], dtype=np.int32),),
            (np.array([1, 1, 1, 1, 1], dtype=np.int32),),
            (np.array([0], dtype=np.int32),),
            (np.array([3], dtype=np.int32),),
            (np.array([5], dtype=np.int32),),
        ]
    )
    def test_mismatch_counts(self, counts):
        with pytest.raises(tf.errors.InvalidArgumentError):
            with self.test_session():
                values = np.array([0, 1, 2, 3], dtype=np.int32)
                values_and_count_to_sparse_tensor(values, counts).eval()

    @parameterized.expand(
        [
            (np.array([1, 1, 1], dtype=np.int32),),
            (np.array([1, 1, 1, 1, 1], dtype=np.int32),),
            (np.array([0], dtype=np.int32),),
            (np.array([3], dtype=np.int32),),
            (np.array([5], dtype=np.int32),),
        ]
    )
    def test_mismatch_counts_of_counts(self, counts_of_counts):
        with pytest.raises(tf.errors.InvalidArgumentError):
            with self.test_session():
                values = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32)
                counts = np.array([1, 1, 1, 1], dtype=np.int32)
                values_and_count_to_sparse_tensor(
                    values, counts, counts_of_counts
                ).eval()

    def test_2d_empty_tensor_only_counts(self,):
        values = np.zeros([0, 2], dtype=np.float32)
        counts = [0, 0]
        st = values_and_count_to_sparse_tensor(values, counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 3], values=[None], dense_shape=[3]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=np.zeros([0, 3]),
                values=np.zeros([0]),
                dense_shape=[2, 0, 2],
            )

    def test_2d_empty_tensor(self,):
        values = np.zeros([0, 2], dtype=np.float32)
        counts = [0, 0]
        counts_of_counts = [0, 2, 0]
        st = values_and_count_to_sparse_tensor(
            values, counts, counts_of_counts=counts_of_counts
        )
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 4], values=[None], dense_shape=[4]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=np.zeros([0, 4]),
                values=np.zeros([0]),
                dense_shape=[3, 2, 0, 2],
            )

    @parameterized.expand(
        [(np.int32,), (np.int64,), (np.float32,), (np.float64,), (binary_type,)]
    )
    def test_1d_only_counts_all_in_one(self, dtype):
        values = np.array([0, 1, 2], dtype=dtype)
        counts = [0, 3]
        st = values_and_count_to_sparse_tensor(values, counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 2], values=[None], dense_shape=[2]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=[[1, 0], [1, 1], [1, 2]],
                values=values.flatten(),
                dense_shape=[2, 3],
            )

    @parameterized.expand(
        [(np.int32,), (np.int64,), (np.float32,), (np.float64,), (binary_type,)]
    )
    def test_1d_only_counts_distributed(self, dtype):
        values = np.array([0, 1, 2], dtype=dtype)
        counts = [1, 0, 1, 0, 1]
        st = values_and_count_to_sparse_tensor(values, counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 2], values=[None], dense_shape=[2]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=[[0, 0], [2, 0], [4, 0]],
                values=values.flatten(),
                dense_shape=[5, 1],
            )

    @parameterized.expand(
        [(np.int32,), (np.int64,), (np.float32,), (np.float64,), (binary_type,)]
    )
    def test_2d_only_counts_all_in_one(self, dtype):
        values = np.array([[0, 1], [2, 3], [4, 5]], dtype=dtype)
        counts = [0, 3]
        st = values_and_count_to_sparse_tensor(values, counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 3], values=[None], dense_shape=[3]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=[
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 2, 0],
                    [1, 2, 1],
                ],
                values=values.flatten(),
                dense_shape=[2, 3, 2],
            )

    @parameterized.expand(
        [(np.int32,), (np.int64,), (np.float32,), (np.float64,), (binary_type,)]
    )
    def test_2d_only_counts_distributed(self, dtype):
        values = np.array([[0, 1], [2, 3], [4, 5]], dtype=dtype)
        counts = [1, 0, 1, 0, 1]
        st = values_and_count_to_sparse_tensor(values, counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 3], values=[None], dense_shape=[3]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=[
                    [0, 0, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [2, 0, 1],
                    [4, 0, 0],
                    [4, 0, 1],
                ],
                values=values.flatten(),
                dense_shape=[5, 1, 2],
            )

    @parameterized.expand(
        [(np.int32,), (np.int64,), (np.float32,), (np.float64,), (binary_type,)]
    )
    def test_2d_distributed(self, dtype):
        values = np.array([[0, 1], [2, 3], [4, 5]], dtype=dtype)
        counts = [1, 0, 1, 0, 1]
        counts_of_counts = [3, 2]
        st = values_and_count_to_sparse_tensor(values, counts, counts_of_counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 4], values=[None], dense_shape=[4]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=[
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 2, 0, 0],
                    [0, 2, 0, 1],
                    [1, 1, 0, 0],
                    [1, 1, 0, 1],
                ],
                values=values.flatten(),
                dense_shape=[2, 3, 1, 2],
            )

    @parameterized.expand(
        [(np.int32,), (np.int64,), (np.float32,), (np.float64,), (binary_type,)]
    )
    def test_multi_dim_only_counts_all_in_one(self, dtype):
        values = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [0, 1]]])
        counts = [0, 3]
        st = values_and_count_to_sparse_tensor(values, counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 4], values=[None], dense_shape=[4]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=[
                    [1, 0, 0, 0],
                    [1, 0, 0, 1],
                    [1, 0, 1, 0],
                    [1, 0, 1, 1],
                    [1, 1, 0, 0],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                    [1, 2, 0, 0],
                    [1, 2, 0, 1],
                    [1, 2, 1, 0],
                    [1, 2, 1, 1],
                ],
                values=values.flatten(),
                dense_shape=[2, 3, 2, 2],
            )

    @parameterized.expand(
        [(np.int32,), (np.int64,), (np.float32,), (np.float64,), (binary_type,)]
    )
    def test_multi_dim_only_counts_distributed(self, dtype):
        values = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [0, 1]]])
        counts = [1, 0, 1, 0, 1]
        st = values_and_count_to_sparse_tensor(values, counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 4], values=[None], dense_shape=[4]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=[
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 1],
                    [2, 0, 0, 0],
                    [2, 0, 0, 1],
                    [2, 0, 1, 0],
                    [2, 0, 1, 1],
                    [4, 0, 0, 0],
                    [4, 0, 0, 1],
                    [4, 0, 1, 0],
                    [4, 0, 1, 1],
                ],
                values=values.flatten(),
                dense_shape=[5, 1, 2, 2],
            )

    @parameterized.expand(
        [(np.int32,), (np.int64,), (np.float32,), (np.float64,), (binary_type,)]
    )
    def test_multi_dim_distributed(self, dtype):
        values = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [0, 1]]])
        counts = [1, 0, 1, 0, 1]
        counts_of_counts = [3, 2]
        st = values_and_count_to_sparse_tensor(values, counts, counts_of_counts)
        self.assertSparseTensorShapeEqual(
            st, indices=[None, 5], values=[None], dense_shape=[5]
        )
        with self.test_session():
            self.assertSparseTensorValueEqual(
                st=st.eval(),
                indices=[
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 2, 0, 0, 0],
                    [0, 2, 0, 0, 1],
                    [0, 2, 0, 1, 0],
                    [0, 2, 0, 1, 1],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                ],
                values=values.flatten(),
                dense_shape=[2, 3, 1, 2, 2],
            )
