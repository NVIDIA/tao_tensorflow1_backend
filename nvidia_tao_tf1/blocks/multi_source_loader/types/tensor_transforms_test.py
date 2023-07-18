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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from parameterized import parameterized
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.types.tensor_transforms import (
    map_and_stack,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.tensor_transforms import (
    sparsify_dense_coordinates,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.tensor_transforms import (
    vector_and_counts_to_sparse_tensor,
)


class TensorTransformsTest(tf.test.TestCase):
    def test_map_and_stack_0_rows(self):
        count = 0
        with self.session():
            indices = tf.range(count)
            out = map_and_stack(lambda i: [[0, i]], indices)
            self.assertEqual(None, out.shape)
            self.assertAllEqual([], out.eval())

    @parameterized.expand([[1, [[0, 0]]], [2, [[0, 0], [0, 1]]]])
    def test_map_and_stack_multiple_rows(self, count, expected):
        with self.session():
            indices = tf.range(count)
            out = map_and_stack(lambda i: [[0, i]], indices)
            self.assertAllEqual(expected, out.eval())

    def test_sparsifies_empty_vector(self):
        classes = tf.constant([])
        counts = tf.constant([], tf.int64)
        with self.session():
            sparse = vector_and_counts_to_sparse_tensor(classes, counts).eval()
            self.assertAllEqual([], sparse.values)
            self.assertEqual((0, 2), sparse.indices.shape)
            self.assertAllEqual((0, 0), sparse.dense_shape)

    @parameterized.expand(
        [
            [["lane"], [1], [[0, 0]]],
            [["lane", "pole"], [2], [[0, 0], [0, 1]]],
            [["lane", "pole"], [1, 1], [[0, 0], [1, 0]]],
            [["lane", "pole", "sign"], [1, 2], [[0, 0], [1, 0], [1, 1]]],
            [["lane", "pole", "sign"], [2, 1], [[0, 0], [0, 1], [1, 0]]],
        ]
    )
    def test_sparsifies_vector(self, classes, counts, expected_indices):
        expected_total_elements = len(counts)
        expected_max_elements = max(counts)

        classes = tf.constant(classes)
        counts = tf.constant(counts, tf.int64)
        with self.session():
            sparse = vector_and_counts_to_sparse_tensor(classes, counts).eval()
            self.assertAllEqual(classes, sparse.values)
            self.assertAllEqual(expected_indices, sparse.indices)
            self.assertEqual(expected_total_elements, sparse.dense_shape[0])
            self.assertEqual(expected_max_elements, sparse.dense_shape[1])

    def test_sparsifies_empty_coordinates(self):
        with self.session() as sess:
            actual = sparsify_dense_coordinates(
                tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.int64)
            )
            actual = sess.run(actual)
            self.assertAllEqual([], actual.values)
            self.assertAllEqual([0, 3], actual.indices.shape)
            self.assertAllEqual([0, 0, 0], actual.dense_shape)

    @parameterized.expand(
        [
            [
                # Single point.
                [[7.0, 7.0]],
                [1],
                [[0, 0, 0], [0, 0, 1]],
                [1, 1, 2],
            ],
            [
                # Two points.
                [
                    # First
                    [7.0, 7.0],
                    # Second
                    [42.0, 7.0],
                ],
                [1, 1],
                [
                    # First indices
                    [0, 0, 0],
                    [0, 0, 1],
                    # Second indices
                    [1, 0, 0],
                    [1, 0, 1],
                ],
                [2, 1, 2],
            ],
            [
                # Single line.
                [[0.0, 0.0], [7.0, 7.0]],
                [2],
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
                [1, 2, 2],
            ],
            [
                # Two lines.
                [
                    # First
                    [0.0, 0.0],
                    [7.0, 7.0],
                    # Second
                    [0.0, 7.0],
                    [7.0, 0.0],
                ],
                [2, 2],
                [
                    # First
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    # Second
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                [2, 2, 2],
            ],
            [
                # Point and a line
                [
                    # Point
                    [7.0, 7.0],
                    # Line
                    [0.0, 0.0],
                    [7.0, 7.0],
                ],
                [1, 2],
                [
                    # Point indices
                    [0, 0, 0],
                    [0, 0, 1],
                    # Line indices
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                [2, 2, 2],
            ],
            [
                # Line and a point
                [
                    # Line
                    [0.0, 0.0],
                    [7.0, 7.0],
                    # Point
                    [7.0, 7.0],
                ],
                [2, 1],
                [
                    # Line indices
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    # Point indices
                    [1, 0, 0],
                    [1, 0, 1],
                ],
                [2, 2, 2],
            ],
            [
                # Point, Line, Point, Line
                [
                    # First point
                    [1.0, 2.0],
                    # First line
                    [1.0, 2.0],
                    [3.0, 4.0],
                    # Second point
                    [2.0, 1.0],
                    # Second line
                    [4.0, 3.0],
                    [2.0, 1.0],
                ],
                [1, 2, 1, 2],
                [
                    # First point indices
                    [0, 0, 0],
                    [0, 0, 1],
                    # First line indices
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                    # Second point indices
                    [2, 0, 0],
                    [2, 0, 1],
                    # Second line indices
                    [3, 0, 0],
                    [3, 0, 1],
                    [3, 1, 0],
                    [3, 1, 1],
                ],
                [4, 2, 2],
            ],
        ]
    )
    def test_sparsifies_coordinates(
        self,
        dense_coordinates,
        vertex_counts_per_polygon,
        expected_indices,
        expected_dense_shape,
    ):
        expected_values = list(itertools.chain.from_iterable(dense_coordinates))
        dense_coordinates = tf.constant(dense_coordinates)
        vertex_counts_per_polygon = tf.constant(
            vertex_counts_per_polygon, dtype=tf.int64
        )

        with self.session() as sess:
            actual = sparsify_dense_coordinates(
                dense_coordinates, vertex_counts_per_polygon
            )
            actual = sess.run(actual)
            self.assertAllEqual(expected_values, actual.values)
            self.assertAllEqual(expected_indices, actual.indices)
            self.assertAllEqual(expected_dense_shape, actual.dense_shape)
