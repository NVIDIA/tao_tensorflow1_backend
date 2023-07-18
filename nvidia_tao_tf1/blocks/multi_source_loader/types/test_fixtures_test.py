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

"""Tests for test fixtures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import tensorflow as tf
import nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures as fixtures


class TestFixturesTest(tf.test.TestCase):
    def test_tags(self):
        tags = fixtures.make_tags([[[[1]]]])
        with self.session() as sess:
            tags = sess.run(tags)
            self.assertAllEqual([[0, 0, 0, 0]], tags.indices)
            self.assertAllEqual([1], tags.values)

    @parameterized.expand(
        [
            [
                [[1]],
                5,
                10,
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 2, 1],
                    [0, 0, 0, 3, 0],
                    [0, 0, 0, 3, 1],
                    [0, 0, 0, 4, 0],
                    [0, 0, 0, 4, 1],
                ],
                [42],
                [[0, 0, 0, 0]],
                [7],
                [[0, 0, 0, 0]],
            ],
            [
                [[1, 1]],
                2,
                8,
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 1],
                ],
                [42, 42],
                [[0, 0, 0, 0], [0, 1, 0, 0]],
                [7, 7],
                [[0, 0, 0, 0], [0, 1, 0, 0]],
            ],
        ]
    )
    def test_polygon2d_labels(
        self,
        shapes_per_frame,
        coordinates_per_polygon,
        expected_total_coordinates,
        expected_coordinate_indices,
        expected_classes,
        expected_class_indices,
        expected_attributes,
        expected_attribute_indices,
    ):
        with self.session() as sess:
            actual = sess.run(
                fixtures.make_polygon2d_label(
                    shapes_per_frame, [42], [7], 128, 248, coordinates_per_polygon
                )
            )

            self.assertAllEqual(
                expected_coordinate_indices, actual.vertices.coordinates.indices
            )
            self.assertAllEqual(
                expected_total_coordinates, len(actual.vertices.coordinates.values)
            )

            self.assertAllEqual(expected_class_indices, actual.classes.indices)
            self.assertAllEqual(expected_classes, actual.classes.values)

            self.assertAllEqual(expected_attribute_indices, actual.attributes.indices)
            self.assertAllEqual(expected_attributes, actual.attributes.values)
