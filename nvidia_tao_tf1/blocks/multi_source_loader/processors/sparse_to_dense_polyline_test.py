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
"""Tests for instance_mapper.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized

from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.sparse_to_dense_polyline import (
    SparseToDensePolyline,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import Polygon2DLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures import (
    make_coordinates2d,
    make_tags,
)


def make_labels(
    shapes_per_frame,
    height,
    width,
    coordinates_per_polygon,
    coordinate_values,
    classes,
    attributes,
):
    return Polygon2DLabel(
        vertices=make_coordinates2d(
            shapes_per_frame=shapes_per_frame,
            height=height,
            width=width,
            coordinates_per_polygon=coordinates_per_polygon,
            coordinate_values=coordinate_values,
        ),
        classes=make_tags(classes),
        attributes=make_tags(attributes),
    )


class TestSparseToDensePolyline(ProcessorTestCase):
    @parameterized.expand(
        [
            # Input to check if the output is computed correctly for vertices
            # having adjacent x axis values.
            [
                [[1]],  # Shapes per frame of Input.
                [[[[0]]]],  # Classes.
                [[[[1]]]],  # Attributes.
                2,  # Input Coordinates per polygon.
                [[1.0, 1.0], [2.0, 3.0]],  # Input Coordinate values.
                [[1]],  # Output shapes per frame.
                [[[[0]]]],  # Output Classes.
                [[[[1]]]],  # Output Attributes.
                2,  # Output coordinates per polygon
                [[1.0, 1.0], [2.0, 3.0]],  # Output coordinate values.
            ],
            # Input to check if the output is computed correctly for vertices
            # with same x axis.
            [
                [[1]],  # Shapes per frame of Input.
                [[[[0]]]],  # Classes.
                [[[[1]]]],  # Attributes.
                2,  # Input Coordinates per polygon.
                [[1.0, 1.0], [1.0, 4.0]],  # Input Coordinate values.
                [[1]],  # Output shapes per frame.
                [[[[0]]]],  # Output Classes.
                [[[[1]]]],  # Output Attributes.
                4,  # Output coordinates per polygon
                [
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [1.0, 3.0],
                    [1.0, 4.0],
                ],  # Output coordinate values.
            ],
            # Input to check if the output is computed correctly for vertices
            # with same x axis but reversed y values.
            [
                [[1]],  # Shapes per frame of Input.
                [[[[0]]]],  # Classes.
                [[[[1]]]],  # Attributes.
                2,  # Input Coordinates per polygon.
                [[1.0, 4.0], [1.0, 1.0]],  # Input Coordinate values.
                [[1]],  # Output shapes per frame.
                [[[[0]]]],  # Output Classes.
                [[[[1]]]],  # Output Attributes.
                4,  # Output coordinates per polygon
                [
                    [1.0, 4.0],
                    [1.0, 3.0],
                    [1.0, 2.0],
                    [1.0, 1.0],
                ],  # Output coordinate values.
            ],
            # Input to check if the op can compute dense vertices correctly
            # when multiple vertices are provided.
            [
                [[1]],  # Shapes per frame of Input.
                [[[[0]]]],  # Classes.
                [[[[1]]]],  # Attributes.
                3,  # Input Coordinates per polygon.
                [[1.0, 1.0], [4.0, 4.0], [7.0, 7.0]],  # Input Coordinate values.
                [[1]],  # Output shapes per frame.
                [[[[0]]]],  # Output Classes.
                [[[[1]]]],  # Output Attributes.
                7,  # Output coordinates per polygon
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [4.0, 4.0],
                    [5.0, 5.0],
                    [6.0, 6.0],
                    [7.0, 7.0],
                ],  # Output coordinate values.
            ],
            # Input to check if the op can compute dense vertices for multiple
            # polylines in a frame.
            [
                [[2]],  # Shapes per frame of Input.
                [[[[0], [2]]]],  # Classes.
                [[[[1], [0]]]],  # Attributes.
                3,  # Input Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [4.0, 4.0],
                    [7.0, 7.0],
                    [8.0, 16.0],
                    [10.0, 20.0],
                    [14.0, 28.0],
                ],  # Input Coordinate values.
                [[2]],  # Output shapes per frame.
                [[[[0], [2]]]],  # Output Classes.
                [[[[1], [0]]]],  # Output Attributes.
                7,  # Output coordinates per polygon
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [4.0, 4.0],
                    [5.0, 5.0],
                    [6.0, 6.0],
                    [7.0, 7.0],
                    [8.0, 16.0],
                    [9.0, 18.0],
                    [10.0, 20.0],
                    [11.0, 22.0],
                    [12.0, 24.0],
                    [13.0, 26.0],
                    [14.0, 28.0],
                ],  # Output coordinate values.
            ],
            # Input to check if the op computes vertices provided with negative slope
            # correctly.
            [
                [[2]],  # Shapes per frame of Input.
                [[[[0], [2]]]],  # Classes.
                [[[[1], [0]]]],  # Attributes.
                3,  # Input Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [4.0, 4.0],
                    [7.0, 7.0],
                    [14.0, 28.0],
                    [10.0, 20.0],
                    [8.0, 16.0],
                ],  # Input Coordinate values.
                [[2]],  # Output shapes per frame.
                [[[[0], [2]]]],  # Output Classes.
                [[[[1], [0]]]],  # Output Attributes.
                7,  # Output coordinates per polygon
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [4.0, 4.0],
                    [5.0, 5.0],
                    [6.0, 6.0],
                    [7.0, 7.0],
                    [14.0, 28.0],
                    [13.0, 26.0],
                    [12.0, 24.0],
                    [11.0, 22.0],
                    [10.0, 20.0],
                    [9.0, 18.0],
                    [8.0, 16.0],
                ],  # Output coordinate values.
            ],
        ]
    )
    def test_sparse_to_dense_polyline(
        self,
        shapes_per_frame,
        classes,
        attributes,
        coordinates_per_polygon,
        coordinate_values,
        expected_shapes_per_frame,
        expected_classes,
        expected_attributes,
        expected_coordinates_per_polygon,
        expected_coordinate_values,
    ):
        with self.session() as session:
            dense_polyline_converter = SparseToDensePolyline()
            input_labels = make_labels(
                shapes_per_frame,
                10,
                10,
                coordinates_per_polygon,
                coordinate_values,
                classes,
                attributes,
            )
            expected_labels = make_labels(
                expected_shapes_per_frame,
                10,
                10,
                expected_coordinates_per_polygon,
                expected_coordinate_values,
                expected_classes,
                expected_attributes,
            )

            output_labels = dense_polyline_converter(input_labels)
            input_labels, output_labels, expected_labels = session.run(
                [input_labels, output_labels, expected_labels]
            )

            output_vertices_indices = output_labels.vertices.coordinates.indices
            output_vertices_values = output_labels.vertices.coordinates.values
            output_vertices_shape = output_labels.vertices.coordinates.dense_shape
            expected_vertices_indices = expected_labels.vertices.coordinates.indices
            expected_vertices_values = expected_labels.vertices.coordinates.values
            expected_vertices_shape = expected_labels.vertices.coordinates.dense_shape

            # Check if the output equals the expected
            self.assertAllEqual(output_vertices_indices, expected_vertices_indices)
            self.assertAllEqual(output_vertices_values, expected_vertices_values)
            self.assertAllEqual(output_vertices_shape, expected_vertices_shape)
