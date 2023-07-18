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

"""Tests for PolygonRasterizer processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized

from nvidia_tao_tf1.blocks.multi_source_loader.processors.multiple_polyline_to_polygon import (  # noqa
    MultiplePolylineToPolygon,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.polygon2d_label import (
    Polygon2DLabel,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures import (
    make_coordinates2d,
    make_tags,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


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


class TestMultiplePolylineToPolygon(ProcessorTestCase):
    @parameterized.expand([[tuple()], [(5,)], [(12, 2)]])
    def test_empty_labels(self, batch_args):
        input_labels = self.make_empty_polygon2d_labels(*batch_args)

        processor = MultiplePolylineToPolygon([0], [0])
        with self.session() as sess:
            transformed_labels = processor.process(input_labels)
            transformed_labels = sess.run(transformed_labels)
            input_labels = sess.run(input_labels)
            transformed_polygons = transformed_labels.vertices.coordinates
            input_polygons = input_labels.vertices.coordinates
            self.assertSparseEqual(transformed_polygons, input_polygons)
            self.assertSparseEqual(transformed_labels.classes, input_labels.classes)

    @parameterized.expand(
        [
            # Input has Polylines that form a triangle in order to check if the op
            # correctly orders these polylines and combines them into a polygon.
            # The polylines are shuffled and need to be sorted in order to form a triangle.
            # For example, Output expected order is (1, 3, 2). This test case covers the
            # following scenario:
            # a) Polylines that need to be sorted to form polygon.
            [
                [[3]],  # Shapes per frame of Input.
                [[[[1], [0], [2]]]],  # Classes
                [[[[0], [0], [0]]]],  # Attributes (same attributes so combine all).
                2,  # Input Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [2.0, 4.0],
                    [3.03, 1.04],
                    [1.02, 1.04],
                    [2.02, 4.03],
                    [3.05, 1.01],
                ],  # Input Coordinate values.
                [[1]],  # Output shapes per frame.
                [[[[0]]]],  # Output Classes.
                [[[[0]]]],  # Output Attributes.
                [0],  # Attribute ids list (For attribute to class mapping)
                [0],  # Class ids list. (For attribute to class mapping)
                6,  # Output Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [2.0, 4.0],
                    [2.02, 4.03],
                    [3.05, 1.01],
                    [3.03, 1.04],
                    [1.02, 1.04],
                ],  # Output Coordinate values.
            ],
            # Input has polylines in reverse order and are shuffled. The test checks
            # if the op can handle the case where polylines that form the polygon
            # are reversed as well shuffled. The output polylines should be sorted
            # in correct order and reverse should be handled. This test case covers
            # the following scenario. (The last two polylines out of the three are
            # reversed.)
            # b) Polylines that need to be sorted and reversed.
            [
                [[3]],  # Shapes per frame of Input.
                [[[[1], [0], [2]]]],  # Classes
                [[[[0], [0], [0]]]],  # Attributes (same attributes so combine all).
                2,  # Input Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [2.0, 4.0],
                    [1.02, 1.04],
                    [3.03, 1.04],
                    [3.05, 1.01],
                    [2.02, 4.03],
                ],  # Input Coordinate values.
                [[1]],  # Output shapes per frame.
                [[[[2]]]],  # Output Classes.
                [[[[0]]]],  # Output Attributes.
                [0],  # Attribute ids list. (For attribute to class mapping)
                [2],  # Class ids list. (For attribute to class mapping)
                6,  # Output Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [2.0, 4.0],
                    [2.02, 4.03],
                    [3.05, 1.01],
                    [3.03, 1.04],
                    [1.02, 1.04],
                ],  # Output Coordinate values.
            ],
            # Input has polylines that form 2 triangles and one polyline is common
            # in both the triangles. The test checks whether the output indices
            # forming the 2 polygons are properly sorted and polyline with
            # multiple attributes are handled properly. The polyline with multiple
            # attributes is added in the 1st as well as 2nd polygon's vertices.
            # c) Handle Polylines with multiple attributes (same polyline in 2
            #    polygons).
            [
                [[5]],  # Shapes per frame of Input.
                [[[[1], [0], [2], [0], [1], [1]]]],  # Classes
                [[[[0], [0], [0, 1], [1], [1]]]],  # Attributes (one polyline has 2
                # attributes).
                2,  # Input Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [2.0, 4.0],
                    [1.02, 1.04],
                    [3.03, 1.04],
                    [3.05, 1.01],
                    [2.02, 4.03],
                    [4.0, 3.0],
                    [2.1, 4.1],
                    [3.04, 1.03],
                    [4.07, 3.09],
                ],  # Input Coordinate values.
                [[2]],  # Output shapes per frame.
                [[[[2], [3]]]],  # Output Classes.
                [[[[1], [0]]]],  # Output Attributes.
                [0, 1],  # Attribute ids list. (For attribute to class mapping)
                [3, 2],  # Class ids list. (For attribute to class mapping)
                6,  # Output Coordinates per polygon.
                [
                    [3.05, 1.01],
                    [2.02, 4.03],
                    [2.1, 4.1],
                    [4.0, 3.0],
                    [4.07, 3.09],
                    [3.04, 1.03],
                    [1.0, 1.0],
                    [2.0, 4.0],
                    [2.02, 4.03],
                    [3.05, 1.01],
                    [3.03, 1.04],
                    [1.02, 1.04],
                ],  # Output Coordinate values.
            ],
        ]
    )
    def test_multiple_polyline_to_polygon(
        self,
        shapes_per_frame,
        classes,
        attributes,
        coordinates_per_polygon,
        coordinate_values,
        expected_shapes_per_frame,
        expected_classes,
        expected_attributes,
        attribute_ids_list,
        class_ids_list,
        expected_coordinates_per_polygon,
        expected_coordinate_values,
    ):
        processor = MultiplePolylineToPolygon(attribute_ids_list, class_ids_list)
        with self.session() as sess:
            input_labels = make_labels(
                shapes_per_frame,
                5,
                5,
                coordinates_per_polygon,
                coordinate_values,
                classes,
                attributes,
            )
            expected_labels = make_labels(
                expected_shapes_per_frame,
                5,
                5,
                expected_coordinates_per_polygon,
                expected_coordinate_values,
                expected_classes,
                expected_attributes,
            )
            transformed_labels = processor.process(input_labels)
            transformed_labels, expected_labels = sess.run(
                [transformed_labels, expected_labels]
            )
            transformed_polygons = transformed_labels.vertices.coordinates
            expected_polygons = expected_labels.vertices.coordinates
            self.assertSparseEqual(transformed_polygons, expected_polygons)
            self.assertSparseEqual(transformed_labels.classes, expected_labels.classes)

    @parameterized.expand(
        [
            # Input to check that the polylines are not combined when none of the
            # attributes are equal. The input should be returned as is to the output.
            # This test case covers the following scenario:
            # d) Mix of polylines with attributes and polylines without attributes.
            # e) None of the polylines are combined because attributes are either
            #    missing or all unequal.
            [
                [[3]],  # Shapes per frame of Input.
                [[[[1], [0], [2]]]],  # Classes
                [[[[0], [1]]]],  # Attributes.
                [0, 1],  # Attribute ids list. (To define Attribute to class mapping)
                [0, 0],  # Class ids list. (To define Attribute to class mapping)
                2,  # Input Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [2.0, 4.0],
                    [1.02, 1.04],
                    [3.03, 1.04],
                    [3.05, 1.01],
                    [2.02, 4.03],
                ],  # Input Coordinate values.
            ],
            # Input to check that the op works with multiple frames in an example.
            # The input coordinates are random values. The test confirms that the
            # op can handle input with multiple time frames in an example as well.
            # f) Handle data with multiple frames in an example (time dimension).
            [
                [[3, 4], [2]],  # Shapes per frame of Input.
                [[[[0], [2], [1]], [[1], [2], [0], [1]]], [[[1], [1]]]],  # Classes.
                [[[[0], [1], [2]], [[1], [0], [2]]], [[[0]]]],  # Attributes.
                [0, 1, 2],  # Attribute ids list. (To define attribute to class mapping)
                [0, 0, 1],  # Class ids list. (To define attribute to class mapping)
                2,  # Input Coordinates per polygon.
                [
                    [1.0, 1.0],
                    [5.0, 3.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [1.0, 2.0],
                    [5.0, 4.0],
                    [1.02, 0.5],
                    [0.2, 0.7],
                    [1.4, 4.1],
                    [2.1, 3.3],
                    [0.7, 0.8],
                    [0.9, 1.1],
                    [4.5, 3.4],
                    [4.0, 4.9],
                    [3.1, 1.2],
                    [2.2, 2.3],
                    [3.3, 2.2],
                    [3.4, 6.5],
                ],  # Input Coordinate values.
            ],
        ]
    )
    def test_not_combined(
        self,
        shapes_per_frame,
        classes,
        attributes,
        attribute_ids_list,
        class_ids_list,
        coordinates_per_polygon,
        coordinate_values,
    ):
        processor = MultiplePolylineToPolygon(attribute_ids_list, class_ids_list)
        with self.session() as sess:
            input_labels = make_labels(
                shapes_per_frame,
                5,
                5,
                coordinates_per_polygon,
                coordinate_values,
                classes,
                attributes,
            )

            transformed_labels = processor.process(input_labels)
            transformed_labels, expected_labels = sess.run(
                [transformed_labels, input_labels]
            )

            expected_polygon_shape = expected_labels.vertices.coordinates.dense_shape
            transformed_polygon_shape = (
                transformed_labels.vertices.coordinates.dense_shape
            )
            expected_class_shape = expected_labels.classes.dense_shape
            transformed_class_shape = transformed_labels.classes.dense_shape
            expected_class_values = expected_labels.classes.values
            transformed_class_values = transformed_labels.classes.values

            # Check whether the shapes of the returned polygons and classes are equal
            self.assertAllEqual(transformed_polygon_shape, expected_polygon_shape)
            self.assertAllEqual(transformed_class_shape, expected_class_shape)
            # Check if number of polygons is the same as input
            self.assertEqual(len(transformed_class_values), len(expected_class_values))

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        processor = MultiplePolylineToPolygon(
            attribute_id_list=[0, 1, 2], class_id_list=[0, 0, 1]
        )
        processor_dict = processor.serialize()
        deserialized_processor_dict = deserialize_tao_object(processor_dict)
        assert (
            processor._attribute_id_list
            == deserialized_processor_dict._attribute_id_list
        )
        assert processor._class_id_list == deserialized_processor_dict._class_id_list
