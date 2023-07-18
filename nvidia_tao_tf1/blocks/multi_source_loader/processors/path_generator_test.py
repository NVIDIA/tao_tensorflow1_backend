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

"""Unitests for the path generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.frame_shape import FrameShape
from nvidia_tao_tf1.blocks.multi_source_loader.processors.path_generator import (
    PathGenerator,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.priors_generator import (
    PriorsGenerator,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Coordinates2DWithCounts,
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
    Polygon2DLabel,
    PolygonLabel,
)
import nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures as fixtures
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestPathGenerator(ProcessorTestCase):
    def _create_path_generator(self, npath_attributes=0, edges_per_path=2):
        class_name_to_id = {"drivepath 0": 0, "drivepath 1": 1, "drivepath -1": 2}
        equidistant_interpolation = False
        return PathGenerator(
            nclasses=3,
            class_name_to_id=class_name_to_id,
            equidistant_interpolation=equidistant_interpolation,
            npath_attributes=npath_attributes,
            edges_per_path=edges_per_path,
        )

    def _make_multi_polygon_label(
        self,
        vertices,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
    ):
        """Create a PolygonLabel.

        Args:
            vertices (list of `float`): Vertices that make up the polygon.
            class_id (int): Identifier for the class that the label represents.
        """
        polygons = tf.constant(vertices, dtype=tf.float32)
        vertices_per_polygon = tf.constant(vertices_per_polyline, dtype=tf.int32)
        class_ids_per_polygon = tf.constant(class_ids_per_polyline, dtype=tf.int32)
        attributes_per_polygon = tf.constant(attributes_per_polyline, dtype=tf.int32)
        polygons_per_image = tf.constant([1], dtype=tf.int32)
        attributes = (tf.constant([], tf.int32),)
        attribute_count_per_polygon = tf.constant([], tf.int32)
        return PolygonLabel(
            polygons=polygons,
            vertices_per_polygon=vertices_per_polygon,
            class_ids_per_polygon=class_ids_per_polygon,
            attributes_per_polygon=attributes_per_polygon,
            polygons_per_image=polygons_per_image,
            attributes=attributes,
            attribute_count_per_polygon=attribute_count_per_polygon,
        )

    def _make_coordinates2d(self, polylines, vertices_per_polyline):
        frame_shape_count = len(vertices_per_polyline)
        indices = []
        for shape_index in range(frame_shape_count):
            for vertex_index in range(vertices_per_polyline[shape_index]):
                for coordinate_index in [0, 1]:
                    indices.append([0, 0, shape_index, vertex_index, coordinate_index])
        dense_coordinates = tf.constant(polylines, dtype=tf.float32)
        sparse_indices = tf.constant(indices, dtype=tf.int64)

        dense_shape = tf.constant(
            (1, 1, frame_shape_count, max(vertices_per_polyline), 2), dtype=tf.int64
        )
        sparse_coordinates = tf.SparseTensor(
            indices=sparse_indices,
            values=tf.reshape(dense_coordinates, (-1,)),
            dense_shape=dense_shape,
        )
        vertices_count = tf.SparseTensor(
            indices=tf.constant(
                [[0, 0, j] for j in range(frame_shape_count)], dtype=tf.int64
            ),
            values=tf.constant(vertices_per_polyline),
            dense_shape=tf.constant([1, 1, frame_shape_count], dtype=tf.int64),
        )
        return Coordinates2DWithCounts(
            coordinates=sparse_coordinates,
            canvas_shape=fixtures.make_canvas2d(1, 100, 100),
            vertices_count=vertices_count,
        )

    @parameterized.expand(
        [
            [np.array([-1, 1, -1, 1]), 0, np.array([-1, 1, -1, 1])],
            [np.array([0, 0, 0]), 0, np.array([0, 0, 0])],
            [np.array([0, 0, 0, 0]), 1, np.array([0, 0, 0, 0])],
            [
                np.array([-1, 3, 1, 3, 2, -1, 2, 1, 0, 0]),
                1,
                np.array([-1, 1, -1, 1, 0, 3, 3, 2, 2, 0]),
            ],
            [
                np.array([0, 0, -1, 2, 0, 0, 1, 2, 3, -1, 3, 1]),
                1,
                np.array([0, -1, 0, 1, -1, 1, 0, 2, 0, 2, 3, 3]),
            ],
        ]
    )
    def test_reorder_attributes(
        self, attributes, npath_attributes, expected_attributes
    ):
        path_generator = self._create_path_generator()
        attributes = tf.constant(attributes)
        reordered_attributes = path_generator._reorder_attributes(
            attributes, npath_attributes
        )
        with self.test_session():
            reordered_attributes_np = reordered_attributes.eval()
            self.assertAllEqual(reordered_attributes_np, expected_attributes)

    def test_constructor_sets_output_classes(self):
        path_generator = self._create_path_generator()
        assert path_generator.class_name_to_id == {
            "drivepath 0": 0,
            "drivepath 1": 1,
            "drivepath -1": 2,
        }

    _POLYLINES = np.array(
        [
            [30, 60],
            [30, 80],
            [90, 60],
            [90, 80],
            [20, 30],
            [20, 50],
            [20, 60],
            [20, 80],
            [40, 30],
            [40, 50],
            [40, 60],
            [40, 80],
            [40, 100],
            [60, 100],
            [80, 100],
            [40, 60],
        ],
        dtype=np.float32,
    )

    @parameterized.expand(
        [
            [
                _POLYLINES[:4],
                np.array([2, 2]),
                np.array([0, 0]),
                np.array([-1, 1]),
                np.array([4, 16]),
                np.array([3]),
            ],
            [
                _POLYLINES[:12],
                np.array([2, 2, 4, 4]),
                np.array([0, 0, 1, 1]),
                np.array([-1, 1, -1, 1]),
                np.array([4, 16]),
                np.array([2, 3]),
            ],
            [
                _POLYLINES[:16],
                np.array([2, 2, 4, 3, 2, 3]),
                np.array([0, 0, 1, 1, 2, 2]),
                np.array([-1, 1, -1, 1, -1, 1]),
                np.array([4, 16]),
                np.array([0, 2]),
            ],
        ]
    )
    def test_process_with_varying_polylines(
        self,
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_ids_per_polyline,
        expected_shapes,
        expected_valid_paths,
    ):
        height = 100
        width = 100
        frames = tf.ones((1, 3, height, width))
        polygon = self._make_multi_polygon_label(
            polylines,
            vertices_per_polyline,
            class_ids_per_polyline,
            attributes_ids_per_polyline,
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        priors_generator = PriorsGenerator(
            npoint_priors=1,
            nlinear_priors=0,
            points_per_prior=3,
            prior_threshold=0.2,
            feature_map_sizes=[(2, 2)],
            image_height=height,
            image_width=width,
        )
        path_generator = self._create_path_generator()
        with self.test_session():
            path_generated = path_generator.encode_dense(example, priors_generator)
            targets = path_generated.labels[LABEL_MAP].eval()
            # Make sure frames are untouched.
            self.assertAllEqual(
                path_generated.instances[FEATURE_CAMERA].eval(), frames.eval()
            )
            self.assertAllEqual(targets.shape, expected_shapes)
            # `targets` has as many rows as number of priors.
            # If one of the priors is assigned to a label,
            # some of the values of that row are > 0.
            # If the prior is not assigned to the label, all the values from that row are <= 0.
            priors_assigned_to_labels = np.unique(np.where(targets > 0)[0])
            self.assertAllEqual(priors_assigned_to_labels, expected_valid_paths)

    @parameterized.expand(
        [
            [
                _POLYLINES[:4],
                np.array([2, 2]),
                np.array([0, 0]),
                np.array([-1, 2, 1, 2]),
                np.array([4, 17]),
                np.array([3]),
            ],
            [
                _POLYLINES[:12],
                np.array([2, 2, 4, 4]),
                np.array([0, 0, 1, 1]),
                np.array([3, -1, 1, 3, -1, 2, 2, 1]),
                np.array([4, 17]),
                np.array([2, 3]),
            ],
            [
                _POLYLINES[:16],
                np.array([2, 2, 4, 3, 2, 3]),
                np.array([0, 0, 1, 1, 2, 2]),
                np.array([-1, 2, 1, 2, 3, -1, 1, 3, -1, 3, 1, 3]),
                np.array([4, 17]),
                np.array([0, 2]),
            ],
        ]
    )
    def test_process_with_varying_polylines_and_path_attributes(
        self,
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_ids_per_polyline,
        expected_shapes,
        expected_valid_paths,
    ):
        height = 100
        width = 100
        frames = tf.ones((1, 3, height, width))
        polygon = self._make_multi_polygon_label(
            polylines,
            vertices_per_polyline,
            class_ids_per_polyline,
            attributes_ids_per_polyline,
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        priors_generator = PriorsGenerator(
            npoint_priors=1,
            nlinear_priors=0,
            points_per_prior=3,
            prior_threshold=0.2,
            feature_map_sizes=[(2, 2)],
            image_height=height,
            image_width=width,
        )
        path_generator = self._create_path_generator(npath_attributes=1)
        with self.test_session():
            path_generated = path_generator.encode_dense(example, priors_generator)
            targets = path_generated.labels[LABEL_MAP].eval()
            # Make sure frames are untouched.
            self.assertAllEqual(
                path_generated.instances[FEATURE_CAMERA].eval(), frames.eval()
            )
            self.assertAllEqual(targets.shape, expected_shapes)
            # `targets` has as many rows as number of priors.
            # If one of the priors is assigned to a label,
            # some of the values of that row are > 0.
            # If the prior is not assigned to the label, all the values from that row are <= 0.
            priors_assigned_to_labels = np.unique(np.where(targets > 0)[0])
            self.assertAllEqual(priors_assigned_to_labels, expected_valid_paths)

    @parameterized.expand(
        [
            [1, 0, 3, np.array([4, 16]), np.array([3])],
            [1, 0, 2, np.array([4, 12]), np.array([])],
            [1, 1, 3, np.array([8, 16]), np.array([3])],
        ]
    )
    def test_process_with_varying_priors(
        self,
        npoint_priors,
        nlinear_priors,
        points_per_prior,
        expected_shapes,
        expected_valid_paths,
    ):
        height = 100
        width = 100
        frames = tf.ones((1, 3, height, width))

        polylines = np.array([[30, 60], [30, 80], [90, 60], [90, 80]])
        vertices_per_polyline = np.array([2, 2])
        class_ids_per_polyline = np.array([0, 0])
        attributes_ids_per_polyline = np.array([-1, 1])
        polygon = self._make_multi_polygon_label(
            polylines,
            vertices_per_polyline,
            class_ids_per_polyline,
            attributes_ids_per_polyline,
        )

        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        priors_generator = PriorsGenerator(
            npoint_priors=npoint_priors,
            nlinear_priors=nlinear_priors,
            points_per_prior=points_per_prior,
            prior_threshold=0.2,
            feature_map_sizes=[(2, 2)],
            image_height=height,
            image_width=width,
        )
        path_generator = self._create_path_generator()
        with self.test_session():
            path_generated = path_generator.encode_dense(example, priors_generator)
            targets = path_generated.labels[LABEL_MAP].eval()
            self.assertAllEqual(
                path_generated.instances[FEATURE_CAMERA].eval(), frames.eval()
            )
            self.assertAllEqual(targets.shape, expected_shapes)
            # Get the rows of the priors assigned to the labels.
            priors_assigned_to_labels = np.unique(np.where(targets > 0)[0])
            self.assertAllEqual(priors_assigned_to_labels, expected_valid_paths)

    @parameterized.expand(
        [
            [[[1]], [1], [0]],
            [[[1, 1]], [1], [0]],
            [[[1], [2]], [1], [0]],
            [[[1, 2], [2, 3]], [1], [0]],
            [[[1, 2, 3], [4, 5, 6]], [1], [0]],
        ]
    )
    def test_encode_with_triangle_shapes(
        self, shapes_per_frame, shape_classes, shape_attributes
    ):
        nclasses = 3
        height = 100
        width = 100
        example_count = len(shapes_per_frame)
        max_frame_count = max(
            [len(frames_per_example) for frames_per_example in shapes_per_frame]
        )
        priors_generator = PriorsGenerator(
            npoint_priors=1,
            nlinear_priors=8,
            points_per_prior=2,
            prior_threshold=0.1,
            feature_map_sizes=[(2, 2)],
            image_height=height,
            image_width=width,
        )
        expected_shape = (
            example_count,
            max_frame_count,
            priors_generator.nall_priors,
            priors_generator.points_per_prior * 4 + nclasses + 1,
        )
        path_generator = self._create_path_generator()
        polygon = fixtures.make_polygon2d_label(
            shapes_per_frame=shapes_per_frame,
            shape_classes=shape_classes,
            shape_attributes=shape_attributes,
            height=height,
            width=width,
        )

        with self.test_session():
            path_generated = path_generator.encode_sparse(
                labels2d=polygon,
                priors_generator=priors_generator,
                image_shape=FrameShape(height=504, width=960, channels=3),
            )
            self.assertAllEqual(expected_shape, path_generated.shape)

    _POLYLINES = np.array(
        [
            [30, 60],
            [30, 80],
            [90, 60],
            [90, 80],
            [20, 30],
            [20, 50],
            [20, 60],
            [20, 80],
            [40, 30],
            [40, 50],
            [40, 60],
            [40, 80],
            [40, 100],
            [60, 100],
            [80, 100],
            [40, 60],
        ],
        dtype=np.float32,
    )

    @parameterized.expand(
        [
            [
                _POLYLINES[:4],
                [2, 2],
                [[[[0], [0]]]],
                [[[[-1], [1]]]],
                2,
                np.array([1, 1, 4, 16]),
                np.array([3]),
            ],
            [
                _POLYLINES[:12],
                [2, 2, 4, 4],
                [[[[0], [0], [1], [1]]]],
                [[[[-1], [1], [-1], [1]]]],
                2,
                np.array([1, 1, 4, 16]),
                np.array([2, 3]),
            ],
            [
                _POLYLINES[:16],
                [2, 2, 4, 3, 2, 3],
                [[[[0], [0], [1], [1], [2], [2]]]],
                [[[[-1], [1], [-1], [1], [-1], [1]]]],
                2,
                np.array([1, 1, 4, 16]),
                np.array([0, 2]),
            ],
            [
                np.concatenate((_POLYLINES[:16], _POLYLINES[:9]), axis=0),
                [2, 2, 4, 3, 2, 3, 3, 3, 3],
                [[[[0], [0], [1], [1], [2], [2], [0], [1], [2]]]],
                [[[[-1], [1], [-1], [1], [-1], [1], [2], [2], [2]]]],
                3,
                np.array([1, 1, 4, 22]),
                np.array([0, 2]),
            ],
        ]
    )
    def test_encode_with_varying_polylines_and_edges_per_path(
        self,
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_ids_per_polyline,
        edges_per_path,
        expected_shapes,
        expected_valid_paths,
    ):
        polygon = Polygon2DLabel(
            vertices=self._make_coordinates2d(polylines, vertices_per_polyline),
            classes=fixtures.make_tags(class_ids_per_polyline),
            attributes=fixtures.make_tags(attributes_ids_per_polyline),
        )

        priors_generator = PriorsGenerator(
            npoint_priors=1,
            nlinear_priors=0,
            points_per_prior=3,
            prior_threshold=0.2,
            feature_map_sizes=[(2, 2)],
            image_height=100,
            image_width=100,
        )
        path_generator = self._create_path_generator(edges_per_path=edges_per_path)
        with self.test_session():
            path_generated = path_generator.encode_sparse(
                labels2d=polygon,
                priors_generator=priors_generator,
                image_shape=FrameShape(height=504, width=960, channels=3),
            )
            targets = path_generated.eval()
            self.assertAllEqual(targets.shape, expected_shapes)
            priors_assigned_to_labels = np.unique(np.where(targets > 0)[2])
            self.assertAllEqual(priors_assigned_to_labels, expected_valid_paths)

    @parameterized.expand(
        [
            [1, 0, 3, np.array([1, 1, 4, 16]), np.array([3])],
            [1, 0, 2, np.array([1, 1, 4, 12]), np.array([])],
            [1, 1, 3, np.array([1, 1, 8, 16]), np.array([3])],
        ]
    )
    def test_encode_with_varying_priors(
        self,
        npoint_priors,
        nlinear_priors,
        points_per_prior,
        expected_shapes,
        expected_valid_paths,
    ):
        polygon = Polygon2DLabel(
            vertices=self._make_coordinates2d(
                [[30, 60], [30, 80], [90, 60], [90, 80]], [2, 2]
            ),
            classes=fixtures.make_tags([[[[0], [0]]]]),
            attributes=fixtures.make_tags([[[[-1], [1]]]]),
        )

        priors_generator = PriorsGenerator(
            npoint_priors=npoint_priors,
            nlinear_priors=nlinear_priors,
            points_per_prior=points_per_prior,
            prior_threshold=0.2,
            feature_map_sizes=[(2, 2)],
            image_height=100,
            image_width=100,
        )
        path_generator = self._create_path_generator()
        with self.test_session():
            path_generated = path_generator.encode_sparse(
                labels2d=polygon,
                priors_generator=priors_generator,
                image_shape=FrameShape(height=504, width=960, channels=3),
            )
            targets = path_generated.eval()

            self.assertAllEqual(targets.shape, expected_shapes)
            priors_assigned_to_labels = np.unique(np.where(targets > 0)[2])

            self.assertAllEqual(priors_assigned_to_labels, expected_valid_paths)

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        path_generator = self._create_path_generator()
        path_generator_dict = path_generator.serialize()
        deserialized_path_generator = deserialize_tao_object(path_generator_dict)
        self.assertEqual(path_generator.nclasses, deserialized_path_generator.nclasses)
        self.assertEqual(
            path_generator.class_name_to_id,
            deserialized_path_generator.class_name_to_id,
        )
        self.assertEqual(
            path_generator._equidistant_interpolation,
            deserialized_path_generator._equidistant_interpolation,
        )
        self.assertEqual(
            path_generator._path_priors, deserialized_path_generator._path_priors
        )
        self.assertEqual(
            path_generator._prior_assignment_constraint,
            deserialized_path_generator._prior_assignment_constraint,
        )
        self.assertEqual(
            path_generator._using_invalid_path_class,
            deserialized_path_generator._using_invalid_path_class,
        )
        self.assertEqual(
            path_generator.npath_attributes,
            deserialized_path_generator.npath_attributes,
        )
        self.assertEqual(
            path_generator._edges_per_path, deserialized_path_generator._edges_per_path
        )
