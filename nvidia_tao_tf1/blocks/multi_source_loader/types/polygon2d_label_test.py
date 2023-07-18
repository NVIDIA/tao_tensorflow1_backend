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
"""Unit tests for Polygon2DLabel"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    test_fixtures as fixtures,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2DWithCounts,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.polygon2d_label import (
    Polygon2DLabel,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sparse_tensor_builder import (
    SparseTensorBuilder,
)


###
# Convenience shorthands for building out the coordinate tensors
###
class C(SparseTensorBuilder):
    """Coordinates."""

    pass


class Poly(SparseTensorBuilder):
    """Polygon/Polyline."""

    pass


class Frame(SparseTensorBuilder):
    """Frame."""

    pass


class Timestep(SparseTensorBuilder):
    """Timestep."""

    pass


class Batch(SparseTensorBuilder):
    """Batch."""

    pass


class Label(SparseTensorBuilder):
    """Class Label."""

    pass


###


def _get_label(
    label_builder, label_classes_builder, label_vertices_counts, label_attributes
):
    sparse_example = label_builder.build(val_type=tf.float32)
    sparse_classes = label_classes_builder.build(val_type=tf.int32)
    sparse_counts = label_vertices_counts.build(val_type=tf.int32)
    sparse_attributes = label_attributes.build(val_type=tf.int32)

    coordinates = Coordinates2DWithCounts(
        coordinates=sparse_example,
        canvas_shape=tf.zeros(1),
        vertices_count=sparse_counts,
    )

    label = Polygon2DLabel(
        vertices=coordinates, classes=sparse_classes, attributes=sparse_attributes
    )
    return label


class Polygon2DLabelTest(tf.test.TestCase):
    def test_reshape_to_4d(self):

        # Want sparse coordinates and classes to overlap but also have coordinates without classes
        # and classes without coordinates.
        coordinates = fixtures.make_coordinates2d(
            # Examples
            [  # Frames
                [0],
                [
                    # Shapes
                    0,
                    1,
                    2,
                ],
            ],
            100,
            200,
        )
        classes = fixtures.make_tags(
            # Examples
            [[[[1]]], [[[0], [1]], [[0], [1]], [[0], [1]]]]  # Frames  # Shapes
        )

        attributes = fixtures.make_tags(
            [  # Examples
                [[[1]]],
                [  # Frames
                    [[0, 1, 3], [10]],  # Shapes
                    [[0, 1, 3], [10]],  # Shapes
                    [[0, 1, 3], [10]],  # Shapes
                ],
            ]
        )

        label = Polygon2DLabel(
            vertices=coordinates, classes=classes, attributes=attributes
        )

        # classes and attributes
        with self.cached_session():
            reshaped = label.compress_frame_dimension()

        # TODO(ehall): verify that this is preserving image, class index correlations between
        #  polygons w/sparse generator from other branch.
        self.assertAllEqual(reshaped.vertices.coordinates.dense_shape, [6, 2, 3, 2])
        self.assertAllEqual(reshaped.classes.dense_shape, [6, 2, 1])
        self.assertAllEqual(reshaped.attributes.dense_shape, [6, 2, 3])

    # Verify that empty attributes are handled properly during compression and do not raise errors.
    def test_handling_of_empty_attributes(self):
        coordinates = fixtures.make_coordinates2d(
            # Examples
            [  # Frames
                [0],
                [
                    # Shapes
                    0,
                    1,
                    2,
                ],
            ],
            100,
            200,
        )
        classes = fixtures.make_tags(
            # Examples
            [[[[1]]], [[[0], [1]], [[0], [1]], [[0], [1]]]]  # Frames  # Shapes
        )
        empty_attributes = tf.SparseTensor(
            indices=tf.zeros((0, 4), tf.int64),
            values=tf.constant([], dtype=tf.string),
            dense_shape=tf.constant((0, 0, 0, 0), dtype=tf.int64),
        )

        label = Polygon2DLabel(
            vertices=coordinates, classes=classes, attributes=empty_attributes
        )
        # classes and attributes
        with self.cached_session():
            reshaped = label.compress_frame_dimension()

        self.assertAllEqual(reshaped.vertices.coordinates.dense_shape, [6, 2, 3, 2])
        self.assertAllEqual(reshaped.classes.dense_shape, [6, 2, 1])
        self.assertAllEqual(reshaped.attributes.dense_shape, [0, 0, 0])

    def test_slice_to_last_frame(self):
        example = Batch(
            Timestep(Frame(Poly(C(1, 1), C(2, 2))), Frame(Poly(C(3, 3), C(4, 4))))
        )

        example_classes = Batch(Timestep(Frame(Label(0)), Frame(Label(1))))

        example_v_counts = Batch(Timestep(Frame(Poly(2))))

        example_attributes = Batch(
            Timestep(Frame(Poly(Label(0))), Frame(Poly(Label(1))))
        )

        example = _get_label(
            example, example_classes, example_v_counts, example_attributes
        )

        example = example.slice_to_last_frame()

        target_example = Batch(Timestep(Frame(Poly(C(3, 3), C(4, 4)))))

        target_classes = Batch(Timestep(Frame(Label(1))))

        target_v_counts = Batch(Timestep(Frame(Poly(2))))

        target_attributes = Batch(Timestep(Frame(Poly(Label(1)))))

        target_example = _get_label(
            target_example, target_classes, target_v_counts, target_attributes
        )

        self._assertSparseEqual(
            target_example.vertices.coordinates, example.vertices.coordinates
        )
        self._assertSparseEqual(target_example.classes, example.classes)
        self._assertSparseEqual(target_example.attributes, example.attributes)

    def _assertSparseEqual(self, expected, actual):
        """Assert that two sparse tensors match.

        Args:
            expected (tf.SparseTensor): Expected tensor.
            actual (tf.SparseTensor): Actual tensor.
        """
        self.assertAllEqual(expected.indices, actual.indices)
        self.assertAllEqual(expected.dense_shape, actual.dense_shape)
        self.assertAllClose(expected.values, actual.values)
