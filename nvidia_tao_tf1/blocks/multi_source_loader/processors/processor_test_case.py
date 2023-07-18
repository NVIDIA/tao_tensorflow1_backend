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

"""Base class for unittests that test processors implemented using TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Coordinates2D,
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
    Polygon2DLabel,
    PolygonLabel,
)
import nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures as fixtures


class ProcessorTestCase(tf.test.TestCase):
    """
    Base class for unit tests testing Processors.

    Processors are tested using this tf.test.TestCase derived class because of the convenient
    test helpers provided by that class. This class adds commonly used methods to avoid repetition.
    """

    def make_polygon_label(self, vertices, class_id=0):
        """Create a PolygonLabel.

        Args:
            vertices (list of `float`): Vertices that make up the polygon.
            class_id (int): Identifier for the class that the label represents.
        """
        polygons = tf.constant(vertices, dtype=tf.float32)
        vertices_per_polygon = tf.constant([len(vertices)], dtype=tf.int32)
        class_ids_per_polygon = tf.constant([class_id], dtype=tf.int32)
        attributes_per_polygon = tf.constant([1], dtype=tf.int32)
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

    def make_empty_polygon2d_labels(self, *args):
        """Create empty Polygon2DLabel.

        Args:
            *args (list of example dims): A set of example dims, e.g. (<batch>, <seq>).
        """
        batch_size = args[0] if args else 0
        list_args = list(args)
        empty_polygon2d = Polygon2DLabel(
            vertices=Coordinates2D(
                coordinates=tf.SparseTensor(
                    indices=tf.reshape(tf.constant([], tf.int64), [0, len(args) + 3]),
                    values=tf.constant([], tf.float32),
                    dense_shape=tf.constant(list_args + [0, 0, 0], tf.int64),
                ),
                canvas_shape=fixtures.make_canvas2d(batch_size, 960, 560),
            ),
            classes=tf.SparseTensor(
                indices=tf.reshape(tf.constant([], tf.int64), [0, len(args) + 2]),
                values=tf.constant([], tf.int32),
                dense_shape=tf.constant(list_args + [0, 0], tf.int64),
            ),
            attributes=tf.SparseTensor(
                indices=tf.reshape(tf.constant([], tf.int64), [0, len(args) + 2]),
                values=tf.constant([], tf.int32),
                dense_shape=tf.constant(list_args + [0, 0], tf.int64),
            ),
        )
        return empty_polygon2d

    def make_example_128x240(self):
        """Return Example with a triangular label."""
        frames = tf.ones((1, 128, 240, 3))
        labels = self.make_polygon_label([[120, 0.0], [240, 128], [0.0, 128]])
        return Example(instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: labels})

    def assert_labels_close(self, expected, actual):
        """Assert that labels match.

        Args:
            expected (PolygonLabel): Expected label.
            actual (PolygonLabel): Actual label.s
        """
        # Polygons are in pixel coordinates - milli-pixel tolerance seems acceptable
        self.assertAllClose(
            expected.polygons.eval(), actual.polygons.eval(), rtol=1e-3, atol=1e-3
        )

        self.assertAllEqual(
            expected.vertices_per_polygon.eval(), actual.vertices_per_polygon.eval()
        )

        self.assertAllEqual(
            expected.class_ids_per_polygon.eval(), actual.class_ids_per_polygon.eval()
        )

        self.assertAllEqual(
            expected.attributes_per_polygon.eval(), actual.attributes_per_polygon.eval()
        )

        self.assertEqual(
            expected.polygons_per_image.eval(), actual.polygons_per_image.eval()
        )

    def assertSparseEqual(self, expected, actual):
        """Assert that two sparse tensors match.

        Args:
            expected (tf.SparseTensor): Expected tensor.
            actual (tf.SparseTensor): Actual tensor.
        """
        self.assertAllEqual(expected.indices, actual.indices)
        self.assertAllEqual(expected.dense_shape, actual.dense_shape)
        self.assertAllClose(expected.values, actual.values)
