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

"""Tests for Scale processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.scale import Scale
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import Images2D
from nvidia_tao_tf1.blocks.multi_source_loader.types import Images2DReference
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    test_fixtures as fixtures,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    SequenceExample,
    TransformedExample,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestScale(ProcessorTestCase):
    @parameterized.expand(
        [
            [-10, 20, re.escape("Scale.height (-10) is not positive.")],
            [10, -20, re.escape("Scale.width (-20) is not positive.")],
        ]
    )
    def test_raises_on_invalid_bounds(self, height, width, message):
        with self.assertRaisesRegexp(ValueError, message):
            Scale(height=height, width=width)

    @parameterized.expand([[1, 1], [1, 2], [2, 1]])
    def test_valid_bounds_do_not_raise(self, height, width):
        Scale(height=height, width=width)

    def test_scales_down(self):
        frames = tf.ones((1, 128, 240, 3))
        labels = self.make_polygon_label([[0.5, 0.0], [1.0, 1.0], [0.0, 1.0]])
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: labels}
        )

        expected_frames = tf.ones((1, 64, 120, 3))
        expected_labels = self.make_polygon_label([[0.25, 0.0], [0.5, 0.5], [0.0, 0.5]])

        with self.test_session():
            scale = Scale(height=64, width=120)
            scaled = scale.process(example)

            self.assertAllClose(
                expected_frames.eval(), scaled.instances[FEATURE_CAMERA].eval()
            )
            self.assert_labels_close(expected_labels, scaled.labels[LABEL_MAP])

    @parameterized.expand(
        [
            [Scale(height=96, width=180), Scale(height=32, width=60)],
            [Scale(height=97, width=181), Scale(height=31, width=61)],
        ]
    )
    def test_composite_of_two_processors_outputs_same(self, first, second):
        frames = tf.ones((1, 128, 240, 3))
        labels = self.make_polygon_label([[0.5, 0.0], [1.0, 1.0], [0.0, 1.0]])
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: labels}
        )

        composite = first.compose(second)

        with self.test_session():
            first_processed = first.process(example)
            second_processed = second.process(first_processed)
            composite_processed = composite.process(example)

            self.assertEqual(
                second_processed.instances[FEATURE_CAMERA].get_shape(),
                composite_processed.instances[FEATURE_CAMERA].get_shape(),
            )
            self.assertAllClose(
                second_processed.instances[FEATURE_CAMERA].eval(),
                composite_processed.instances[FEATURE_CAMERA].eval(),
            )

            self.assert_labels_close(
                second_processed.labels[LABEL_MAP],
                composite_processed.labels[LABEL_MAP],
            )

    @parameterized.expand([[False], [True]])
    def test_unbatched_scale(self, use_images2d_reference):
        original_width = 80
        original_height = 64

        example = fixtures.make_example(
            width=original_width,
            height=original_height,
            use_images2d_reference=use_images2d_reference,
        )

        new_width = original_width * 2
        new_height = original_height * 2
        augmentation = Scale(width=new_width, height=new_height)

        processed = augmentation.process(example)
        assert type(processed) == TransformedExample

        processed = tf.compat.v1.Session().run(processed)

        expected_canvas_width = (1, 1, new_width)
        expected_canvas_height = (1, 1, new_height)

        # Check transformation canvas shape.
        assert (
            processed.transformation.canvas_shape.width.shape == expected_canvas_width
        )
        assert (
            processed.transformation.canvas_shape.height.shape == expected_canvas_height
        )

        # Check color transformation matrix.
        expected_color_matrix = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ]
        )
        np.testing.assert_equal(
            processed.transformation.color_transform_matrix, expected_color_matrix
        )

        # Check spatial transformation matrix. Note that transformation happen from output
        # to input, that's why the diagonal entries are 0.5 instead of 2.0.
        expected_spatial_matrix = np.array(
            [[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]]
        )
        np.testing.assert_equal(
            processed.transformation.spatial_transform_matrix, expected_spatial_matrix
        )

    @parameterized.expand([[False, False], [True, False], [False, True], [True, True]])
    def test_batched_scale(self, dynamic_batch_size, use_images2d_reference):
        feed_dict = {}
        expected_batch_size = 3
        if dynamic_batch_size:
            batch_size = tf.compat.v1.placeholder(dtype=tf.int32)
            feed_dict = {batch_size: expected_batch_size}
        else:
            batch_size = expected_batch_size

        shapes_per_frame = []
        coordinate_values = []
        for i in range(expected_batch_size):
            shapes_per_frame.append([1])
            coordinate_values.append([float(i), 0.0])
            coordinate_values.append([float(i) + 10.0, 5.0])
            coordinate_values.append([float(i) + 8.0, 9.0])

        original_width = 80
        original_height = 64

        new_width = original_width * 2
        new_height = original_height * 2

        expected_shape = (1, 3, new_height, new_width)
        expected_canvas_width = (1, new_width)
        expected_canvas_height = (1, new_height)
        if batch_size is not None:
            expected_shape = (expected_batch_size,) + expected_shape
            expected_canvas_height = (expected_batch_size,) + expected_canvas_height
            expected_canvas_width = (expected_batch_size,) + expected_canvas_width

        example = fixtures.make_example(
            width=original_width,
            height=original_height,
            example_count=batch_size,
            shapes_per_frame=shapes_per_frame,
            coordinates_per_polygon=3,
            coordinate_values=coordinate_values,
            use_images2d_reference=use_images2d_reference,
        )
        original = tf.compat.v1.Session().run(example, feed_dict=feed_dict)

        augmentation = Scale(width=new_width, height=new_height)

        # Compute transformations.
        processed = augmentation.process(example)
        assert type(processed) == TransformedExample

        # Peek transformation.
        transformation = tf.compat.v1.Session().run(
            processed.transformation, feed_dict=feed_dict
        )

        # Check transformation canvas shape.
        assert transformation.canvas_shape.width.shape == expected_canvas_width
        assert transformation.canvas_shape.height.shape == expected_canvas_height

        # Check color transformation matrix.
        expected_color_matrix = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ]
        )
        expected_color_matrix = np.tile(
            expected_color_matrix, [expected_batch_size, 1, 1]
        )
        np.testing.assert_equal(
            transformation.color_transform_matrix, expected_color_matrix
        )

        # Check spatial transformation matrix. Note that transformation happen from output
        # to input, that's why the diagonal entries are 0.5 instead of 2.0.
        expected_spatial_matrix = np.array(
            [[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]]
        )
        expected_spatial_matrix = np.tile(
            expected_spatial_matrix, [expected_batch_size, 1, 1]
        )
        np.testing.assert_equal(
            transformation.spatial_transform_matrix, expected_spatial_matrix
        )

        # Apply transformations to examples.
        processed = processed()
        assert type(processed) == SequenceExample

        processed = tf.compat.v1.Session().run(processed, feed_dict=feed_dict)

        feature_camera = processed.instances[FEATURE_CAMERA]
        if type(feature_camera) == Images2D:
            # Check image shape.
            assert feature_camera.images.shape == expected_shape
            assert feature_camera.canvas_shape.width.shape == expected_canvas_width
            assert feature_camera.canvas_shape.height.shape == expected_canvas_height
        else:
            # Images2DReference hasn't been processed yet.
            assert type(feature_camera) == Images2DReference
            assert feature_camera.input_width == original_width
            assert feature_camera.input_height == original_height

        # Check vertices canvas shape.
        vertices_canvas_shape = processed.labels[LABEL_MAP].vertices.canvas_shape
        assert vertices_canvas_shape.width.shape == expected_canvas_width
        assert vertices_canvas_shape.height.shape == expected_canvas_height

        # Check that processed coords are 2x the original ones.
        original_coordinates = original.labels[LABEL_MAP].vertices.coordinates
        processed_coordinates = processed.labels[LABEL_MAP].vertices.coordinates
        np.testing.assert_equal(
            original_coordinates.values * 2.0, processed_coordinates.values
        )
        np.testing.assert_equal(
            original_coordinates.indices, processed_coordinates.indices
        )
        np.testing.assert_equal(
            original_coordinates.dense_shape, processed_coordinates.dense_shape
        )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        augmentation = Scale(width=10, height=12)
        augmentation_dict = augmentation.serialize()
        deserialized_augmentation = deserialize_tao_object(augmentation_dict)
        self.assertEqual(
            str(augmentation._transform), str(deserialized_augmentation._transform)
        )
