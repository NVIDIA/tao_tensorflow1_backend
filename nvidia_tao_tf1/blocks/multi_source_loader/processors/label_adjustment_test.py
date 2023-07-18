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

"""Tests for LabelAdjustment processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re


from mock import Mock
import numpy as np
from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_FIRST
from nvidia_tao_tf1.blocks.multi_source_loader.processors.label_adjustment import (
    LabelAdjustment,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Canvas2D,
    Coordinates2D,
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
    Polygon2DLabel,
    SequenceExample,
)
import nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures as fixtures
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


def make_sequence_example(coordinates):
    frames = tf.ones((1, 128, 240, 3))

    coordinates_2d = Coordinates2D(
        coordinates=coordinates, canvas_shape=Canvas2D(height=0, width=0)
    )
    polygon_label_2d = Polygon2DLabel(
        vertices=coordinates_2d,
        classes=fixtures.make_tags([[[[1]]]]),
        attributes=fixtures.make_tags([[[[1]]]]),
    )
    return SequenceExample(
        instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon_label_2d}
    )


class TestLabelAdjustment(ProcessorTestCase):
    @parameterized.expand(
        [
            [0, 2, 4, re.escape("Scale: 0 is not positive.")],
            [-1, 2, 4, re.escape("Scale: -1 is not positive.")],
            [1, -1, 4, re.escape("Translation x: -1 cannot be a negative number.")],
            [1, 2, -1, re.escape("Translation y: -1 cannot be a negative number.")],
        ]
    )
    def test_raises_on_invalid_arguments(
        self, scale, translation_x, translation_y, message
    ):
        with self.assertRaisesRegexp(ValueError, message):
            LabelAdjustment(
                scale=scale, translation_x=translation_x, translation_y=translation_y
            )

    def test_supports_channels_first(self):
        label_adjustment = LabelAdjustment()
        assert label_adjustment.supported_formats == [CHANNELS_FIRST]

    def test_does_not_compose(self):
        label_adjustment = LabelAdjustment()
        assert label_adjustment.can_compose(Mock()) is False

    def test_compose_raises(self):
        with self.assertRaises(NotImplementedError):
            label_adjustment = LabelAdjustment()
            label_adjustment.compose(Mock())

    def test_adjust_scale(self):
        frames = tf.ones((1, 128, 240, 3))
        labels = self.make_polygon_label(
            vertices=[[60, 32], [180, 32], [180, 96], [60, 96]]
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: labels}
        )
        expected_labels = self.make_polygon_label(
            vertices=[[30, 16], [90, 16], [90, 48], [30, 48]]
        )

        with self.test_session():
            label_adjustment = LabelAdjustment(scale=0.5)
            adjusted = label_adjustment.process(example)
            self.assert_labels_close(expected_labels, adjusted.labels[LABEL_MAP])

    def test_adjust_translation(self):
        frames = tf.ones((1, 128, 240, 3))
        polygon = self.make_polygon_label(
            vertices=[[60, 32], [180, 32], [180, 96], [60, 96]]
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        expected_labels = self.make_polygon_label(
            vertices=[[40, 2], [160, 2], [160, 66], [40, 66]]
        )

        with self.test_session():
            label_adjustment = LabelAdjustment(translation_x=20, translation_y=30)
            adjusted = label_adjustment.process(example)
            self.assert_labels_close(expected_labels, adjusted.labels[LABEL_MAP])

    def test_adjust_scale_and_translation(self):
        frames = tf.ones((1, 128, 240, 3))
        polygon = self.make_polygon_label(
            vertices=[[60, 32], [180, 32], [180, 96], [60, 96]]
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        expected_labels = self.make_polygon_label(
            vertices=[[10, -14], [70, -14], [70, 18], [10, 18]]
        )

        with self.test_session():
            label_adjustment = LabelAdjustment(
                scale=0.5, translation_x=20, translation_y=30
            )
            adjusted = label_adjustment.process(example)
            self.assert_labels_close(expected_labels, adjusted.labels[LABEL_MAP])

    def test_sequence_example_adjust_scale(self):
        coordinates = tf.SparseTensor(
            indices=[
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 2, 1],
                [0, 0, 0, 3, 0],
                [0, 0, 0, 3, 1],
            ],
            values=tf.constant([60, 32, 180, 32, 180, 96, 60, 96], dtype=tf.float32),
            dense_shape=[1, 1, 1, 4, 2],
        )

        sequence_example = make_sequence_example(coordinates)

        expected_coordinates = tf.SparseTensor(
            indices=coordinates.indices,
            values=tf.constant([30, 16, 90, 16, 90, 48, 30, 48]),
            dense_shape=coordinates.dense_shape,
        )

        with self.test_session():
            label_adjustment = LabelAdjustment(scale=0.5)
            adjusted = label_adjustment.process(sequence_example)
            self.assertAllClose(
                expected_coordinates.eval(),
                adjusted.labels[LABEL_MAP].vertices.coordinates.eval(),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_sequence_example_adjust_translation(self):
        coordinates = tf.SparseTensor(
            indices=[
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 2, 1],
                [0, 0, 0, 3, 0],
                [0, 0, 0, 3, 1],
            ],
            values=tf.constant([60, 32, 180, 32, 180, 96, 60, 96], dtype=tf.float32),
            dense_shape=[1, 1, 1, 4, 2],
        )

        sequence_example = make_sequence_example(coordinates)

        expected_coordinates = tf.SparseTensor(
            indices=coordinates.indices,
            values=tf.constant([40, 2, 160, 2, 160, 66, 40, 66]),
            dense_shape=coordinates.dense_shape,
        )

        with self.test_session():
            label_adjustment = LabelAdjustment(translation_x=20, translation_y=30)
            adjusted = label_adjustment.process(sequence_example)
            self.assertAllClose(
                expected_coordinates.eval(),
                adjusted.labels[LABEL_MAP].vertices.coordinates.eval(),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_sequence_example_adjust_scale_and_translation(self):
        coordinates = tf.SparseTensor(
            indices=[
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 2, 1],
                [0, 0, 0, 3, 0],
                [0, 0, 0, 3, 1],
            ],
            values=tf.constant([60, 32, 180, 32, 180, 96, 60, 96], dtype=tf.float32),
            dense_shape=[1, 1, 1, 4, 2],
        )
        sequence_example = make_sequence_example(coordinates)
        expected_coordinates = tf.SparseTensor(
            indices=coordinates.indices,
            values=tf.constant([10, -14, 70, -14, 70, 18, 10, 18]),
            dense_shape=coordinates.dense_shape,
        )

        with self.test_session():
            label_adjustment = LabelAdjustment(
                scale=0.5, translation_x=20, translation_y=30
            )
            adjusted = label_adjustment.process(sequence_example)
            self.assertAllClose(
                expected_coordinates.eval(),
                adjusted.labels[LABEL_MAP].vertices.coordinates.eval(),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_sequence_example_adjust_scale_and_translation_empty(self):
        empty_indices = tf.zeros((0, 4), tf.int64)
        empty_values = tf.constant([], dtype=tf.float32)
        empty_dense_shape = [0, 0, 0, 0]

        empty_coordinates = tf.SparseTensor(
            indices=empty_indices, values=empty_values, dense_shape=empty_dense_shape
        )
        sequence_example = make_sequence_example(empty_coordinates)

        with self.test_session():
            label_adjustment = LabelAdjustment(
                scale=0.5, translation_x=20, translation_y=30
            )
            adjusted = label_adjustment.process(sequence_example)
            adjusted_coordinates = adjusted.labels[LABEL_MAP].vertices.coordinates
            adjusted_indices = adjusted_coordinates.indices
            adjusted_values = adjusted_coordinates.values
            adjusted_dense_shape = adjusted_coordinates.dense_shape
            # assertAllEqual seems to have problems with empty tensors so have to revert to this
            # manual checking.
            self.assertTrue(
                np.array_equal(empty_indices.eval(), adjusted_indices.eval())
            )
            self.assertTrue(np.array_equal(empty_values.eval(), adjusted_values.eval()))
            self.assertAllEqual(empty_dense_shape, adjusted_dense_shape.eval())

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        label_adjustment = LabelAdjustment(
            scale=0.5, translation_x=20, translation_y=30
        )
        label_adjustment_dict = label_adjustment.serialize()
        deserialized_label_adjustment = deserialize_tao_object(label_adjustment_dict)
        assert label_adjustment._scale == deserialized_label_adjustment._scale
        assert (
            label_adjustment._translation_x
            == deserialized_label_adjustment._translation_x
        )
        assert (
            label_adjustment._translation_y
            == deserialized_label_adjustment._translation_y
        )
