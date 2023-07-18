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

import re
from parameterized import parameterized

from nvidia_tao_tf1.blocks.multi_source_loader.processors.polygon_rasterizer import (
    PolygonRasterizer,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
import nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures as fixtures
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestPolygonRasterizer(ProcessorTestCase):
    @parameterized.expand(
        [
            [0, 1, "height: 0 is not a positive number."],
            [1, 0, "width: 0 is not a positive number."],
        ]
    )
    def test_assertions_fail_for_invalid_arguments(
        self, height, width, expected_message
    ):
        with self.assertRaisesRegexp(ValueError, re.escape(expected_message)):
            PolygonRasterizer(
                height=height, width=width, one_hot=True, binarize=True, nclasses=1
            )

    def test_empty_polygon2d_labels(self):
        new_height = 252
        new_width = 480
        batch_size = 32
        frame_size = 1
        class_count = 2
        expected_shape = (
            batch_size,
            frame_size,
            class_count + 1,
            new_height,
            new_width,
        )
        empty_polygon2d = self.make_empty_polygon2d_labels(batch_size, frame_size)
        processor = PolygonRasterizer(
            height=new_height,
            width=new_width,
            one_hot=True,
            binarize=True,
            nclasses=class_count,
        )
        with self.session() as sess:
            rasterized = processor.process(empty_polygon2d)
            self.assertEqual(expected_shape, rasterized.shape)

            rasters = sess.run(rasterized)
            self.assertEqual(expected_shape, rasters.shape)

    @parameterized.expand(
        [
            [[[1]]],
            [[[1, 1]]],
            [[[1], [2]]],
            [[[1, 2], [2, 3]]],
            [[[1, 2, 3], [4, 5, 6]]],
        ]
    )
    def test_rasterizes_polygon2d_labels(self, shapes_per_frame):
        new_height = 252
        new_width = 480
        class_count = 2
        example_count = len(shapes_per_frame)
        max_frame_count = max(
            [len(frames_per_example) for frames_per_example in shapes_per_frame]
        )
        expected_shape = (
            example_count,
            max_frame_count,
            class_count + 1,
            new_height,
            new_width,
        )

        processor = PolygonRasterizer(
            height=new_height,
            width=new_width,
            one_hot=True,
            binarize=True,
            nclasses=class_count,
        )

        with self.session() as sess:
            polygon = fixtures.make_polygon2d_label(
                shapes_per_frame=shapes_per_frame,
                shape_classes=[1],
                shape_attributes=[0],
                height=new_height,
                width=new_width,
            )
            rasterized = processor.process(polygon)
            self.assertEqual(expected_shape, rasterized.shape)

            rasters = sess.run(rasterized)
            self.assertEqual(expected_shape, rasters.shape)

    @parameterized.expand(
        [
            [[[1]]],
            [[[1, 1]]],
            [[[1], [2]]],
            [[[1, 2], [2, 3]]],
            [[[1, 2, 3], [4, 5, 6]]],
        ]
    )
    def test_rasterizes_polygon2d_labels_not_one_hot(self, shapes_per_frame):
        new_height = 252
        new_width = 480
        class_count = 2
        example_count = len(shapes_per_frame)
        max_frame_count = max(
            [len(frames_per_example) for frames_per_example in shapes_per_frame]
        )
        expected_shape = (example_count, max_frame_count, 1, new_height, new_width)

        processor = PolygonRasterizer(
            height=new_height,
            width=new_width,
            one_hot=False,
            binarize=True,
            nclasses=class_count,
        )

        with self.session() as sess:
            polygon = fixtures.make_polygon2d_label(
                shapes_per_frame=shapes_per_frame,
                shape_classes=[1],
                shape_attributes=[0],
                height=new_height,
                width=new_width,
            )
            rasterized = processor.process(polygon)
            self.assertEqual(expected_shape, rasterized.shape)

            rasters = sess.run(rasterized)
            self.assertEqual(expected_shape, rasters.shape)

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        processor = PolygonRasterizer(
            height=10, width=12, one_hot=False, binarize=True, nclasses=2
        )
        processor_dict = processor.serialize()
        deserialized_processor = deserialize_tao_object(processor_dict)
        self.assertEqual(
            processor._rasterize.height, deserialized_processor._rasterize.height
        )
        self.assertEqual(
            processor._rasterize.width, deserialized_processor._rasterize.width
        )
        self.assertEqual(processor.converters, deserialized_processor.converters)
