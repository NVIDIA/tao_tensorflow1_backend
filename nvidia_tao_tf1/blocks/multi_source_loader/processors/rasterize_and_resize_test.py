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

"""Tests for RasterizeAndResize processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from mock import call, Mock, patch

from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_LAST
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.rasterize_and_resize import (
    RasterizeAndResize,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestRasterizeAndResize(ProcessorTestCase):
    @parameterized.expand(
        [
            [
                0,
                1,
                tf.image.ResizeMethod.BILINEAR,
                "height: 0 is not a positive number.",
            ],
            [
                1,
                0,
                tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                "width: 0 is not a positive number.",
            ],
            [1, 1, "that", "Unrecognized resize_method: 'that'."],
        ]
    )
    def test_assertions_fail_for_invalid_arguments(
        self, height, width, method, expected_message
    ):
        with self.assertRaisesRegexp(ValueError, re.escape(expected_message)):
            RasterizeAndResize(
                height=height,
                width=width,
                one_hot=True,
                binarize=True,
                resize_method=method,
                class_count=1,
            )

    def test_supports_channels_last(self):
        processor = RasterizeAndResize(
            height=1, width=1, one_hot=True, binarize=True, class_count=1
        )
        assert processor.supported_formats == [CHANNELS_LAST]

    def test_does_not_compose(self):
        processor = RasterizeAndResize(
            height=1, width=1, one_hot=True, binarize=True, class_count=1
        )
        assert not processor.can_compose(Mock())

    def test_compose_raises(self):
        with self.assertRaises(NotImplementedError):
            processor = RasterizeAndResize(
                height=1, width=1, one_hot=True, binarize=True, class_count=1
            )
            processor.compose(Mock())

    def test_does_not_resize_frames_when_disabled(self):
        frames = tf.ones((16, 128, 240, 3))
        polygon = self.make_polygon_label([[120, 0.0], [240, 128], [0.0, 128]])

        height = 256
        width = 480
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        processor = RasterizeAndResize(
            height=height,
            width=width,
            one_hot=True,
            binarize=True,
            class_count=1,
            resize_frames=False,
        )
        with self.test_session():
            rasterized = processor.process(example)
            self.assertAllEqual(
                rasterized.instances[FEATURE_CAMERA].eval().shape, [16, 128, 240, 3]
            )

    @parameterized.expand([[64, 120], [252, 480], [504, 960], [1008, 1920]])
    def test_resizes_frames_to_requested_size(self, new_height, new_width):
        frames = tf.ones((16, 128, 240, 3))
        polygon = self.make_polygon_label([[120, 0.0], [240, 128], [0.0, 128]])
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        processor = RasterizeAndResize(
            height=new_height,
            width=new_width,
            one_hot=True,
            binarize=True,
            class_count=1,
            resize_frames=True,
        )
        with self.test_session():
            rasterized = processor.process(example)
            self.assertAllEqual(
                rasterized.instances[FEATURE_CAMERA].eval().shape,
                [16, new_height, new_width, 3],
            )
            self.assertAllEqual(
                rasterized.labels[LABEL_MAP].eval().shape, [1, new_height, new_width, 2]
            )

    @parameterized.expand(
        [
            [tf.image.ResizeMethod.AREA],
            [tf.image.ResizeMethod.BICUBIC],
            [tf.image.ResizeMethod.BILINEAR],
            [tf.image.ResizeMethod.NEAREST_NEIGHBOR],
        ]
    )
    @patch("tensorflow.image.resize", side_effect=tf.image.resize)
    def test_resizes_with_given_algorithms(self, method, spied_resize):
        frames = tf.ones((16, 128, 240, 3))
        polygon = self.make_polygon_label(
            vertices=[[60, 32], [180, 32], [180, 96], [60, 96]]
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        processor = RasterizeAndResize(
            height=252,
            width=480,
            one_hot=True,
            binarize=True,
            class_count=1,
            resize_frames=True,
            resize_method=method,
        )
        with self.test_session():
            rasterized = processor.process(example)
            self.assertAllEqual(
                rasterized.instances[FEATURE_CAMERA].eval().shape, [16, 252, 480, 3]
            )
            self.assertAllEqual(
                rasterized.labels[LABEL_MAP].eval().shape, [1, 252, 480, 2]
            )
            spied_resize.assert_has_calls(
                [call(frames, size=(252, 480), method=method)]
            )

    @parameterized.expand([[252, 480, 1], [252, 480, 3], [252, 480, 4], [252, 480, 7]])
    def test_rasterizes_labels(self, new_height, new_width, class_count):
        frames = tf.ones((8, 128, 240, class_count))
        polygon = self.make_polygon_label(
            vertices=[[60, 32], [180, 32], [180, 96], [60, 96]]
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: polygon}
        )
        processor = RasterizeAndResize(
            height=new_height,
            width=new_width,
            one_hot=True,
            binarize=True,
            class_count=class_count,
            resize_frames=True,
        )
        with self.test_session():
            rasterized = processor.process(example)
            self.assertAllEqual(
                rasterized.instances[FEATURE_CAMERA].eval().shape,
                [8, new_height, new_width, class_count],
            )
            self.assertAllEqual(
                rasterized.labels[LABEL_MAP].eval().shape,
                [1, new_height, new_width, class_count + 1],
            )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        processor = RasterizeAndResize(
            height=10,
            width=15,
            one_hot=True,
            binarize=True,
            class_count=3,
            resize_frames=True,
        )
        processor_dict = processor.serialize()
        deserialized_dict = deserialize_tao_object(processor_dict)
        assert processor._height == deserialized_dict._height
        assert processor._width == deserialized_dict._width
        assert processor._resize_frames == deserialized_dict._resize_frames
        assert processor._resize_method == deserialized_dict._resize_method
