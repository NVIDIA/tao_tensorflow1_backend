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

"""Tests for Crop processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.crop import Crop
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestCrop(ProcessorTestCase):
    @parameterized.expand([[0, 0, 1, 1], [1, 1, 2, 2]])
    def test_valid_bounds_do_not_raise(self, left, top, right, bottom):
        Crop(left=left, top=top, right=right, bottom=bottom)

    def test_crops_in_half(self):
        frames = tf.ones((1, 128, 240, 3))
        labels = self.make_polygon_label(
            vertices=[[60, 32], [180, 32], [180, 96], [60, 96]]
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: labels}
        )

        expected_frames = tf.ones((1, 128, 120, 3))
        # X coordinates become negative because cropping from the left translates the left side out
        # of view
        expected_labels = self.make_polygon_label(
            vertices=[[-60, 32], [60, 32], [60, 96], [-60, 96]]
        )

        with self.test_session():
            crop = Crop(left=120, top=0, right=240, bottom=128)
            cropped = crop.process(example)

            self.assertAllClose(
                expected_frames.eval(), cropped.instances[FEATURE_CAMERA].eval()
            )
            self.assert_labels_close(expected_labels, cropped.labels[LABEL_MAP])

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        processor = Crop(left=120, top=0, right=240, bottom=128)
        processor_dict = processor.serialize()
        deserialized_processor = deserialize_tao_object(processor_dict)
        self.assertEqual(
            str(processor._transform), str(deserialized_processor._transform)
        )
