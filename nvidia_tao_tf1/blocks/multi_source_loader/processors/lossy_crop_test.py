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

from nvidia_tao_tf1.blocks.multi_source_loader.processors.lossy_crop import (
    LossyCrop,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    FEATURE_CAMERA,
    LABEL_MAP,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures import (
    make_example,
    make_images2d,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestLossyCrop(ProcessorTestCase):
    @parameterized.expand([[0, 0, 1, 1], [1, 1, 2, 2]])
    def test_valid_bounds_do_not_raise(self, left, top, right, bottom):
        LossyCrop(left=left, top=top, right=right, bottom=bottom)

    def test_crops_in_half(self):
        example = make_example(
            128, 240, coordinate_values=[[42.0, 26.5], [116.0, 53.5], [74.0, 40.5]]
        )
        expected_images2d = make_images2d(
            example_count=1, frames_per_example=1, height=128, width=120
        )
        # X coordinates become negative because cropping from the left translates the left side out
        # of view
        expected_coordinates = [-78.0, 26.5, -4.0, 53.5, -46.0, 40.5]

        with self.test_session():
            crop = LossyCrop(left=120, top=0, right=240, bottom=128)
            cropped = crop.process(example)

            self.assertAllClose(
                expected_images2d.images.eval(),
                cropped.instances[FEATURE_CAMERA].images.eval(),
            )

            self.assertAllClose(
                expected_coordinates,
                cropped.labels[LABEL_MAP].vertices.coordinates.values,
            )
            self.assertEqual(
                120, cropped.labels[LABEL_MAP].vertices.canvas_shape.width.shape[0]
            )
            self.assertEqual(
                128, cropped.labels[LABEL_MAP].vertices.canvas_shape.height.shape[0]
            )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        crop = LossyCrop(left=120, top=0, right=240, bottom=128)
        crop_dict = crop.serialize()
        deserialized_crop = deserialize_tao_object(crop_dict)
        assert crop._left == deserialized_crop._left
        assert crop._top == deserialized_crop._top
        assert crop._right == deserialized_crop._right
        assert crop._bottom == deserialized_crop._bottom
