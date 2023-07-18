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

"""Tests for RandomBrightness processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from mock import patch
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_brightness import (
    RandomBrightness,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from modulus.processors.augment.color import random_brightness_matrix


class TestRandomBrightness(ProcessorTestCase):
    @patch(
        "modulus.processors.augment.color.random_brightness_matrix",
        side_effect=random_brightness_matrix,
    )
    def test_delegates_to_random_brightness_matrix(
        self, spied_random_brightness_matrix
    ):
        example = self.make_example_128x240()
        augmentation = RandomBrightness(scale_max=0.5, uniform_across_channels=False)
        augmentation.process(example)
        spied_random_brightness_matrix.assert_called_with(
            brightness_scale_max=0.5,
            brightness_uniform_across_channels=False,
            batch_size=None,
        )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        augmentation = RandomBrightness(scale_max=0.5, uniform_across_channels=False)
        augmentation_dict = augmentation.serialize()
        deserialized_augmentation = deserialize_tao_object(augmentation_dict)
        self.assertEqual(
            str(augmentation._transform), str(deserialized_augmentation._transform)
        )
