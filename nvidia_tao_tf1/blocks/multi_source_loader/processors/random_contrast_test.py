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

"""Tests for RandomContrast processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from mock import patch

from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_contrast import (
    RandomContrast,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from modulus.processors.augment.color import random_contrast_matrix


class TestRandomContrast(ProcessorTestCase):
    @patch(
        "modulus.processors.augment.color.random_contrast_matrix",
        side_effect=random_contrast_matrix,
    )
    def test_delegates_to_random_contrast_matrix(self, spied_random_contrast_matrix):
        example = self.make_example_128x240()
        augmentation = RandomContrast(scale_max=180.0, center=0.0)
        augmentation.process(example)
        spied_random_contrast_matrix.assert_called_with(
            scale_max=180.0, center=0.0, batch_size=None
        )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        augmentation = RandomContrast(scale_max=180.0, center=0.0)
        augmentation_dict = augmentation.serialize()
        deserialized_augmentation = deserialize_tao_object(augmentation_dict)
        self.assertEqual(
            str(augmentation._transform), str(deserialized_augmentation._transform)
        )
