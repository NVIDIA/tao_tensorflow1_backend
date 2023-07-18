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

"""Tests for RandomTranslation processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from mock import patch

from parameterized import parameterized

from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_translation import (
    RandomTranslation,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from modulus.processors.augment.spatial import random_translation_matrix


def _esc(message):
    """Escape passed in string for regular expressions."""
    return re.escape(message)


class TestRandomTranslation(ProcessorTestCase):
    @parameterized.expand(
        [
            [
                -0.1,
                _esc(
                    "RandomTranslation.probability (-0.1) is not within the range [0.0, 1.0]."
                ),
            ],
            [
                1.1,
                _esc(
                    "RandomTranslation.probability (1.1) is not within the range [0.0, 1.0]."
                ),
            ],
        ]
    )
    def test_raises_on_invalid_probability(self, probability, message):
        with self.assertRaisesRegexp(ValueError, message):
            RandomTranslation(max_x=7, max_y=7, probability=probability)

    @patch(
        "modulus.processors.augment.spatial.random_translation_matrix",
        side_effect=random_translation_matrix,
    )
    def test_delegates_to_random_translation_matrix(
        self, spied_random_translation_matrix
    ):
        example = self.make_example_128x240()
        augmentation = RandomTranslation(max_x=90, max_y=45, probability=1.0)
        augmentation.process(example)
        spied_random_translation_matrix.assert_called_with(
            max_x=90, max_y=45, batch_size=None
        )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        augmentation = RandomTranslation(max_x=90, max_y=45, probability=1.0)
        augmentation_dict = augmentation.serialize()
        deserialized_augmentation = deserialize_tao_object(augmentation_dict)
        self.assertEqual(
            str(augmentation._transform), str(deserialized_augmentation._transform)
        )
