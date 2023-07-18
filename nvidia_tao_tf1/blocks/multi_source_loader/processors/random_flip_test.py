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

"""Tests for RandomFlip processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from mock import patch
from parameterized import parameterized

from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_flip import (
    RandomFlip,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from modulus.processors.augment.spatial import random_flip_matrix


def _esc(message):
    """Escape passed in string for regular expressions."""
    return re.escape(message)


class TestRandomFlip(ProcessorTestCase):
    @parameterized.expand(
        [
            [
                -0.1,
                0.5,
                _esc(
                    "RandomFlip.horizontal_probability (-0.1) is not within the range [0.0, 1.0]."
                ),
            ],
            [
                1.1,
                0.5,
                _esc(
                    "RandomFlip.horizontal_probability (1.1) is not within the range [0.0, 1.0]."
                ),
            ],
            [
                0.5,
                -0.1,
                _esc(
                    "RandomFlip.vertical_probability (-0.1) is not within the range [0.0, 1.0]."
                ),
            ],
            [
                0.5,
                1.1,
                _esc(
                    "RandomFlip.vertical_probability (1.1) is not within the range [0.0, 1.0]."
                ),
            ],
        ]
    )
    def test_raises_on_invalid_horizontal_probability(
        self, horizontal_probability, vertical_probability, message
    ):
        with self.assertRaisesRegexp(ValueError, message):
            RandomFlip(
                horizontal_probability=horizontal_probability,
                vertical_probability=vertical_probability,
            )

    @parameterized.expand([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    @patch(
        "modulus.processors.augment.spatial.random_flip_matrix",
        side_effect=random_flip_matrix,
    )
    def test_delegates_to_random_flip_matrix(
        self, horizontal_probability, vertical_probability, spied_random_flip_matrix
    ):
        example = self.make_example_128x240()
        augmentation = RandomFlip(
            horizontal_probability=horizontal_probability,
            vertical_probability=vertical_probability,
        )
        augmentation.process(example)
        spied_random_flip_matrix.assert_called_with(
            horizontal_probability=horizontal_probability,
            vertical_probability=vertical_probability,
            height=128,
            width=240,
            batch_size=None,
        )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        augmentation = RandomFlip(horizontal_probability=1.0, vertical_probability=1.0)
        augmentation_dict = augmentation.serialize()
        deserialized_augmentation = deserialize_tao_object(augmentation_dict)
        self.assertEqual(
            str(augmentation._transform), str(deserialized_augmentation._transform)
        )
