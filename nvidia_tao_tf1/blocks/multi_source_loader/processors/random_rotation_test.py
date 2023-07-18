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

"""Tests for RandomRotation processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mock import Mock, patch
import pytest

from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_rotation import (
    RandomRotation,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestRandomRotation(ProcessorTestCase):
    @patch(
        "nvidia_tao_tf1.blocks.multi_source_loader.processors.random_rotation._RandomRotation"
    )
    def test_forwards_constructor_arguments_to_transform(
        self, mocked_rotation_transform
    ):
        min_angle = Mock()
        max_angle = Mock()
        probability = Mock()
        RandomRotation(
            min_angle=min_angle, max_angle=max_angle, probability=probability
        )
        mocked_rotation_transform.assert_called_with(
            min_angle=min_angle, max_angle=max_angle, probability=probability
        )

    @patch(
        "nvidia_tao_tf1.blocks.multi_source_loader.processors.random_rotation._RandomRotation"
    )
    def test_forwards_transform_value_errors_to_caller(self, mocked_rotation_transform):
        def raise_value_error(**kwargs):
            raise ValueError("test error")

        mocked_rotation_transform.side_effect = raise_value_error

        with pytest.raises(ValueError) as exc:
            RandomRotation(min_angle=4, max_angle=7, probability=1.0)
        assert "test error" in str(exc)

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        augmentation = RandomRotation(min_angle=4, max_angle=7, probability=1.0)
        augmentation_dict = augmentation.serialize()
        deserialized_augmentation = deserialize_tao_object(augmentation_dict)
        self.assertEqual(
            str(augmentation._transform), str(deserialized_augmentation._transform)
        )
