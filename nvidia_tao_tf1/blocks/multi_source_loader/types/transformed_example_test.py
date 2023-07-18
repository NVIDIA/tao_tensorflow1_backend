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

"""Unit tests for the TransformedExample datastructure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from mock import Mock, patch
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    SequenceExample,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.transformed_example import (
    TransformedExample,
)

TestInstance = namedtuple("TestInstance", ["apply"])
TestLabel = namedtuple("TestLabel", ["apply"])


class TransformedExampleTest(tf.test.TestCase):
    def test_apply_recurses_instances(self):
        with patch.object(TestInstance, "apply") as mocked_apply:
            example = SequenceExample(
                instances={"test": TestInstance(apply="test")}, labels={}
            )
            transformation = Mock()
            transformed = TransformedExample(transformation, example)
            mocked_apply.assert_not_called()
            transformed()
            mocked_apply.assert_called_once_with(transformation)

    def test_apply_recurses_labels(self):
        with patch.object(TestLabel, "apply") as mocked_apply:
            example = SequenceExample(
                instances={}, labels={"test": TestLabel(apply="test")}
            )
            transformation = Mock()
            transformed = TransformedExample(transformation, example)
            mocked_apply.assert_not_called()
            transformed()
            mocked_apply.assert_called_once_with(transformation)

    def test_apply_recurses_lists(self):
        with patch.object(TestLabel, "apply") as mocked_apply:
            example = SequenceExample(
                instances={}, labels={"test": [TestLabel(apply="test")]}
            )
            transformation = Mock()
            transformed = TransformedExample(transformation, example)
            mocked_apply.assert_not_called()
            transformed()
            mocked_apply.assert_called_once_with(transformation)

    def test_apply_recurses_sets(self):
        with patch.object(TestLabel, "apply") as mocked_apply:
            example = SequenceExample(
                instances={}, labels={"test": set([TestLabel(apply="test")])}
            )
            transformation = Mock()
            transformed = TransformedExample(transformation, example)
            mocked_apply.assert_not_called()
            transformed()
            mocked_apply.assert_called_once_with(transformation)

    def test_apply_recurses_dicts(self):
        with patch.object(TestLabel, "apply") as mocked_apply:
            example = SequenceExample(
                instances={}, labels={"test": {"child": TestLabel(apply="test")}}
            )
            transformation = Mock()
            transformed = TransformedExample(transformation, example)
            mocked_apply.assert_not_called()
            transformed()
            mocked_apply.assert_called_once_with(transformation)

    def test_apply_recurses_namedtuples_without_apply(self):
        ApplylessNamedtuple = namedtuple("ApplylessNamedtuple", ["instance"])
        with patch.object(TestLabel, "apply") as mocked_apply:
            example = SequenceExample(
                instances={},
                labels={"test": ApplylessNamedtuple(instance=TestLabel(apply="test"))},
            )
            transformation = Mock()
            transformed = TransformedExample(transformation, example)
            mocked_apply.assert_not_called()
            transformed()
            mocked_apply.assert_called_once_with(transformation)

    def test_does_not_recurse_into_namedtuples_with_apply(self):
        # We consider namedtuples with an "apply" method to be a leaf node and stop recursive
        # application of transformations when we encounter one.
        TestLabelWithInstance = namedtuple(
            "TestLabelWithInstance", ["apply", "instance"]
        )
        with patch.object(TestLabelWithInstance, "apply") as mocked_apply, patch.object(
            TestInstance, "apply"
        ) as mocked_instance_apply:
            example = SequenceExample(
                instances={},
                labels={
                    "test": {
                        "child": TestLabelWithInstance(
                            apply="test", instance=TestInstance(apply="test")
                        )
                    }
                },
            )
            transformation = Mock()
            transformed = TransformedExample(transformation, example)
            mocked_apply.assert_not_called()
            mocked_instance_apply.assert_not_called()
            transformed()
            mocked_apply.assert_called_once_with(transformation)
            mocked_instance_apply.assert_not_called()
