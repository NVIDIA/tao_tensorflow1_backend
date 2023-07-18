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

"""Tests for AssetLoader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from mock import patch
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.processors.asset_loader import (
    AssetLoader,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    SequenceExample,
)

TestInstance = collections.namedtuple("TestInstance", ["load"])
TestLabel = collections.namedtuple("TestLabel", ["load"])


class AssetLoaderTest(tf.test.TestCase):
    def test_load_recurses_instances(self):
        with patch.object(TestInstance, "load") as mocked_load:
            example = SequenceExample(
                instances={"test": TestInstance(load="test")}, labels={}
            )
            mocked_load.assert_not_called()
            AssetLoader()(example)
            mocked_load.assert_called_once()

    def test_load_recurses_labels(self):
        with patch.object(TestLabel, "load") as mocked_load:
            example = SequenceExample(
                instances={}, labels={"test": TestLabel(load="test")}
            )

            mocked_load.assert_not_called()
            AssetLoader()(example)
            mocked_load.assert_called_once()

    def test_load_recurses_lists(self):
        with patch.object(TestLabel, "load") as mocked_load:
            example = SequenceExample(
                instances={}, labels={"test": [TestLabel(load="test")]}
            )
            mocked_load.assert_not_called()
            AssetLoader()(example)
            mocked_load.assert_called_once()

    def test_load_recurses_sets(self):
        with patch.object(TestLabel, "load") as mocked_load:
            example = SequenceExample(
                instances={}, labels={"test": set([TestLabel(load="test")])}
            )
            mocked_load.assert_not_called()
            AssetLoader()(example)
            mocked_load.assert_called_once()

    def test_load_recurses_dicts(self):
        with patch.object(TestLabel, "load") as mocked_load:
            example = SequenceExample(
                instances={}, labels={"test": {"child": TestLabel(load="test")}}
            )
            mocked_load.assert_not_called()
            AssetLoader()(example)
            mocked_load.assert_called_once()

    def test_load_recurses_namedtuples_without_load(self):
        LoadlessNamedtuple = collections.namedtuple("LoadlessNamedtuple", ["instance"])
        with patch.object(TestLabel, "load") as mocked_load:
            example = SequenceExample(
                instances={},
                labels={"test": LoadlessNamedtuple(instance=TestLabel(load="test"))},
            )
            mocked_load.assert_not_called()
            AssetLoader()(example)
            mocked_load.assert_called_once()

    def test_does_not_recurse_into_namedtuples_with_load(self):
        # We consider namedtuples with an "load" method to be a leaf node and stop recursion
        # when we encounter one.
        TestLabelWithInstance = collections.namedtuple(
            "TestLabelWithInstance", ["load", "instance"]
        )
        with patch.object(TestLabelWithInstance, "load") as mocked_load, patch.object(
            TestInstance, "load"
        ) as mocked_instance_load:
            example = SequenceExample(
                instances={},
                labels={
                    "test": {
                        "child": TestLabelWithInstance(
                            load="test", instance=TestInstance(load="test")
                        )
                    }
                },
            )
            mocked_load.assert_not_called()
            mocked_instance_load.assert_not_called()
            AssetLoader()(example)
            mocked_load.assert_called_once()
            mocked_instance_load.assert_not_called()
