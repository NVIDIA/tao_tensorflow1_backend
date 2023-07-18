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

"""Tests for TemporalBatcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader import types
from nvidia_tao_tf1.blocks.multi_source_loader.processors.temporal_batcher import (
    TemporalBatcher,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import test_fixtures
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    SequenceExample,
)


class TemporalBatcherTest(tf.test.TestCase):
    def test_raises_with_invalid_size(self):
        with self.assertRaises(ValueError):
            TemporalBatcher(size=0)

    def test_raises_when_session_information_not_present(self):
        batcher = TemporalBatcher(size=2)
        example = SequenceExample(instances={"dummy": tf.constant(0)}, labels={})
        dataset = tf.data.Dataset.from_tensors(example)
        with self.assertRaises(ValueError):
            dataset.apply(batcher)

    @parameterized.expand([[1], [2], [3], [4]])
    def test_window_size(self, size):
        example = test_fixtures.make_example_3d(16, 32)
        dataset = tf.data.Dataset.from_tensors(example)
        dataset = dataset.repeat()
        dataset = dataset.apply(TemporalBatcher(size=size))
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        batch = iterator.get_next()

        # test for static shape inference
        self.assertEqual(
            [size, 3, 16, 32],
            batch.instances[types.FEATURE_CAMERA].images.shape.as_list(),
        )

        with self.cached_session() as session:
            batched = session.run(batch)
            assert (size, 3, 16, 32) == batched.instances[
                types.FEATURE_CAMERA
            ].images.shape
