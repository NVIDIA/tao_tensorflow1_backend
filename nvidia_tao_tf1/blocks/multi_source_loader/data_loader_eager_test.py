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

"""Tests for DataLoader in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import tensorflow as tf

tf.compat.v1.enable_eager_execution()  # noqa

from nvidia_tao_tf1.blocks.multi_source_loader import processors
from nvidia_tao_tf1.blocks.multi_source_loader.data_loader import DataLoader
from nvidia_tao_tf1.blocks.multi_source_loader.sources.synthetic_data_source import (
    SyntheticDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import FEATURE_CAMERA
from nvidia_tao_tf1.blocks.multi_source_loader.types import test_fixtures


class DataLoaderTest(tf.test.TestCase):
    """Test the DataLoader in eager mode."""

    @parameterized.expand(
        [[1, 1, 1], [2, 1, 2], [3, 1, 3], [4, 2, 1], [5, 2, 2], [6, 2, 3]]
    )
    def test_iterates_all_examples(self, batch_count, batch_size, window_size):
        example_count = batch_count * batch_size * window_size
        dataloader = DataLoader(
            data_sources=[
                SyntheticDataSource(
                    preprocessing=[],
                    example_count=example_count,
                    template=test_fixtures.make_example_3d(12, 24),
                )
            ],
            augmentation_pipeline=[],
            batch_size=batch_size,
            shuffle=False,
            repeat=False,
            sampling=None,
            preprocessing=[processors.TemporalBatcher(size=window_size)],
        )
        dataloader.set_shard()
        batches_read = 0
        for example in dataloader():
            batches_read += 1
            self.assertEqual(
                (batch_size, window_size, 3, 12, 24),
                example.instances[FEATURE_CAMERA].images.shape,
            )

        self.assertEqual(batch_count, batches_read)
