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

"""Unit tests for Images2D."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    test_fixtures as fixtures,
)


class Images2DTest(tf.test.TestCase):
    @parameterized.expand([[1, 1], [1, 2], [2, 1], [2, 2]])
    def test_transform_succeeds(self, example_count, frames_per_example):
        with self.session() as sess:
            images = fixtures.make_images2d(
                example_count=example_count,
                frames_per_example=frames_per_example,
                height=604,
                width=960,
            )
            transformation = fixtures.make_identity_transform(
                count=example_count, height=604, width=960, timesteps=frames_per_example
            )
            transformed_images = images.apply(transformation)
            original, transformed = sess.run([images, transformed_images])

            self.assertAllClose(original, transformed)
