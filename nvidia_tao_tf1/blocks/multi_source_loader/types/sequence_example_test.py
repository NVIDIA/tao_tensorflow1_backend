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

import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    test_fixtures as fixtures,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    FEATURE_CAMERA,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_MAP,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    SequenceExample,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    TransformedExample,
)


class SequenceExampleTest(tf.test.TestCase):
    def test_transform_returns_transformed_example(self):
        example = fixtures.make_example(height=604, width=960)
        transformation = fixtures.make_identity_transform(
            count=2, height=604, width=960
        )
        transformed = example.transform(transformation)
        self.assertEqual(TransformedExample, type(transformed))

    def test_call_applies_identity_transform(self):
        example = fixtures.make_example(height=604, width=960)
        transformation = fixtures.make_identity_transform(
            count=1, height=604, width=960
        )
        transformed = example.transform(transformation)
        with self.session() as sess:
            example, transformed = sess.run([example, transformed()])
            self.assertAllClose(
                example.labels[LABEL_MAP].vertices,
                transformed.labels[LABEL_MAP].vertices,
            )
            self.assertAllClose(
                example.instances[FEATURE_CAMERA].images,
                transformed.instances[FEATURE_CAMERA].images,
            )

    def test_call_returns_sequence_example(self):
        example = fixtures.make_example(height=604, width=960)
        transformation = fixtures.make_identity_transform(
            count=1, height=604, width=960
        )
        transformed = example.transform(transformation)
        example = transformed()
        self.assertEqual(SequenceExample, type(example))
