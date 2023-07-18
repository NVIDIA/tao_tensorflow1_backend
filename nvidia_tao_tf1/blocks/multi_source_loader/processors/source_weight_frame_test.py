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

"""Test for SourceWeightSQLFrameProcessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from parameterized import parameterized
import tensorflow as tf
import modulus
from nvidia_tao_tf1.blocks.multi_source_loader.processors.source_weight_frame import (
    SourceWeightSQLFrameProcessor,
)


class TestSourceWeightSQLFrameProcessor(tf.test.TestCase):
    """Test for SourceWeightSQLFrameProcessor."""

    @parameterized.expand([(1.0, 1.0), (2.0, 2.0)])
    def test_process(self, source_weight, expected_weight):

        row = [
            "person",
            [[50.5, 688.83], [164.28, 766.8]],
            1,
            0.0,  # This is the field added by add_field
        ]

        label_indices = {"BOX": {"is_cvip": 2, "vertices": 1, "classifier": 0}}

        instances = {"source_weight": 3}

        example = modulus.types.types.Example(instances=instances, labels=label_indices)
        source_weight_processor = SourceWeightSQLFrameProcessor(source_weight)

        processed_row = source_weight_processor.map(example, row)
        self.assertEqual(expected_weight, processed_row[instances["source_weight"]])
