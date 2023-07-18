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

"""Tests for TestFilterProcessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mock import Mock
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import (
    CHANNELS_FIRST,
    CHANNELS_LAST,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.filter2d_processor import (
    Filter2DProcessor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import FEATURE_CAMERA
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class ConstantFilter2DProcessor(Filter2DProcessor):
    @property
    def probability(self):
        return 1.0

    def get_filters(self):
        return [
            tf.reshape(tf.constant([0.5]), shape=[1, 1]),
            tf.reshape(tf.constant([0.6]), shape=[1, 1]),
        ]


class TestFilterProcessor(ProcessorTestCase):
    def test_supports_channels_first_and_channels_last(self):
        processor = ConstantFilter2DProcessor()
        assert processor.supported_formats == [CHANNELS_FIRST, CHANNELS_LAST]

    def test_process_fails_on_none(self):
        with self.assertRaises(TypeError):
            ConstantFilter2DProcessor().process(None)

    def test_process_fails_when_input_is_not_an_example(self):
        with self.assertRaises(TypeError):
            ConstantFilter2DProcessor().process(Mock())

    def test_uses_provided_filter(self):
        processor = ConstantFilter2DProcessor()
        processor.data_format = CHANNELS_LAST

        with self.test_session() as session:
            example = self.make_example_128x240()
            np_example = session.run(example.instances[FEATURE_CAMERA])

            processed = processor.process(example)
            np_processed = session.run(processed.instances[FEATURE_CAMERA])

        self.assertAllClose(np_processed, 0.3 * np_example)

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        processor = ConstantFilter2DProcessor()
        processor_dict = processor.serialize()
        deserialized_processor_dict = deserialize_tao_object(processor_dict)
        assert processor.data_format == deserialized_processor_dict.data_format
