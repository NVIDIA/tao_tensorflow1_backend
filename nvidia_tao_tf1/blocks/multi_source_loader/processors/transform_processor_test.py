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

"""Tests for TestTransformProcessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mock import Mock

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import (
    CHANNELS_FIRST,
    CHANNELS_LAST,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    FEATURE_CAMERA,
    LABEL_MAP,
)


class TestTransformProcessor(ProcessorTestCase):
    def test_construction_fails_when_transfomer_is_none(self):
        with self.assertRaises(ValueError):
            TransformProcessor(None)

    def test_supports_channels_last(self):
        processor = TransformProcessor(Mock())
        assert processor.supported_formats == [CHANNELS_FIRST, CHANNELS_LAST]

    def test_process_fails_when_input_is_not_an_example(self):
        with self.assertRaises(TypeError):
            TransformProcessor(Mock()).process(Mock())

    def test_uses_provided_transform(self):
        transform = Mock()
        processor = TransformProcessor(transform)
        # identity transform produces outputs identical to inputs
        processor._transform = Mock(side_effect=lambda transform: transform)

        with self.test_session() as session:
            example = self.make_example_128x240()

            processed = processor.process(example)
            processor._transform.assert_called_once()

            self.assertAllEqual(
                session.run(processed.instances[FEATURE_CAMERA]),
                session.run(example.instances[FEATURE_CAMERA]),
            )
            self.assertAllClose(
                session.run(processed.labels[LABEL_MAP]),
                session.run(example.labels[LABEL_MAP]),
            )
