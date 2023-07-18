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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mock import MagicMock, Mock, patch, PropertyMock
import pytest

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import (
    CHANNELS_FIRST,
    CHANNELS_LAST,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.convert_data_format import (
    ConvertDataFormat,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.pipeline import Pipeline
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
    SequenceExample,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


@patch(
    "nvidia_tao_tf1.blocks.multi_source_loader.processors.convert_data_format.tf.transpose"
)
def test_converts_channels_first_to_last(mocked_transpose):
    frames = Mock()
    instances = {FEATURE_CAMERA: frames}
    labels = {}
    example = Example(instances=instances, labels=labels)
    transposed_frames = Mock()
    transposed_instances = {FEATURE_CAMERA: transposed_frames}
    mocked_transpose.return_value = transposed_frames
    expected_output = Example(instances=transposed_instances, labels=labels)

    converter = ConvertDataFormat(CHANNELS_FIRST, CHANNELS_LAST)
    outputs = converter.process(example)

    mocked_transpose.assert_called_with(a=frames, perm=(0, 2, 3, 1))
    assert expected_output == outputs


@patch(
    "nvidia_tao_tf1.blocks.multi_source_loader.processors.convert_data_format.tf.transpose"
)
def test_converts_channels_last_to_first(mocked_transpose):
    frames = Mock()
    instances = {FEATURE_CAMERA: frames}
    labels = {}
    example = Example(instances=instances, labels=labels)
    transposed_frames = Mock()
    transposed_instances = {FEATURE_CAMERA: transposed_frames}
    mocked_transpose.return_value = transposed_frames
    expected_output = Example(instances=transposed_instances, labels=labels)

    converter = ConvertDataFormat(CHANNELS_LAST, CHANNELS_FIRST)
    outputs = converter.process(example)

    mocked_transpose.assert_called_with(a=frames, perm=(0, 3, 1, 2))
    assert expected_output == outputs


def test_empty_pipeline_outputs_inputs():
    pipeline = Pipeline(
        [], input_data_format=CHANNELS_FIRST, output_data_format=CHANNELS_FIRST
    )
    inputs = [Mock()]
    outputs = pipeline.process(inputs)
    assert outputs == inputs


def test_passes_outputs_to_the_next_processor():
    processor1 = Mock(supported_formats=[CHANNELS_FIRST])
    processor1.can_compose.return_value = False
    processor1_output = Mock()
    processor1.process.return_value = processor1_output

    processor2 = Mock(supported_formats=[CHANNELS_FIRST])
    processor2.can_compose.return_value = False
    processor2_output = Mock()
    processor2.process.return_value = processor2_output

    pipeline = Pipeline(
        [processor1, processor2],
        input_data_format=CHANNELS_FIRST,
        output_data_format=CHANNELS_FIRST,
    )
    example = Mock()
    output = pipeline.process([example])

    processor1.process.assert_called_with(example)
    processor2.process.assert_called_with(processor1_output)
    assert output == [processor2_output]


@pytest.mark.parametrize(
    "input_format, output_format, conversion_count",
    [
        [CHANNELS_FIRST, CHANNELS_FIRST, 0],
        [CHANNELS_FIRST, CHANNELS_LAST, 1],
        [CHANNELS_LAST, CHANNELS_FIRST, 1],
        [CHANNELS_LAST, CHANNELS_LAST, 0],
    ],
)
@patch(
    "nvidia_tao_tf1.blocks.multi_source_loader.processors.pipeline.ConvertDataFormat"
)
def test_empty_pipeline_conversions(
    _MockedConvertDataFormat, input_format, output_format, conversion_count
):
    pipeline = Pipeline(
        [], input_data_format=input_format, output_data_format=output_format
    )
    pipeline._build()

    assert len(pipeline._processors) == conversion_count
    assert _MockedConvertDataFormat.call_count == conversion_count


@pytest.mark.parametrize(
    "input_format, processor_format, output_format, conversion_count",
    [
        [CHANNELS_FIRST, CHANNELS_FIRST, CHANNELS_FIRST, 0],
        [CHANNELS_FIRST, CHANNELS_FIRST, CHANNELS_LAST, 1],
        [CHANNELS_FIRST, CHANNELS_LAST, CHANNELS_FIRST, 2],
        [CHANNELS_FIRST, CHANNELS_LAST, CHANNELS_LAST, 1],
        [CHANNELS_LAST, CHANNELS_FIRST, CHANNELS_FIRST, 1],
        [CHANNELS_LAST, CHANNELS_FIRST, CHANNELS_LAST, 2],
        [CHANNELS_LAST, CHANNELS_LAST, CHANNELS_FIRST, 1],
        [CHANNELS_LAST, CHANNELS_LAST, CHANNELS_LAST, 0],
    ],
)
@patch(
    "nvidia_tao_tf1.blocks.multi_source_loader.processors.pipeline.ConvertDataFormat"
)
def test_single_single_format_processor_pipeline(
    _MockedConvertDataFormat,
    input_format,
    processor_format,
    output_format,
    conversion_count,
):
    convert_instance = Mock()
    _MockedConvertDataFormat.return_value = convert_instance

    processor = Mock()
    processor_output = Mock()
    processor.supported_formats = [processor_format]
    processor.process.return_value = processor_output

    pipeline = Pipeline(
        [processor], input_data_format=input_format, output_data_format=output_format
    )
    pipeline._build()

    assert len(pipeline._processors) == (conversion_count + 1)
    assert _MockedConvertDataFormat.call_count == conversion_count


@pytest.mark.parametrize(
    "input_format, output_format, conversion_count",
    [
        [CHANNELS_FIRST, CHANNELS_FIRST, 0],
        [CHANNELS_FIRST, CHANNELS_LAST, 1],
        [CHANNELS_LAST, CHANNELS_FIRST, 1],
        [CHANNELS_LAST, CHANNELS_LAST, 0],
    ],
)
@patch(
    "nvidia_tao_tf1.blocks.multi_source_loader.processors.pipeline.ConvertDataFormat"
)
def test_single_dual_format_processor(
    _MockedConvertDataFormat, input_format, output_format, conversion_count
):
    convert_instance = Mock()
    _MockedConvertDataFormat.return_value = convert_instance

    processor = Mock()
    processor_output = Mock()
    processor.supported_formats = [CHANNELS_FIRST, CHANNELS_LAST]
    processor.process.return_value = processor_output

    pipeline = Pipeline(
        [processor], input_data_format=input_format, output_data_format=output_format
    )
    pipeline._build()

    assert len(pipeline._processors) == (conversion_count + 1)
    assert _MockedConvertDataFormat.call_count == conversion_count


def test_does_not_convert_sequence_examples():
    with patch.object(ConvertDataFormat, "process") as mocked_process:
        pipeline = Pipeline(
            [], input_data_format=CHANNELS_FIRST, output_data_format=CHANNELS_LAST
        )

        example = SequenceExample(instances={}, labels={})
        pipeline(example)

        assert 0 == mocked_process.call_count


def test_single_composable_processor_is_not_modified():
    transformer1 = Mock(supported_formats=[CHANNELS_LAST])
    transformer1.can_compose.return_value = True

    pipeline = Pipeline(
        [transformer1],
        input_data_format=CHANNELS_LAST,
        output_data_format=CHANNELS_LAST,
    )

    assert len(pipeline) == 1
    assert pipeline[0] == transformer1


def test_composes_two_consecutive_processors_into_one():
    transformer1 = Mock(supported_formats=[CHANNELS_LAST])
    transformer1.can_compose.return_value = True

    composed = Mock(supported_formats=[CHANNELS_LAST])
    composed.can_compose.return_value = False
    transformer1.compose.return_value = composed

    transformer2 = Mock(supported_formats=[CHANNELS_LAST])
    transformer2.can_compose.return_value = True

    pipeline = Pipeline(
        [transformer1, transformer2],
        input_data_format=CHANNELS_LAST,
        output_data_format=CHANNELS_LAST,
    )
    assert len(pipeline) == 2

    pipeline._build()

    assert len(pipeline) == 1


def test_does_not_compose_last_composable_processor():
    transformer1 = Mock(supported_formats=[CHANNELS_LAST])
    transformer1.can_compose.return_value = False

    transformer2 = Mock(supported_formats=[CHANNELS_LAST])
    transformer2.can_compose.return_value = False

    transformer3 = Mock(supported_formats=[CHANNELS_LAST])
    transformer3.can_compose.return_value = True

    pipeline = Pipeline(
        [transformer1, transformer2, transformer3],
        input_data_format=CHANNELS_LAST,
        output_data_format=CHANNELS_LAST,
    )
    assert len(pipeline) == 3

    pipeline._build()

    assert len(pipeline) == 3


def test_does_not_compose_when_separated_by_composable():
    transformer1 = MagicMock()
    type(transformer1).supported_formats = PropertyMock(return_value=[CHANNELS_LAST])
    transformer1.can_compose.return_value = False

    transformer2 = MagicMock()
    type(transformer2).supported_formats = PropertyMock(return_value=[CHANNELS_LAST])
    transformer2.can_compose.return_value = False

    transformer3 = MagicMock()
    type(transformer3).supported_formats = PropertyMock(return_value=[CHANNELS_LAST])
    transformer3.can_compose.return_value = False

    pipeline = Pipeline(
        [transformer1, transformer2, transformer3],
        input_data_format=CHANNELS_LAST,
        output_data_format=CHANNELS_LAST,
    )
    assert len(pipeline) == 3

    pipeline._build()

    assert len(pipeline) == 3


def test_does_not_compose_processors_when_disabled():
    transformer1 = Mock(supported_formats=[CHANNELS_LAST])
    transformer1.can_compose.return_value = True

    composed = Mock(supported_formats=[CHANNELS_LAST])
    composed.can_compose.return_value = False
    transformer1.compose.return_value = composed

    transformer2 = Mock(supported_formats=[CHANNELS_LAST])
    transformer2.can_compose.return_value = True

    pipeline = Pipeline(
        [transformer1, transformer2],
        input_data_format=CHANNELS_LAST,
        output_data_format=CHANNELS_LAST,
        compose=False,
    )
    assert len(pipeline) == 2

    pipeline._build()

    assert len(pipeline) == 2
    assert pipeline[0] == transformer1
    assert pipeline[1] == transformer2


def test_serialization_and_deserialization():
    """Test that it is a TAOObject that can be serialized and deserialized."""
    pipeline = Pipeline(
        [],
        input_data_format=CHANNELS_LAST,
        output_data_format=CHANNELS_LAST,
        compose=False,
    )
    pipeline_dict = pipeline.serialize()
    deserialized_dict = deserialize_tao_object(pipeline_dict)
    assert pipeline._processors == deserialized_dict._processors
    assert pipeline._output_data_format == deserialized_dict._output_data_format
    assert pipeline._input_data_format == deserialized_dict._input_data_format
    assert pipeline._compose == deserialized_dict._compose
