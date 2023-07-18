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
"""Pipelines consists of chained together processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import logging

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_FIRST
from nvidia_tao_tf1.blocks.multi_source_loader.processors.convert_data_format import (
    ConvertDataFormat,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    SequenceExample,
    TransformedExample,
)
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args

logger = logging.getLogger(__name__)


def _compose(previous_processors, next_processor):
    """
    Compose the last processor in a sequence with the next one if possible.

    Args:
        previous_processors (list of Processors): Processors that already been composed or None when
            nothing has been composed yet.
        next_processor (Processor): Processor to attempt composing.
    Returns:
        (list of Processors): Previous processors combined with the next processor.
    """
    if not previous_processors:
        # Nothing to compose with
        return [next_processor]

    # Compose the last processor with the next one if possible.
    candidate = previous_processors[-1]
    if candidate.can_compose(next_processor):
        return previous_processors[:-1] + [candidate.compose(next_processor)]

    # Could not compose, add to existing list of processors.
    return previous_processors + [next_processor]


def _compose_processors(processors):
    """
    Compose all consecutive composable processors keeping non-composable processors intact.

    Args:
        processors (list of Processors): Processors to attempt composing.

    Returns:
        (list of Processors): List of processors that will be identical with the input list of
            processors when nothing could be composed. If composition was possible, the list will
            be shorter than the input list. When all processors were composable, the list will
            contain only one processor.
    """
    return functools.reduce(_compose, processors, [])


class Pipeline(TAOObject):
    """Configurable sequence of processors or transforming inputs (images, labels)."""

    @save_args
    def __init__(
        self,
        processors,
        input_data_format=CHANNELS_FIRST,
        output_data_format=CHANNELS_FIRST,
        compose=True,
    ):
        """
        Construct a pipeline.

        The passed in processors will be processed in sequence and conversions between data formats
        will be done automatically based on the format of individual processors and the desired
        input and output data formats.

        Args:
            processors (list of `Processor`): Sequence of processors to pass inputs through.
            input_data_format (DataFormat): Data format to use for output examples.
            output_data_format (DataFormat): Data format of input examples.
            compose (Boolean): If True, pipeline will attempt to compose consecutive processors.
        """
        super(Pipeline, self).__init__()
        self._processors = processors
        self._output_data_format = output_data_format
        self._input_data_format = input_data_format
        self._compose = compose
        self._is_built = False

    def _build(self):
        # Combine/compose consecutive processors when possible to reduce compute.
        processors = self._processors
        if self._compose:
            processors = _compose_processors(processors)
        # Add conversions that automatically handle NCHW <-> NHWC incompatibilities between
        # processors.
        self._processors = self._inject_conversion_processors(
            processors, self._input_data_format, self._output_data_format
        )
        self._is_built = True

    def __len__(self):
        """Return the number of processors in this pipeline."""
        return len(self._processors)

    def __add__(self, other):
        """Combine two pipelines."""
        assert not self._is_built, "Combining built pipelines is not supported"
        return Pipeline(
            self._processors + other._processors,
            self._input_data_format,
            self._output_data_format,
        )

    def __iter__(self):
        """Iterate over all processors in this pipeline."""
        for processor in self._processors:
            yield processor

    def __repr__(self):
        """Return a string representation of this pipeline."""
        return "Pipeline([{}], input_data_format={}, output_data_format={}, compose={}".format(
            ",".join([repr(processor) for processor in self._processors]),
            self._input_data_format,
            self._output_data_format,
            self._compose,
        )

    def __getitem__(self, index):
        """Get processor at given location.

        Because conversion processors can be automatically inserted between other processors, the
        processor is not necessarily the same as would be returned by indexing into the list of
        processors passed to the constructor of this class.

        Args:
            index (int): Index of the processor to return.

        Returns:
            processor (Processor): Processor at the given index.
        """
        return self._processors[index]

    def _inject_conversion_processors(self, processors, input_format, output_format):
        """Inject data format conversion processors in between processors."""
        processors_with_conversions = []
        previous_output_format = input_format
        # Naively insert data format conversions. TODO(vkallioniemi): make this minimize the
        # number of conversions done (build a tree, search for shortest path.)
        for i, processor in enumerate(processors):
            if previous_output_format not in processor.supported_formats:
                next_data_format = processor.supported_formats[0]
                logger.warning(
                    "inserting conversion after processor %d: %s -> %s",
                    i,
                    previous_output_format,
                    next_data_format,
                )

                conversion = ConvertDataFormat(previous_output_format, next_data_format)
                processors_with_conversions.append(conversion)
                previous_output_format = next_data_format

            processor.data_format = previous_output_format
            processors_with_conversions.append(processor)

        if previous_output_format != output_format:
            logger.warning(
                "inserting conversion at the end of the pipeline: %s -> %s",
                previous_output_format,
                output_format,
            )
            conversion = ConvertDataFormat(previous_output_format, output_format)
            processors_with_conversions.append(conversion)

        return processors_with_conversions

    def process(self, examples):
        """
        Process examples by letting all processors process them in sequence.

        Args:
            examples (List of `Example`s): Input data with frames in input_data_format.

        Returns:
            (List of `Example`s): Processed examples with frames in output_data_format.

        Raises:
            ValueError: When passed invalid arguments.
        """
        if not examples:
            raise ValueError("Pipeline.process called with 0 arguments.")

        processed_examples = []
        for example in examples:
            processed_example = self(example)
            processed_examples.append(processed_example)

        return processed_examples

    def __call__(self, example):
        """
        Process example by letting all processors process it in sequence.

        Args:
            example (Example): Input data with frames in input_data_format.

        Returns:
            (Example): Processed example with frames in output_data_format.

        Raises:
            ValueError: When passed invalid arguments.
        """
        if not self._is_built:
            self._build()

        processed_example = example
        for processor in self._processors:
            # SequenceExamples are always channels_first
            if isinstance(processor, ConvertDataFormat) and isinstance(
                processed_example, (SequenceExample, TransformedExample)
            ):
                continue

            processed_example = processor.process(processed_example)

        return processed_example
