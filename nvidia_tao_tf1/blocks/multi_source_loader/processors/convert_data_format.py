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
"""Processor for converting between data formats.  Used internally by pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import (
    CHANNELS_FIRST,
    CHANNELS_LAST,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor import (
    Processor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
)
from nvidia_tao_tf1.core.coreobject import save_args


class ConvertDataFormat(Processor):
    """Processor for converting between data formats. Used internally by pipelines."""

    NCHW_TO_NHWC = (0, 2, 3, 1)
    NHWC_TO_NCHW = (0, 3, 1, 2)

    @save_args
    def __init__(self, input_format, output_format):
        """Construct a converter.

        Args:
            input_format (DataFormat): Input data format.
            output_format (DataFormat): Output data format that inputs will be converted to.
        """
        super(ConvertDataFormat, self).__init__()
        self._input_format = input_format
        self._output_format = output_format

    def can_compose(self, other):
        """
        Determine whether two processors can be composed into a single one.

        Args:
            other (Processor): Other processor instance.

        Returns:
            (Boolean): False - composition not supported.
        """
        return False

    def compose(self, other):
        """Compose two processors into a single one.

        Args:
            other (Processor): Other processor instance.

        Raises:
            RuntimeError: Always raises exception because composition is not supported.
        """
        raise RuntimeError(
            "Compose called on ConvertDataFormat which is not composable"
        )

    def supported_formats(self):
        """Return supported data formats."""
        return [CHANNELS_FIRST, CHANNELS_LAST]

    def process(self, example):
        """
        Convert input data format to match the data_format of this processor.

        Args:
            example (Example): Example with frames in source data_format.

        Returns:
            (Example): Examples with frames converted to target format.
        """
        if self._input_format == self._output_format:
            return example

        frames = example.instances[FEATURE_CAMERA]
        if (
            self._input_format == CHANNELS_LAST
            and self._output_format == CHANNELS_FIRST
        ):
            return Example(
                instances={
                    FEATURE_CAMERA: tf.transpose(a=frames, perm=self.NHWC_TO_NCHW)
                },
                labels=example.labels,
            )
        if (
            self._input_format == CHANNELS_FIRST
            and self._output_format == CHANNELS_LAST
        ):
            return Example(
                instances={
                    FEATURE_CAMERA: tf.transpose(a=frames, perm=self.NCHW_TO_NHWC)
                },
                labels=example.labels,
            )
        raise RuntimeError(
            "Unhandled conversion - from: {} to {}".format(
                self._input_format, self._output_format
            )
        )
