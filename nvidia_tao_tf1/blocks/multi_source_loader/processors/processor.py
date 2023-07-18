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
"""Interface that a pipeline processors must implement."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod, abstractproperty

from nvidia_tao_tf1.core.coreobject import AbstractTAOObject


class Processor(AbstractTAOObject):
    """Interface that a pipeline processors must implement."""

    def __init__(self):
        """Construct a processor."""
        super(Processor, self).__init__()
        self._data_format = None

    def __str__(self):
        """Returns a string representation of this processor."""
        return type(self).__name__

    @abstractproperty
    def supported_formats(self):
        """Data formats supported by this processor.

        Returns:
            data_formats (list of 'DataFormat'): Input data formats that this processor supports.
        """

    @property
    def data_format(self):
        """Data format of the image/frame tensors to process."""
        return self._data_format

    @data_format.setter
    def data_format(self, value):
        self._data_format = value

    @abstractmethod
    def can_compose(self, other):
        """
        Determine whether two processors can be composed into a single one.

        Args:
            other (Processor): Other processor instance.

        Returns:
            (Boolean): True if this processor knows how to compose the other processor.
        """
        raise NotImplementedError("Processors.can_compose not implemented.")

    @abstractmethod
    def compose(self, other):
        """Compose two processors into a single one.

        Example:
            Given a list of processors:

                processors = [Processor(can_compose=False), Processor(can_compose=True),
                              Processor(can_compose=True)]

            The two consecutive processors at the end of the list can be combined into a single
            processor to reduce compute:

                composed = [Processor(can_compose=False), Processor(can_compose=True)]

            The main use case for this interface are processors based on spatial and color transform
            matrices. The caller (e.g. Pipeline) can traverse a list of processors and compose
            consecutive composable processors into a single processor by calling their compose
            method.

        Args:
            other (Processor): Other processor instance.

        Returns:
            processor (Processor): A new processor that performs the same processing as the
                                   individual processors applied in sequence.
        """
        raise NotImplementedError("ComposableProcessor.compose not implemented")

    @abstractmethod
    def process(self, example):
        """
        Process an example.

        Args:
            example (Example): Example with frames in format specified by data_format.

        Returns:
            (Example): Processed example.
        """
        raise NotImplementedError("Processors.process not implemented.")
