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

"""A type safer wrapper for handling Keras impicitly defined data_format enums."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from functools import lru_cache

from nvidia_tao_tf1.core.coreobject import TAOObject, save_args


"""
Structure for storing 4D dimension name and index mappings.

4D images are used as batches for non-temporal/non-sequence models. To access e.g. the colors
of a 'channels_first' image tensor you would:

    axis = DataFormat('channels_first').axis_5d
    colors = image[:, axis.channel]

Args:
    batch (int): Index of batch/image/example dimension. This dimension spans [0, batch size).
    row (int): Index of row dimension. This dimension spans [0, image height).
    column (int): Index of column dimension. This dimension spans [0, image width).
    channel (int): Index of channel dimension. This dimension spans [0, image colors channels).
"""
TensorAxis4D = namedtuple("TensorAxis4D", ["batch", "row", "column", "channel"])


"""
Structure for storing 5D dimension name and index mappings.

5D images are used as batches for temporal/sequence models. To access e.g. the colors
of a 'channels_first' image tensor you would:

    axis = DataFormat('channels_first').axis_5d
    colors = image[:, axis.channel]

Args:
    batch (int): Index of batch/image/example dimension. This dimension spans [0, batch size).
    time (int): Index of time dimension. This dimension spans [0, time steps).
    row (int): Index of row dimension. This dimension spans [0, image height).
    column (int): Index of column dimension. This dimension spans [0, image width).
    channel (int): Index of channel dimension. This dimension spans [0, image colors channels).
"""
TensorAxis5D = namedtuple("TensorAxis5D", ["batch", "time", "row", "column", "channel"])


class DataFormat(TAOObject):
    """
    A wrapper class for handling Keras impicitly defined data_format enums.

    Images are represented as 3D tensors in TF. Historically, NVIDIA libraries like CuDNN were
    optimized to work on representations that store color information in the first
    dimension (CHANNELS_FIRST) TensorFlow and other libraries often times store the color
    informatoin in the last dimension (CHANNELS_LAST).
    """

    CHANNELS_FIRST = "channels_first"
    CHANNELS_LAST = "channels_last"
    VALID_FORMATS = [CHANNELS_FIRST, CHANNELS_LAST]

    @save_args
    def __init__(self, data_format):
        """Construct DataFormat object.

        Args:
            data_format (str): Either 'channels_first' or 'channels_last' (i.e. NCHW vs NHWC).

        Raises:
            exception (ValueError): when invalid data_format is provided.
        """
        super(DataFormat, self).__init__()
        if data_format not in self.VALID_FORMATS:
            raise ValueError("Unrecognized data_format '{}'.".format(data_format))

        self._data_format = data_format

    def __str__(self):
        """Return string representation compatible with Keras.

        Returns:
            data_format (str): String representation of this object.
        """
        return self._data_format

    def __eq__(self, other):
        """Compare two data format objecst for equality.

        Returns:
            equal (bool): True if objects represent the same value.
        """
        return self._data_format == other._data_format

    def convert_shape(self, shape, target_format):
        """Converts a shape tuple from one data format to another.

        Args:
            shape (list, tuple): The shape in this data format.
            target_format (DataFormat): The desired data format.

        Returns:
            shape (list, tuple): The shape in the target format.
        """
        if self._data_format == target_format._data_format:
            return shape

        assert 4 <= len(shape) <= 5
        converted = [shape[0], shape[1], 0, 0]
        if len(shape) == 4:
            src_axis = self.axis_4d
            dest_axis = target_format.axis_4d
        else:
            converted.append(0)
            src_axis = self.axis_5d
            dest_axis = target_format.axis_5d
        converted[dest_axis.channel] = shape[src_axis.channel]
        converted[dest_axis.row] = shape[src_axis.row]
        converted[dest_axis.column] = shape[src_axis.column]
        if isinstance(shape, tuple):
            converted = tuple(converted)
        return converted

    @property
    @lru_cache(maxsize=None)
    def axis_4d(self):
        """Return data format dependent axes of 4D tensors.

        Returns:
            (TensorAxis4D): Axes of tensors for this data format.
        """
        if self._data_format == self.CHANNELS_FIRST:
            return TensorAxis4D(batch=0, channel=1, row=2, column=3)

        return TensorAxis4D(batch=0, row=1, column=2, channel=3)

    @property
    @lru_cache(maxsize=None)
    def axis_5d(self):
        """Return data format dependent axes of 5D tensors.

        Returns:
            (TensorAxis5D): Axes of tensors for this data format.
        """
        if self._data_format == self.CHANNELS_FIRST:
            return TensorAxis5D(batch=0, time=1, channel=2, row=3, column=4)

        return TensorAxis5D(batch=0, time=1, row=2, column=3, channel=4)

    def __hash__(self):
        """Get the hash for this data format."""
        return hash(self._data_format)


CHANNELS_FIRST = DataFormat(DataFormat.CHANNELS_FIRST)
CHANNELS_LAST = DataFormat(DataFormat.CHANNELS_LAST)
