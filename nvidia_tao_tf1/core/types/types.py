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
"""Modulus standard types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

Example = namedtuple("Example", ["instances", "labels"])

Canvas2D = namedtuple("Canvas2D", ["height", "width"])

Transform = namedtuple(
    "Transform", ["canvas_shape", "color_transform_matrix", "spatial_transform_matrix"]
)


class DataFormat(object):
    """Structure to hold the dataformat types.

    NOTE: HumanLoop's export format of fp16 is typically CHW (channels first) and BGR,
        while OpenCV defaults to HWC (channels last) and BGR.
    """

    CHANNELS_FIRST = "channels_first"
    CHANNELS_LAST = "channels_last"

    @staticmethod
    def convert(tensor, from_format, to_format):
        """Helper function for handling data format conversions for 3D and 4D tf tensors.

        Args:
            tensor (Tensor): A 4D or 3D input tensor.
            from_format (str): Data format of input tensor, can be either 'channels_first'
                (NCHW / CHW) or 'channels_last' (NHWC / HWC).
            to_format (str): Data format of output tensor, can be either 'channels_first'
                (NCHW / CHW) or 'channels_last' (NHWC / HWC).

        Returns:
            Tensor: Transposed input tensor with 'to_format' as the data format.
        """
        assert from_format in [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
        assert to_format in [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
        ndims = tensor.shape.ndims
        if ndims not in [3, 4]:
            raise NotImplementedError(
                "Input tensor should be of rank 3 or 4, given {}.".format(ndims)
            )

        if from_format == to_format:
            return tensor
        if (
            from_format == DataFormat.CHANNELS_FIRST
            and to_format == DataFormat.CHANNELS_LAST
        ):
            if ndims == 4:
                return tf.transpose(a=tensor, perm=(0, 2, 3, 1))
            return tf.transpose(a=tensor, perm=(1, 2, 0))
        # Channels last to channels first case.
        if ndims == 4:
            return tf.transpose(a=tensor, perm=(0, 3, 1, 2))
        return tf.transpose(a=tensor, perm=(2, 0, 1))


_DATA_FORMAT = DataFormat.CHANNELS_FIRST


def set_data_format(data_format):
    """Set the default Modulus data format."""
    global _DATA_FORMAT  # pylint: disable=W0603
    if data_format not in [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]:
        raise ValueError("Data format `{}` not supported.".format(data_format))
    _DATA_FORMAT = data_format


def data_format():
    """Get the default Modulus data format."""
    global _DATA_FORMAT  # pylint: disable=W0602,W0603
    return _DATA_FORMAT
