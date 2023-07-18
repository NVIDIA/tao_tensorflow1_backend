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
"""Processor for applying a Modulus Transform to input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.processors import ColorTransform, SpatialTransform
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import data_format as modulus_data_format, Transform


class ColorTransformer(Processor):
    """Processor for applying a Modulus color transform to input."""

    def __init__(
        self, transform, min_clip=0.0, max_clip=255.0, data_format=None, **kwargs
    ):
        """Construct processor that uses a Transform instance to transform input tensors.

        Args:
            transform (Transform): Input Transform instance that defines a set of transformations
                to be applied to an input tensor.
            min_clip (float): Value to clip all minimum numbers too.
            max_clip (float): Value to clip all maximum numbers to.
            data_format (string): A string representing the dimension ordering of the input
                images, must be one of 'channels_last' (NHWC) or 'channels_first' (NCHW).
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(ColorTransformer, self).__init__(**kwargs)
        if not isinstance(transform, Transform):
            raise TypeError(
                "Expecting an argument of type 'Transform', "
                "given: {}.".format(type(transform).__name__)
            )
        self._data_format = (
            data_format if data_format is not None else modulus_data_format()
        )
        self._transform = transform
        self._transform_processor = ColorTransform(
            min_clip=min_clip, max_clip=max_clip, data_format=self._data_format
        )

    def call(self, applicant):
        """Process tensors by applying transforms according to their data types.

        Args:
            applicant (Tensor): Input tensor to be transformed.

        Returns:
            Tensor: Transformed result tensor.
        """
        input_shape = tf.shape(input=applicant)
        batch_size = input_shape[0]
        ctms = tf.tile(
            tf.expand_dims(self._transform.color_transform_matrix, axis=0),
            [batch_size, 1, 1],
        )
        return self._transform_processor(applicant, ctms=ctms)


class SpatialTransformer(Processor):
    """Processor for applying a Modulus spatial transform to input."""

    def __init__(
        self,
        transform,
        method="bilinear",
        background_value=0.5,
        verbose=False,
        data_format=None,
        **kwargs
    ):
        """Construct processor that uses a Transform instance to transform input tensors.

        Args:
            transform (Transform): Input Transform instance that defines a set of transformations
                to be applied to an input tensor.
            method (string): Sampling method used. Can be 'bilinear' or 'bicubic'.
            background_value (float): The value the background canvas should have.
            verbose (bool): Toggle verbose output during processing.
            data_format (string): A string representing the dimension ordering of the input
                images, must be one of 'channels_last' (NHWC) or 'channels_first' (NCHW).
            kwargs (dict): keyword arguments passed to parent class.

        Returns:
            Transform: Final Transform instance with either cropping or scaling applied.
        """
        super(SpatialTransformer, self).__init__(**kwargs)
        if not isinstance(transform, Transform):
            raise TypeError(
                "Expecting an argument of type 'Transform', "
                "given: {}.".format(type(transform).__name__)
            )
        self._data_format = (
            data_format if data_format is not None else modulus_data_format()
        )
        self._transform = transform
        self._spatial_transform = SpatialTransform(
            method=method,
            background_value=background_value,
            data_format=self._data_format,
            verbose=verbose,
        )

    def call(self, applicant):
        """Process tensors by applying transforms according to their data types.

        Args:
            applicant (Tensor): Input tensor to be transformed.

        Returns:
            Tensor: Transformed result tensor.
        """
        input_shape = tf.shape(input=applicant)
        batch_size = input_shape[0]

        stms = tf.tile(
            tf.expand_dims(self._transform.spatial_transform_matrix, axis=0),
            [batch_size, 1, 1],
        )

        output = self._spatial_transform(
            applicant,
            stms=stms,
            shape=(
                int(self._transform.canvas_shape.height),
                int(self._transform.canvas_shape.width),
            ),
        )
        return output
