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
"""Processor for applying random blur transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.processors.processors import Processor

import tensorflow as tf


class RandomBlur(Processor):
    """Random blur processor."""

    def __init__(self, blur_choices, blur_probability, channels, **kwargs):
        """Construct a random blur processor.

        Args:
            blur_choices (list): Choices of odd integer kernel size for blurring.
            blur_probability (float): Probability that a blur will occur.
        """
        super(RandomBlur, self).__init__(**kwargs)
        for size in blur_choices:
            if size % 2 == 0:
                raise ValueError("RandomBlur.blur_choices ({}) contains an even "
                                 "kernel size ({}).".format(blur_choices, size))
            if size < 1:
                raise ValueError("RandomBlur.blur_choices ({}) contains an invalid "
                                 "kernel size ({}).".format(blur_choices, size))
        if blur_probability < 0.0 or blur_probability > 1.0:
            raise ValueError(
                "RandomBlur.blur_probability ({}) is not within the range [0, 1].".format(
                    blur_probability))

        self._blur_choices_list = list(blur_choices)
        self._blur_choices = tf.convert_to_tensor(self._blur_choices_list, dtype=tf.int32)
        self._blur_probability = blur_probability
        self._channels = channels

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomBlur(blur_choices={}, blur_probability={})".format(self._blur_choices,
                                                                         self._blur_probability)

    def _build(self, *args, **kwargs):
        """Initialize random variables used for op.

        The build function should be used when wanting to apply the same random blur to multiple
        images.
        """
        self._kernel = self._get_random_kernel(self._channels)
        blur_probability = tf.random_uniform([], minval=0.0, maxval=1.0)
        self._should_blur = tf.less(blur_probability, self._blur_probability)

    def call(self, image, output_height, output_width):
        """Return a blurred image.

        Args:
            image (Tensor): Image to be blurred (HWC).
            output_height (int): Output image height.
            output_width (int): Output image width.

        Returns:
            output_image (Tensor): Image that may blurred.
        """
        if self._kernel is None:
            return image

        batch = tf.stack([image])
        blurred = tf.nn.depthwise_conv2d(batch, self._kernel, strides=[1, 1, 1, 1],
                                         padding='VALID', data_format='NHWC')

        output_image = tf.cond(self._should_blur, lambda: blurred, lambda: batch)
        output_image = tf.squeeze(output_image, axis=0)

        output_image = tf.image.resize_images(output_image, (output_height, output_width),
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return output_image

    def _get_random_kernel(self, channels=3):
        """Generate random average kernel.

        Returns:
            kernel (Tensor float32): Average kernel for 3 channel images.
                                     Intended to be used with conv2d.
            channels (int64): Channels of kernel, default to 3.
        """
        if not self._blur_choices_list:
            return None

        random_index = tf.random_uniform([], minval=0, maxval=len(self._blur_choices_list),
                                         dtype=tf.int32)
        size = self._blur_choices[random_index]
        kernel = tf.ones((size, size, channels, 1), dtype=tf.float32)
        kernel /= (tf.cast(size, dtype=tf.float32) ** 2)

        return kernel
