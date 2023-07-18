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
"""Modulus Blur Processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import Processor


class Blur(Processor):
    """Base class for blur transforms."""

    @save_args
    def __init__(self, random=True, **kwargs):
        """__init__ function for blur class."""
        super(Blur, self).__init__(**kwargs)
        self.random = random

    def _convolve_filter(self, images, kernels):
        """Convolve a filter channel-wise."""
        image_blurs = []
        for idx in range(images.shape[1]):
            # isolate a channel to blur
            image = tf.cast(tf.expand_dims(images[:, idx, ...], axis=1), tf.float32)
            blurred_channel = tf.nn.conv2d(
                input=image,
                filters=kernels,
                strides=[1, 1, 1, 1],
                data_format="NCHW",
                padding="SAME",
                name="gaussian_blur",
            )
            image_blurs.append(blurred_channel)

        blurred = tf.concat(image_blurs, axis=1)
        return tf.cast(blurred, images.dtype)

    def _make_gaussian_kernel(self, size, std):
        """Make 2D gaussian Kernel for convolution.

        see:
        https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
        """
        if self.random and size > 1:
            size = tf.random.uniform(minval=1, maxval=size, dtype=tf.int32, shape=[])
            size = tf.cast(size, dtype=tf.float32)
            std = tf.random.uniform(minval=0, maxval=std, dtype=tf.float32, shape=[])

        if std is None or std == 0:
            # Set std if not specified.
            std = (
                tf.multiply(tf.multiply(tf.cast(size, tf.float32) - 1, 0.5) - 1, 0.3)
                + 0.8
            )
        d = tf.compat.v1.distributions.Normal(
            tf.cast(0.0, tf.float32), tf.cast(std, tf.float32)
        )
        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum("i,j->ij", vals, vals)
        gauss_kernel /= tf.reduce_sum(input_tensor=gauss_kernel)
        gauss_kernel = tf.cast(gauss_kernel, tf.float32)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        kernels = gauss_kernel
        return kernels

    def _gaussian_blur(self, images, size, std):
        """Make a gaussian blur of the image."""
        kernels = self._make_gaussian_kernel(size, std)
        return self._convolve_filter(images, kernels)

    def call(self, images, size=1, std=0, prob=0.5):
        """Blur the image.

        Args:
            images (tensor): A tensor of images in NCHW format.
            size (int): The size of the gaussian filter for blurring
                If random, then a filter size will be picked uniformly
                from the range [1, size].
            std (float): The standard deviation of the gaussian filter
                for blurring. If random then the standard deviation will
                be picked uniformly from the range [0, std].
            prob (float): The probability of applying the blur to the image.
                Only applicable if Random, otherwise the blur is always applied.
        Outputs:
            The blurred image.
        """
        assert size >= 1, "Gaussian Kernel size must be positive integer."
        if self.random:
            application_prob = tf.random.uniform(shape=[], maxval=1.0)
            no_aug_cond = tf.greater(application_prob, prob)
            return tf.cond(
                pred=no_aug_cond,
                true_fn=lambda: images,
                false_fn=lambda: self._gaussian_blur(images, size=size, std=std),
            )
        return self._gaussian_blur(images, size=size, std=std)
