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
"""Modulus Pixel Removal Processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import Processor


class PixelRemoval(Processor):
    """Base class for blur transforms."""

    @save_args
    def __init__(self, random=True, **kwargs):
        """__init__ function for pixel removal transformation."""
        super(PixelRemoval, self).__init__(**kwargs)
        self.random = random

    def _sample(self, dist, shape):
        """A wrapper to make it possible to Mock the sample function."""
        return dist.sample((shape[0], 1, shape[2], shape[3]))

    def make_selection_condition(self, pct, max_block, shape):
        """Make an image mask with uniformly distributed patches."""

        # define a standard normal distribution
        dist = tf.compat.v1.distributions.Normal(loc=0.0, scale=1.0)

        # make a tensor of samples that is unique for each image
        samples = self._sample(dist, (shape[0], 1, shape[2], shape[3]))

        # pass a gaussian filter over the pixels to get patches
        samples = self._uniform_blur(samples, size=max_block)

        # get an array of probabilities
        probs = 1 - dist.cdf(samples)
        probs = tf.tile(probs, [1, 3, 1, 1])

        if self.random:
            random_pct = tf.greater(pct, 0)
            pct = tf.cond(
                pred=random_pct,
                true_fn=lambda: tf.random.uniform(
                    shape=[], minval=0, maxval=pct, dtype=tf.float32
                ),
                false_fn=lambda: pct,
            )
        comparison = tf.less(probs, pct)
        return comparison

    def _make_uniform_kernel(self, size):
        """Make a kernel of all the same number."""
        # assume 3 channel image
        size = tf.constant(size, dtype=tf.int32)
        if self.random:
            random_size = tf.greater(size, 1)
            size = tf.cond(
                pred=random_size,
                true_fn=lambda: tf.random.uniform(
                    shape=[], minval=1, maxval=size, dtype=tf.int32
                ),
                false_fn=lambda: size,
            )
        kernel = tf.ones((3, size, size), dtype=tf.float32)
        kernel = kernel / tf.cast(size, dtype=tf.float32)
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
        return kernel

    def _uniform_blur(self, images, **kwargs):
        """Uniformly blur the image."""
        kernels = self._make_uniform_kernel(**kwargs)
        return self._convolve_filter(images, kernels)

    def _convolve_filter(self, images, kernels):
        """Convolve a filter channel-wise."""
        image_blurs = []
        for idx in range(images.shape[1]):
            # isolate a channel to blur
            image = tf.expand_dims(images[:, idx, ...], axis=1)
            blurred_channel = tf.nn.conv2d(
                input=image,
                filters=kernels[idx],
                strides=[1, 1, 1, 1],
                data_format="NCHW",
                padding="SAME",
                name="gaussian_blur",
            )
            image_blurs.append(blurred_channel)

        blurred = tf.concat(image_blurs, axis=1)
        return blurred

    def call(self, images, pct=0.2, max_block=1, prob=1.0):
        """Call function for BlurTransform.

        Args:
            Images (ndarray/tensor): An np array or tensor of images in the format (NCHW).
            pct (float): The percentage of pixels to drop in the image.
            max_block (int): The largest size of area to be taken out in one chunk.
            prob (float): The probability of applying the augmentation. Only
                applicable if random.

        Outputs:
            The image with chunks blacked out,
        """
        condition = self.make_selection_condition(pct, max_block, images.shape)
        masked = tf.compat.v1.where(condition, tf.zeros_like(images), images)
        if self.random:
            application_prob = tf.random.uniform(shape=[], maxval=1.0)
            no_aug_cond = tf.greater(application_prob, prob)
            return tf.cond(
                pred=no_aug_cond, true_fn=lambda: images, false_fn=lambda: masked
            )
        return masked
