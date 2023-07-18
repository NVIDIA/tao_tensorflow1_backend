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
"""Modulus Additive Noise Processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import Processor


class AdditiveNoise(Processor):
    """Additive noise transformation class."""

    @save_args
    def __init__(self, random=True, **kwargs):
        """__init__ method."""
        super(AdditiveNoise, self).__init__(**kwargs)
        self.random = random

    def call(self, images, var, prob=1.0):
        """Add random gaussian noise to each channel of the image.

        Args:
            images (tensor): An array of images in NCHW format.
            var (float): The variance of the noise to be added to the image.
                If random, the variance is chosen uniformly from [0, var].
            prob (float): The probability of applying the augmentation to
                the image. Only applicable if random.

        Outputs:
            The image with noise added.
        """
        loc = tf.constant(0.0, dtype=images.dtype)
        var = tf.constant(var, dtype=images.dtype)
        if self.random:
            var = tf.random.uniform(shape=[], maxval=1.0, dtype=images.dtype) * var

        dist = tf.compat.v1.distributions.Normal(loc=loc, scale=var)
        samples = dist.sample(images.shape)

        if self.random:
            application_prob = tf.random.uniform(shape=[], maxval=1.0)
            no_aug_cond = tf.greater(application_prob, prob)
            return tf.cond(
                pred=no_aug_cond,
                true_fn=lambda: images,
                false_fn=lambda: images + samples,
            )

        return images + samples
