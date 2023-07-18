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
"""Processor for applying random gamma transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.processors.processors import Processor

import tensorflow as tf


class RandomGamma(Processor):
    """Random gamma processor."""

    def __init__(self, gamma_type, gamma_mu, gamma_std, gamma_max, gamma_min, gamma_probability,
                 **kwargs):
        """Construct a random gamma processor.

        Args:
            gamma_type (string): Describes type of random sampling for gamma ['normal', 'uniform'].
            gamma_mu (float): Mu for gamma normal distribution.
            gamma_std (float): Standard deviation for gamma normal distribution.
            gamma_max (float): Maximum value for gamma uniform distribution.
            gamma_min (float): Minimum value for gamma uniform distribution.
            gamma_probability (float): Probability that a gamma correction will occur.
        """
        super(RandomGamma, self).__init__(**kwargs)
        if gamma_type not in ('normal', 'uniform'):
            raise ValueError("RandomGamma.gamma_type ({}) is not one of "
                             "['normal', 'uniform'].".format(gamma_type))
        if gamma_mu < 0:
            raise ValueError("RandomGamma.gamma_mu ({}) is not positive.".format(gamma_mu))
        if gamma_std < 0:
            raise ValueError("RandomGamma.gamma_std ({}) is not positive.".format(gamma_std))
        if gamma_min < 0:
            raise ValueError("RandomGamma.gamma_min ({}) is not positive.".format(gamma_min))
        if gamma_max < 0:
            raise ValueError("RandomGamma.gamma_max ({}) is not positive.".format(gamma_max))
        if gamma_max < gamma_min:
            raise ValueError("RandomGamma.gamma_max ({}) is less than "
                             "RandomGamma.gamma_min ({}).".format(gamma_max, gamma_min))
        if gamma_max == gamma_min and gamma_max != 1.0:
            raise ValueError("RandomGamma.gamma_max ({}) is equal to RandomGamma.gamma_min "
                             "({}) but is not 1.0.".format(gamma_max, gamma_min))
        if gamma_probability < 0.0 or gamma_probability > 1.0:
            raise ValueError(
                "RandomGamma.gamma_probability ({}) is not within the range [0, 1].".format(
                    gamma_probability))

        self._gamma_type = gamma_type
        self._gamma_mu = float(gamma_mu)
        self._gamma_std = float(gamma_std)
        self._gamma_max = float(gamma_max)
        self._gamma_min = float(gamma_min)
        self._gamma_probability = gamma_probability

    def __repr__(self):
        """Return a string representation of the processor."""
        _rep = "RandomGamma(gamma_type={}, gamma_mu={}, gamma_max={}, gamma_std={}, " \
               "gamma_min={}, gamma_probability={})".format(self._gamma_type,
                                                            self._gamma_mu,
                                                            self._gamma_std,
                                                            self._gamma_max,
                                                            self._gamma_min,
                                                            self._gamma_probability)
        return _rep

    def _build(self, *args, **kwargs):
        """Initialize random variables used for op.

        The build function should be used when wanting to apply the same random gamma to multiple
        images.
        """
        gamma_probability = tf.random_uniform([], minval=0.0, maxval=1.0)
        self._should_gamma = tf.less(gamma_probability, self._gamma_probability)
        if self._gamma_type == 'uniform':
            self._random_gamma = tf.random_uniform([], minval=self._gamma_min,
                                                   maxval=self._gamma_max)
        elif self._gamma_type == 'normal':
            val = tf.random_normal([], mean=self._gamma_mu, stddev=self._gamma_std)
            # NOTE: Using absolute value would obtain more useful random gammas as opposed to using
            #       relu: tf.nn.relu(dist.sample([])) which would set all negative gamma to 0.
            # Set all negative gamma to its absolute value.
            self._random_gamma = tf.abs(val)

    def call(self, image):
        """Return a gamma corrected image.

        Args:
            image (Tensor): Image to be gamma corrected (NHWC) or (HWC) or (NCHW) or (CHW).

        Returns:
            output_image (Tensor): Image that may be gamma corrected.
                                   Same data format as input (NHWC) or (HWC) or (NCHW) or (CHW).
        """
        if self._gamma_min == self._gamma_max == 1.0 and self._gamma_type == 'uniform':
            return image

        corrected_image = tf.image.adjust_gamma(image, gamma=self._random_gamma)
        output_image = tf.cond(self._should_gamma, lambda: corrected_image, lambda: image)

        return output_image
