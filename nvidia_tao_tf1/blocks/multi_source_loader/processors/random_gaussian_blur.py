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
"""Processor for applying random Gaussian blurring augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.processors.filter2d_processor import (
    Filter2DProcessor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.gaussian_kernel import (
    gaussian_kernel,
)
from nvidia_tao_tf1.core.coreobject import save_args


class RandomGaussianBlur(Filter2DProcessor):
    """GaussianBlur processor that randomly blurs images."""

    @save_args
    def __init__(
        self, min_filter_size=2, max_filter_size=5, max_stddev=None, probability=None
    ):
        """Construct a RandomGaussianBlur processor.

        Args:
            min_filter_size (int): The mininum filter size of a Gassian blur filter.
            max_filter_size (int): The maximum filter size of a Gassian blur filter.
            max_stddev (float): The maximum standard deviation of the Gaussian blur filter's shape.
            probability (float): Probability at which blurring occurs.
        """
        super(RandomGaussianBlur, self).__init__()
        # Set filter_size.
        if min_filter_size < 0:
            raise ValueError(
                "RandomGaussianBlur.min_filter_size ({}) must be an positive integer.".format(
                    min_filter_size
                )
            )
        if max_filter_size < 0:
            raise ValueError(
                "RandomGaussianBlur.max_filter_size ({}) must be an positive integer.".format(
                    max_filter_size
                )
            )
        if min_filter_size > max_filter_size:
            raise ValueError(
                "RandomGaussianBlur.min_filter_size ({}) must not be larger than \
                RandomGaussianBlur.max_filter_size ({}).".format(
                    min_filter_size, max_filter_size
                )
            )
        self._min_filter_size = min_filter_size
        self._max_filter_size = max_filter_size

        self._max_stddev = max_stddev

        # Set probability.
        if probability < 0.0 or probability > 1.0:
            raise ValueError(
                "RandomGaussianBlur.probability ({}) is not within the range "
                "[0.0, 1.0].".format(probability)
            )
        self._probability = probability

    @property
    def probability(self):
        """Probability to apply filters."""
        return self._probability

    def get_filters(self):
        """Return a list of filters.

        Because the filter is separable, each element contains a decomposed filter.
        """
        # Get filter_size.
        filter_size = tf.random.uniform(
            minval=self._min_filter_size,
            maxval=self._max_filter_size + 1,
            dtype=tf.int32,
            shape=[],
        )

        # Set stddev.
        if self._max_stddev is None:
            # Set stddev if not specified.
            stddev = (
                tf.multiply(
                    tf.multiply(tf.cast(filter_size, tf.float32) - 1, 0.5) - 1, 0.3
                )
                + 0.8
            )
        else:
            stddev = tf.random.uniform(
                minval=0, maxval=self._max_stddev, dtype=tf.float32, shape=[]
            )
        # Get gaussian_kernel.
        gaussian_filter_list = []
        # _gaussian_kernel is a tensor with size [filter_size, 1].
        _gaussian_kernel = gaussian_kernel(filter_size, 0.0, stddev)
        gaussian_filter_list.append(_gaussian_kernel)
        gaussian_filter_list.append(tf.reshape(_gaussian_kernel, [1, filter_size]))
        return gaussian_filter_list
