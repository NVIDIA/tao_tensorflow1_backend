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
"""Processor for applying random translation augmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def gaussian_kernel(size, mean=0, stddev=None):
    """Return a Gaussian kernel as a Gaussian blurring filter.

    Args:
        size (int): The filter size.
        mean (float): Mean of the normal distribution.
        stddev (float): Std of the normal distribution.
    Return:
        (tensor): A float tensor of shape [size, 1].
    """
    # If stddev is not given, infer it from the filter size.
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.
    # https://docs.opencv.org/2.4/modules/imgproc/doc/
    # filtering.html?highlight=gaussianblur#Mat%20
    # getGaussianKernel(int%20ksize,%20double%20sigma,%20int%20ktype)
    if stddev is None:
        stddev = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

    normal_distribution = tf.compat.v1.distributions.Normal(mean, stddev)

    # When size  = 5,
    # vals = g(-2), g(-1), g(0), g(1), g(2), where
    # g(x) = 1/(sqrt(2*pi)*sigma) * exp(-x^2/(2*sigma^2)).
    # array([0.05399096, 0.24197073, 0.3989423 , 0.24197073, 0.05399096].
    range_x = tf.range(start=0, limit=size, dtype=tf.int32) - (size - 1) // 2
    vals = normal_distribution.prob(tf.cast(range_x, tf.float32))
    kernel = tf.reshape(vals, [size, 1])

    return kernel / tf.reduce_sum(input_tensor=kernel)
