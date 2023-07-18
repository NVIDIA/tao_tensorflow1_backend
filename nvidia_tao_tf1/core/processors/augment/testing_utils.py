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
"""Utility functions used for tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange
import tensorflow as tf


def assert_truncated_normal_distribution(values, mean, stddev):
    """Check that ``values`` are from truncated normal distribution with ``mean`` and ``stddev``."""
    # Check that bounds fit. Truncated normal distribution cuts off values further than
    # two times stddev away from mean.
    assert np.max(values) <= mean + stddev * 2.0
    assert np.min(values) >= mean - stddev * 2.0

    # Standard deviation of estimate of a mean is stddev/sqrt(num_samples), we're using three times
    # the standard deviation as a tolerance threshold, just to be safe.
    tolerance = stddev / np.sqrt(len(values)) * 3

    # Check that the sample mean fits.
    assert np.isclose(np.mean(values), mean, atol=tolerance)


def assert_uniform_distribution(values, min_bound, max_bound):
    """Check that ``values`` are from uniform distribution with ``max`` and ``min`` bounds."""
    # Check that bounds fit.
    assert np.max(values) <= max_bound
    assert np.min(values) >= min_bound

    # Calculate stddev of uniform distribution.
    stddev = (max_bound - min_bound) / np.sqrt(12)
    # Standard deviation of estimate of a mean is stddev/sqrt(num_samples), we're using four times
    # the standard deviation as a tolerance threshold, just to be safe.
    tolerance = stddev / np.sqrt(len(values)) * 4

    # Check that sample mean fits.
    assert np.isclose(np.mean(values), (max_bound + min_bound) / 2.0, atol=tolerance)


def assert_bernoulli_distribution(values, p):
    """Check that ``values`` are from bernoulli with ``p`` probability of event."""
    # Calculate stddev of bernoulli distribution.
    stddev = np.sqrt(p * (1 - p))
    if type(values[0]) is np.ndarray:
        num_values = sum([len(v) for v in values])
    else:
        num_values = len(values)
    # Standard deviation of estimate of a mean is stddev/sqrt(num_samples).
    tolerance = stddev / np.sqrt(num_values) * 2.0

    # Check that events are generated with correct probability.
    event_count = np.array(values).sum()
    event_probability = float(event_count) / float(num_values)
    assert np.isclose(event_probability, p, atol=tolerance)


def sample_tensors(tensors, num_samples):
    """Sample ``num_samples`` values of list of tensors."""
    samples = [list() for _ in xrange(len(tensors))]
    with tf.compat.v1.Session() as sess:
        for _ in xrange(num_samples):
            new_samples = sess.run(tensors)
            # This "zips" ``new_samples`` with ``samples``. ``new_samples`` is a list of values of
            # length N. ``samples`` is a list of N lists. We're appending first item from
            # ``new_samples`` to first list of ``samples``.
            samples = [
                old_samples + [new_sample]
                for old_samples, new_sample in zip(samples, new_samples)
            ]
    return samples
