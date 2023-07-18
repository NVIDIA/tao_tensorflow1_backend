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

"""Tests for Additive Noise class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf
from nvidia_tao_tf1.core.processors.augment.additive_noise import AdditiveNoise
from nvidia_tao_tf1.core.utils import set_random_seed


@pytest.mark.parametrize("var", [0.05, 0.3, 0.55])
@pytest.mark.parametrize("width", [450, 960])
@pytest.mark.parametrize("height", [960, 450])
def test_additive_gaussian_noise(var, width, height, tmpdir):
    """Test the additive gaussian noise class."""
    # Use fixed seed to remove test flakiness.
    set_random_seed(42)
    transform = AdditiveNoise(random=False)
    test_image = np.random.random((1, 3, width, height))
    aug_image = transform(test_image, var)
    sess = tf.compat.v1.Session()
    aug_image = sess.run(aug_image)
    diff = test_image - aug_image
    diff = diff.flatten()
    tolerance = var / np.sqrt(len(diff)) * 3
    assert np.isclose(np.mean(diff), 0, atol=tolerance)


@pytest.mark.parametrize("width", [160])
@pytest.mark.parametrize("height", [240])
@pytest.mark.parametrize("var, prob", [(0, 1.0), (0.5, 0.0), (0.2, 1.0), (0.5, 1)])
def test_random_pixel_removal(width, height, var, prob, tmpdir):
    """Run random augmentations to make sure they work as expected."""
    transform = AdditiveNoise(random=True)
    sess = tf.compat.v1.Session()
    test_img = np.random.random((1, 3, width, height))
    aug_img = sess.run(transform(test_img, var=var, prob=prob))
    if prob == 0.0 or var == 0:
        np.testing.assert_equal(aug_img, test_img)
