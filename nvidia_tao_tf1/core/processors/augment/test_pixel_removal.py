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

"""Tests for PixelRemoval class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from mock import MagicMock
import numpy as np
import pytest
import tensorflow as tf
from nvidia_tao_tf1.core.processors.augment.pixel_removal import PixelRemoval


test_dir = "nvidia_tao_tf1/core/processors/augment/test_data/pixel_removal/"
test_inputs = [
    (0.025, 5, "_0"),
    # prob
    (0.275, 5, "_1"),
    (0.525, 5, "_2"),
    # max_block
    (0.025, 10, "_3"),
    (0.025, 15, "_4"),
]


@pytest.mark.parametrize("width", [160])
@pytest.mark.parametrize("height", [240])
@pytest.mark.parametrize("prob,max_block,post", test_inputs)
def test_pixel_removal(width, height, prob, max_block, post, tmpdir):
    """Iterate through every augmentation and run it.

    Load a correctly augmented image to compare against.
    """
    transform = PixelRemoval(random=False)

    mocked_noise = np.load(test_dir + "mocked_noise.npy")
    transform._sample = MagicMock(return_value=mocked_noise)
    sess = tf.compat.v1.Session()
    test_img = cv2.imread(test_dir + "test_image.jpg")
    test_img = cv2.resize(test_img, (width, height))
    test_img = np.transpose(test_img, [2, 0, 1])
    test_img = np.expand_dims(test_img, 0)
    test_img = test_img.astype(float) / 255.0
    aug_img = sess.run(transform(test_img, max_block=max_block, pct=prob))
    filename = test_dir + "pixel_removal" + post + ".npy"
    aug_img = np.squeeze(aug_img, 0)
    aug_img = np.transpose(aug_img, [1, 2, 0])
    aug_img = (aug_img * 255).astype(np.dtype("int8"))
    target_img = np.load(filename)
    np.testing.assert_allclose(aug_img, target_img, atol=1.0)


test_inputs_random = [
    (0.00, 1, 0.0),
    (0.00, 1, 1.0),
    (0.275, 2, 0.5),
    (0.525, 10, 0.7),
    (1.00, 20, 1.0),
]


@pytest.mark.parametrize("width", [160])
@pytest.mark.parametrize("height", [240])
@pytest.mark.parametrize("pct,max_block, prob", test_inputs_random)
def test_random_pixel_removal(width, height, pct, max_block, prob, tmpdir):
    """Run random augmentations to make sure they work as expected."""
    transform = PixelRemoval(random=True)
    sess = tf.compat.v1.Session()
    test_img = cv2.imread(test_dir + "test_image.jpg")
    test_img = cv2.resize(test_img, (width, height))
    test_img = np.transpose(test_img, [2, 0, 1])
    test_img = np.expand_dims(test_img, 0)
    test_img = test_img.astype(float) / 255.0
    aug_img = sess.run(transform(test_img, max_block=max_block, pct=pct, prob=prob))
    if prob == 0.0:
        np.testing.assert_equal(aug_img, test_img)
