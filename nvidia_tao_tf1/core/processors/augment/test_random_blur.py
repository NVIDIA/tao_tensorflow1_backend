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

"""Tests for RandomBlur class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from mock import MagicMock
import numpy as np
import pytest
import tensorflow as tf
from nvidia_tao_tf1.core.processors.augment.random_blur import RandomBlur


test_dir = "nvidia_tao_tf1/core/processors/augment/test_data/random_blur/"
test_inputs = [
    (5, 5, 0.2, 5, "_0"),
    # std
    (5, 7.5, 0.2, 5, "_1"),
    (5, 10, 0.2, 5, "_2"),
    # size
    (15, 5, 0.2, 5, "_3"),
    (25, 5, 0.2, 5, "_4"),
    # blur_max_block
    (5, 5, 0.2, 35, "_5"),
    (5, 5, 0.2, 65, "_6"),
    # blur_prob
    (5, 5, 0.6, 5, "_7"),
    (5, 5, 1.0, 5, "_8"),
]


@pytest.mark.parametrize("width", [160])
@pytest.mark.parametrize("height", [240])
@pytest.mark.parametrize("size,std,blur_prob,blur_max_block,post", test_inputs)
def test_random_blur(width, height, size, std, blur_prob, blur_max_block, post, tmpdir):
    """Iterate through every augmentation and run it.

    Load a correctly augmented image to compare against.
    """
    transform = RandomBlur(random=False)

    mocked_noise = np.load(test_dir + "mocked_noise.npy")
    transform.pixel_remover._sample = MagicMock(return_value=mocked_noise)
    sess = tf.compat.v1.Session()
    test_img = cv2.imread(test_dir + "test_image.jpg")
    test_img = cv2.resize(test_img, (width, height))
    test_img = np.transpose(test_img, [2, 0, 1])
    test_img = np.expand_dims(test_img, 0)
    test_img = test_img.astype(float) / 255.0
    aug_img = sess.run(
        transform(
            test_img,
            size=size,
            std=std,
            prob=0.5,
            blur_max_block=blur_max_block,
            blur_pct=blur_prob,
        )
    )
    filename = test_dir + "random_gaussian_blur" + post + ".npy"
    aug_img = np.squeeze(aug_img, 0)
    aug_img = np.transpose(aug_img, [1, 2, 0])
    aug_img = (aug_img * 255).astype(np.dtype("int8"))
    target_img = np.load(filename)
    np.testing.assert_allclose(aug_img, target_img, atol=1.0)


test_inputs_random = [
    (5, 0, 0.2, 5, 0.0),
    # std
    (5, 0.1, 0.2, 5, 1.0),
    (5, 10.0, 0.2, 5, 0.5),
    # size
    (1, 500, 0.2, 5, 0.8),
    (100, 5, 0.2, 5, 1.0),
    # blur_max_block
    (5, 5, 0.2, 1, 1.0),
    (5, 5, 0.2, 100, 1.0),
    # blur_pct
    (5, 5, 0.0, 5, 0.5),
    (5, 5, 1.0, 5, 0.8),
]


@pytest.mark.parametrize("width", [160])
@pytest.mark.parametrize("height", [240])
@pytest.mark.parametrize("size,std,blur_pct,blur_max_block,prob", test_inputs_random)
def test_random_pixel_removal(
    width, height, size, std, blur_max_block, blur_pct, prob, tmpdir
):
    """Run random augmentations to make sure they work as expected."""
    transform = RandomBlur(random=True)
    sess = tf.compat.v1.Session()
    test_img = cv2.imread(test_dir + "test_image.jpg")
    test_img = cv2.resize(test_img, (width, height))
    test_img = np.transpose(test_img, [2, 0, 1])
    test_img = np.expand_dims(test_img, 0)
    test_img = test_img.astype(np.float32) / 255.0
    aug_img = sess.run(
        transform(
            test_img,
            size=size,
            std=std,
            blur_max_block=blur_max_block,
            blur_pct=blur_pct,
            prob=prob,
        )
    )
    if prob == 0.0:
        np.testing.assert_equal(aug_img, test_img)
