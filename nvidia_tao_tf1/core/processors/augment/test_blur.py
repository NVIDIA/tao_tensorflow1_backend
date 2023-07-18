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

"""Tests for Blur class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import pytest
import tensorflow as tf
import nvidia_tao_tf1.core
from nvidia_tao_tf1.core.processors.augment.blur import Blur


test_dir = "nvidia_tao_tf1/core/processors/augment/test_data/blur/"
test_inputs = [
    (4, 7, "_max_size_0"),
    (7, 7, "_max_size_1"),
    (10, 7, "_max_size_2"),
    (7, 5, "_max_std_0"),
    (7, 10, "_max_std_1"),
    (7, 15, "_max_std_2"),
]


@pytest.mark.parametrize("width", [160])
@pytest.mark.parametrize("height", [240])
@pytest.mark.parametrize("size,std,post", test_inputs)
@pytest.mark.parametrize("random", [True, False])
def test_blur(width, height, size, std, post, random, tmpdir):
    """Iterate through every augmentation and run it.

    Load a correctly augmented image to compare against.
    """
    transform = Blur(random=random)
    np.random.seed(17)
    nvidia_tao_tf1.core.utils.set_random_seed(17)
    tf.compat.v1.set_random_seed(17)
    filegroup = "uniform_gaussian_blur"
    if random:
        filegroup = "random_uniform_gaussian_blur"
    sess = tf.compat.v1.Session()
    test_img = cv2.imread(test_dir + "test_image.jpg")
    test_img = cv2.resize(test_img, (height, width))
    test_img = np.transpose(test_img, [2, 0, 1])
    test_img = np.expand_dims(test_img, 0)
    test_img = test_img.astype(float) / 255.0
    aug_img = sess.run(transform(test_img, size=size, std=std))
    filename = os.path.join(test_dir, filegroup + post + ".npy")
    aug_img = np.squeeze(aug_img, 0)
    aug_img = np.transpose(aug_img, [1, 2, 0])
    aug_img = (aug_img * 255).astype(np.dtype("int8"))
    target_img = np.load(filename)

    np.testing.assert_allclose(aug_img, target_img, atol=1.0)


test_inputs_random = [(1, 7, 0.0), (1, 0, 1.0), (2, 1, 0.5), (10, 0.2, 0.7)]


@pytest.mark.parametrize("width", [160])
@pytest.mark.parametrize("height", [240])
@pytest.mark.parametrize("size,std,prob", test_inputs_random)
def test_random_pixel_removal(width, height, size, std, prob, tmpdir):
    """Run random augmentations to make sure they work as expected."""
    transform = Blur(random=True)
    sess = tf.compat.v1.Session()
    test_img = cv2.imread(test_dir + "test_image.jpg")
    test_img = cv2.resize(test_img, (width, height))
    test_img = np.transpose(test_img, [2, 0, 1])
    test_img = np.expand_dims(test_img, 0)
    test_img = test_img.astype(np.float32) / 255.0
    aug_img = sess.run(transform(test_img, size=size, std=std, prob=prob))
    if prob == 0.0:
        np.testing.assert_equal(aug_img, test_img)
