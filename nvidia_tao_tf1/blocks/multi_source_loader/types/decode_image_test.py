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
"""Unit test of tf.io.decode_image function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
from PIL import Image
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.images2d_reference import (
    decode_image
)


def test_jpeg_1_as_1():
    """test of decoding grayscale jpeg image as grayscale."""
    # generate a random image
    f, frame_path = tempfile.mkstemp(suffix=".jpg")
    os.close(f)
    image_data = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    random_image = Image.fromarray(image_data)
    random_image.save(frame_path)
    # read the image
    image_data = np.array(Image.open(frame_path))
    data = tf.io.read_file(frame_path)
    image = decode_image(data, channels=1, extension=".jpg")
    with tf.Session() as sess:
        image_data_tf = sess.run(image)[..., 0]
    assert np.array_equal(image_data, image_data_tf)


def test_png_1_as_1():
    """test of decoding grayscale png image as grayscale."""
    # generate a random image
    f, frame_path = tempfile.mkstemp(suffix=".png")
    os.close(f)
    image_data = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    random_image = Image.fromarray(image_data)
    random_image.save(frame_path)
    # read the image
    image_data = np.array(Image.open(frame_path))
    data = tf.io.read_file(frame_path)
    image = decode_image(data, channels=1, extension=".png")
    with tf.Session() as sess:
        image_data_tf = sess.run(image)[..., 0]
    assert np.array_equal(image_data, image_data_tf)


def test_jpeg_3_as_1():
    """test of decoding RGB jpeg image as grayscale."""
    # generate a random RGB image
    f, frame_path = tempfile.mkstemp(suffix=".jpg")
    os.close(f)
    image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    random_image = Image.fromarray(image_data)
    random_image.save(frame_path)
    # read the image as grayscale
    data = tf.io.read_file(frame_path)
    image = decode_image(data, channels=1, extension=".jpg")
    with tf.Session() as sess:
        image_data_tf = sess.run(image)
    # due to implementation differences, we cannnot match tf and PIL
    # for the converted grayscale images
    # so just check the channel number is 1
    assert image_data_tf.shape[-1] == 1
    # and the output grayscale image is not first channel data
    # of the RGB image
    image_data = np.array(Image.open(frame_path))
    assert not np.array_equal(image_data_tf[..., 0], image_data[..., 0])


def test_png_3_as_1():
    """test of decoding RGB png image as grayscale."""
    # generate a random RGB image
    f, frame_path = tempfile.mkstemp(suffix=".png")
    os.close(f)
    image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    random_image = Image.fromarray(image_data)
    random_image.save(frame_path)
    # read the image as grayscale
    data = tf.io.read_file(frame_path)
    image = decode_image(data, channels=1, extension=".png")
    with tf.Session() as sess:
        image_data_tf = sess.run(image)
    # due to implementation differences, we cannnot match tf and PIL
    # for the converted grayscale images
    # so just check the channel number is 1
    assert image_data_tf.shape[-1] == 1
    # and the output grayscale image is not first channel data
    # of the RGB image
    image_data = np.array(Image.open(frame_path))
    assert not np.array_equal(image_data_tf[..., 0], image_data[..., 0])


def test_jpeg_3_as_3():
    """test of decoding RGB jpeg image as RGB image."""
    # generate a random RGB image
    f, frame_path = tempfile.mkstemp(suffix=".jpg")
    os.close(f)
    image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    random_image = Image.fromarray(image_data)
    random_image.save(frame_path)
    # read the image
    data = tf.io.read_file(frame_path)
    image = decode_image(data, channels=3, extension=".jpg")
    with tf.Session() as sess:
        image_data_tf = sess.run(image)
    # check for the image shape
    # we cannnot guarantee the tf output image and PIL output image
    # are identical due to implementation differences of them.
    assert image_data_tf.shape == (10, 10, 3)


def test_png_3_as_3():
    """test of decoding RGB png image as RGB image."""
    # generate a random RGB image
    f, frame_path = tempfile.mkstemp(suffix=".png")
    os.close(f)
    image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    random_image = Image.fromarray(image_data)
    random_image.save(frame_path)
    # read the image
    data = tf.io.read_file(frame_path)
    image = decode_image(data, channels=3, extension=".png")
    with tf.Session() as sess:
        image_data_tf = sess.run(image)
    # check for the image shape
    # we cannnot guarantee the tf output image and PIL output image
    # are identical due to implementation differences of them.
    assert image_data_tf.shape == (10, 10, 3)
