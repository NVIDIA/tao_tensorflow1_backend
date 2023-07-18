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

"""Main test for image_data_source.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import PIL
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader import data_loader
from nvidia_tao_tf1.blocks.multi_source_loader import frame_shape
from nvidia_tao_tf1.blocks.multi_source_loader import types
from nvidia_tao_tf1.blocks.multi_source_loader.sources import image_data_source
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


@pytest.fixture(scope="session")
def _image_files():
    return ["image1.jpg", "image2.jpg"]


def _create_images(tmpdir, image_files):
    image_paths = []
    tmp_dir = str(tmpdir.mkdir("images"))
    for i in range(len(image_files)):
        img = np.zeros((80, 128, 3), dtype=np.uint8)
        grad = np.linspace(0, 255, img.shape[1])
        img[16:32, :, 0] = grad
        im = PIL.Image.fromarray(img)
        image_path = os.path.join(tmp_dir, image_files[i])
        im.save(image_path)
        image_paths.append(image_path)
    return image_paths


def test_len_image_data_source(_image_files):
    image_source = image_data_source.ImageDataSource(
        _image_files, ".jpg", frame_shape.FrameShape(height=80, width=128, channels=3)
    )
    assert len(image_source) == 2


def test_image_data_source(tmpdir, _image_files):
    image_paths = _create_images(tmpdir, _image_files)
    image_source = image_data_source.ImageDataSource(
        image_paths, ".jpg", frame_shape.FrameShape(height=80, width=128, channels=3)
    )
    data_loader_ = data_loader.DataLoader(
        [image_source], augmentation_pipeline=[], batch_size=1
    )
    data_loader_.set_shard()
    batch = data_loader_()
    assert len(data_loader_) == 2
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.get_collection("iterator_init"))
        for _ in range(len(data_loader_)):
            example = sess.run(batch)
            image = example.instances[types.FEATURE_CAMERA].images
            scaled_pixels = np.argwhere(image > 1)
            assert image.shape == (1, 3, 80, 128)
            # To test that '_decode_function' was applied.
            assert not scaled_pixels


def test__serialization_and_deserialization(_image_files):
    """Test TAOObject serialization and deserialization on ImageDataSource."""
    source = image_data_source.ImageDataSource(
        image_paths=_image_files, extension=".jpg", image_shape=[10, 10, 3]
    )
    source_dict = source.serialize()
    deserialized_source = deserialize_tao_object(source_dict)
    assert source._image_paths == deserialized_source._image_paths
    assert source.extension == deserialized_source.extension
    assert source._image_shape == deserialized_source._image_shape
