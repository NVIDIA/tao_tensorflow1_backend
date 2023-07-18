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
"""test YOLO v4 keras sequence dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
from PIL import Image
import pytest

from nvidia_tao_tf1.cv.yolo_v3.proto.dataset_config_pb2 import (
    YOLOv3DatasetConfig,
    YOLOv3DataSource,
)
from nvidia_tao_tf1.cv.yolo_v4.dataio.data_sequence import YOLOv4DataSequence
from nvidia_tao_tf1.cv.yolo_v4.proto.augmentation_config_pb2 import AugmentationConfig


@pytest.fixture
def _test_experiment_spec():
    img = np.random.randint(low=0, high=255, size=(375, 500, 3), dtype=np.uint8)
    gt = ["bicycle 0 0 0 1 45 493 372 0 0 0 0 0 0 0",
          "bicycle 0 0 0 54 24 500 326 0 0 0 0 0 0 0",
          "bicycle 0 0 0 54 326 500 326 0 0 0 0 0 0 0"]
    if not os.path.exists("tmp_labels/"):
        os.mkdir("tmp_labels/")
    with open("tmp_labels/0.txt", "w") as f:
        for line in gt:
            f.write(line + "\n")
    if not os.path.exists("tmp_imgs/"):
        os.mkdir("tmp_imgs/")
    tmp_im = Image.fromarray(img)
    tmp_im.save("tmp_imgs/0.jpg")
    yield
    shutil.rmtree("tmp_labels")
    shutil.rmtree("tmp_imgs")


@pytest.fixture
def dataset_config():
    """dataset config."""
    source = YOLOv3DataSource()
    source.label_directory_path = "tmp_labels/"
    source.image_directory_path = "tmp_imgs/"
    dataset = YOLOv3DatasetConfig()
    dataset.data_sources.extend([source])
    dataset.target_class_mapping.update({"bicycle": "bicycle"})
    dataset.validation_data_sources.extend([source])
    dataset.image_extension = "jpg"
    return dataset


@pytest.fixture
def augmentation_config():
    """augmentation config."""
    aug_config = AugmentationConfig()
    aug_config.output_width = 320
    aug_config.output_height = 320
    aug_config.output_channel = 3
    return aug_config


def test_data_sequence(dataset_config, augmentation_config, _test_experiment_spec):
    # init dataloader:
    train_dataset = YOLOv4DataSequence(
        dataset_config=dataset_config,
        augmentation_config=augmentation_config,
        batch_size=1,
        is_training=True,
        encode_fn=None
    )
    val_dataset = YOLOv4DataSequence(
        dataset_config=dataset_config,
        augmentation_config=augmentation_config,
        batch_size=1,
        is_training=False,
        encode_fn=None
    )
    # test load gt label for train
    train_imgs, train_labels = train_dataset[0]
    val_imgs, val_labels = val_dataset[0]
    assert train_labels[0].shape[-1] == 6
    assert val_labels[0].shape[-1] == 6
    # test filter wrong gt label
    assert val_labels[0].shape[0] == 2
    # test preprocess
    assert train_imgs[0].shape == (3, 320, 320)
    assert val_imgs[0].shape == (3, 320, 320)
