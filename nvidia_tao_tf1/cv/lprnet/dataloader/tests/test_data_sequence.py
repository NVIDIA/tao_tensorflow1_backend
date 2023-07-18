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
"""test lprnet keras sequence dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
from PIL import Image
import pytest

from nvidia_tao_tf1.cv.lprnet.dataloader.data_sequence import LPRNetDataGenerator
from nvidia_tao_tf1.cv.lprnet.utils.spec_loader import load_experiment_spec


@pytest.fixture
def _test_experiment_spec():
    img = np.random.randint(low=0, high=255, size=(52, 105, 3), dtype=np.uint8)
    gt = "3H0X429"
    experiment_spec = load_experiment_spec(merge_from_default=True)
    if not os.path.exists("tmp_labels/"):
        os.mkdir("tmp_labels/")
    with open("tmp_labels/0.txt", "w") as f:
        f.write(gt)
    if not os.path.exists("tmp_imgs/"):
        os.mkdir("tmp_imgs/")
    tmp_im = Image.fromarray(img)
    tmp_im.save("tmp_imgs/0.jpg")
    with open("tmp_ch_list_data.txt", "w") as f:
        for ch in gt:
            f.write(ch + "\n")
    experiment_spec.dataset_config.data_sources[0].label_directory_path = "tmp_labels/"
    experiment_spec.dataset_config.data_sources[0].image_directory_path = "tmp_imgs/"
    experiment_spec.dataset_config.validation_data_sources[0].label_directory_path = "tmp_labels/"
    experiment_spec.dataset_config.validation_data_sources[0].image_directory_path = "tmp_imgs/"
    experiment_spec.dataset_config.characters_list_file = "tmp_ch_list_data.txt"
    experiment_spec.training_config.batch_size_per_gpu = 1
    yield experiment_spec
    shutil.rmtree("tmp_labels")
    shutil.rmtree("tmp_imgs")
    os.remove("tmp_ch_list_data.txt")


def test_data_sequence(_test_experiment_spec):
    train_data = LPRNetDataGenerator(experiment_spec=_test_experiment_spec,
                                     is_training=True,
                                     shuffle=True,
                                     time_step=24)

    val_data = LPRNetDataGenerator(experiment_spec=_test_experiment_spec,
                                   is_training=False,
                                   shuffle=True,
                                   time_step=24)

    assert len(train_data) == 1

    train_im, train_label = train_data[0]
    val_im, val_label = val_data[0]

    assert train_label[0].shape[-1] == 10
    assert len(val_label[0]) == 7
    assert train_im[0].shape == (3, 48, 96)
    assert val_im[0].shape == (3, 48, 96)
