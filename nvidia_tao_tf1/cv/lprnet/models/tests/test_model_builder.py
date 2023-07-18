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
"""test lprnet model builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pytest
import nvidia_tao_tf1.cv.lprnet.models.model_builder as model_builder
from nvidia_tao_tf1.cv.lprnet.utils.spec_loader import load_experiment_spec

backbone_configs = [
                    ('baseline', 10, 24),
                    ('baseline', 18, 24),
                    ('resnet', 34, 6),
                    ('resnet', 50, 6),
                    ('vgg', 16, 6),
                   ]


@pytest.fixture
def experiment_spec():
    experiment_spec = load_experiment_spec(merge_from_default=True)
    label = "abcdefg"
    with open("tmp_ch_list.txt", "w") as f:
        for ch in label:
            f.write(ch + "\n")
    experiment_spec.dataset_config.characters_list_file = "tmp_ch_list.txt"
    yield experiment_spec
    os.remove("tmp_ch_list.txt")


@pytest.mark.parametrize("model_arch, nlayers, expected_time_step",
                         backbone_configs)
def test_model_builder(model_arch, nlayers, expected_time_step, experiment_spec):
    experiment_spec.lpr_config.arch = model_arch
    experiment_spec.lpr_config.nlayers = nlayers

    _, model, time_step = model_builder.build(experiment_spec)

    assert time_step == expected_time_step

    model.predict(np.random.randn(1, 3, 48, 96))
