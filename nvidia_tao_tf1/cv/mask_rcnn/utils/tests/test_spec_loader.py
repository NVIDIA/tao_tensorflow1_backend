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

"""Test spec loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from nvidia_tao_tf1.cv.mask_rcnn.utils.spec_loader import load_experiment_spec


def test_spec_loader():
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path = os.path.join(file_path, '../experiment_specs/default.txt')
    experiment_spec = load_experiment_spec(default_spec_path, merge_from_default=False)
    # params
    # use_amp = experiment_spec.use_amp
    warmup_learning_rate = experiment_spec.warmup_learning_rate
    init_learning_rate = experiment_spec.init_learning_rate
    train_bs = experiment_spec.train_batch_size
    eval_bs = experiment_spec.eval_batch_size
    expected = [0.0001, 0.02, 2, 4]
    assert np.allclose([warmup_learning_rate, init_learning_rate, train_bs, eval_bs], expected)

    n_classes = experiment_spec.data_config.num_classes
    h, w = eval(experiment_spec.data_config.image_size)
    expected = [91, 832, 1344]
    assert np.allclose([n_classes, h, w], expected)

    nlayers = experiment_spec.maskrcnn_config.nlayers
    gt_mask_size = experiment_spec.maskrcnn_config.gt_mask_size
    arch = experiment_spec.maskrcnn_config.arch
    expected = [50, 112]
    assert np.allclose([nlayers, gt_mask_size], expected)
    assert arch == 'resnet'
