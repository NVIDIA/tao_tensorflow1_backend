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

"""Test COCO dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest

from nvidia_tao_tf1.cv.retinanet.dataio.coco_loader import RetinaCocoDataSequence
from nvidia_tao_tf1.cv.retinanet.utils.spec_loader import load_experiment_spec


@pytest.mark.skipif(
    os.getenv("RUN_ON_CI", "0") == "1",
    reason="Cannot be run on CI"
)
def test_coco_dataloader():
    bs = 2
    training = False
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    spec_path = os.path.join(file_path, '../../retinanet/experiment_specs/default_spec.txt')
    experiment_spec = load_experiment_spec(spec_path, merge_from_default=False)
    dataset_config = experiment_spec.dataset_config
    aug_config = experiment_spec.augmentation_config

    d = RetinaCocoDataSequence(dataset_config=dataset_config,
                               augmentation_config=aug_config,
                               batch_size=bs, is_training=training,
                               encode_fn=None)

    img, _ = d.__getitem__(0)
    assert img.shape == (2, 3, 512, 512)
    assert len(d.classes) == 80
    assert d.n_samples == 5000
