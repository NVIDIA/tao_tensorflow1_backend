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

from nvidia_tao_tf1.cv.retinanet.utils.spec_loader import load_experiment_spec


def test_spec_loader():
    experiment_spec = load_experiment_spec(merge_from_default=True)
    # params
    img_channels = experiment_spec.augmentation_config.output_channel
    img_height = experiment_spec.augmentation_config.output_height
    img_width = experiment_spec.augmentation_config.output_width
    freeze_blocks = experiment_spec.retinanet_config.freeze_blocks
    freeze_bn = experiment_spec.retinanet_config.freeze_bn
    nlayers = experiment_spec.retinanet_config.nlayers
    arch = experiment_spec.retinanet_config.arch

    assert arch == 'resnet'
    assert nlayers == 18
    assert freeze_bn
    assert freeze_blocks == [0.0, 1.0]
    assert img_width == img_height == 512
    assert img_channels == 3
