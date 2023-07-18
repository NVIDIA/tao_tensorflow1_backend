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
"""test retinanet arch builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
from nvidia_tao_tf1.cv.retinanet.architecture.retinanet import retinanet


@pytest.mark.parametrize("arch, nlayers, n_anchor_levels, qat, feature_size",
                         [('vgg', 16, 1, False, 256),
                          ('resnet', 18, 1, False, 128),
                          ('efficientnet_b0', None, 3, False, 256),
                          ('mobilenet_v1', None, 3, True, 256),
                          ('mobilenet_v2', None, 2, False, 16),
                          ('squeezenet', None, 2, True, 32),
                          ('darknet', 53, 1, False, 64)])
def test_arch(arch, nlayers, n_anchor_levels, qat, feature_size):
    model = retinanet(
        (3, 512, 512),
        20,
        arch=arch,
        nlayers=nlayers,
        kernel_regularizer=None,
        freeze_blocks=[0],
        freeze_bn=None,
        min_scale=0.1,
        max_scale=0.8,
        scales=None,
        aspect_ratios_global=[1, 0.5, 2],
        aspect_ratios_per_layer=None,
        two_boxes_for_ar1=False,
        steps=None,
        offsets=None,
        clip_boxes=False,
        variances=[0.1, 0.1, 0.2, 0.2],
        input_tensor=None,
        n_anchor_levels=n_anchor_levels,
        qat=qat,
        feature_size=feature_size)

    assert model.get_layer('retinanet_predictions').output_shape[-2:] == \
        (16368 * n_anchor_levels, 32)
