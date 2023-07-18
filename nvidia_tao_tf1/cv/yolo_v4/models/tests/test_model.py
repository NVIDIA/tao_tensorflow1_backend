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
"""test yolo models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.models import Model
import numpy as np
import pytest
from nvidia_tao_tf1.cv.yolo_v4.models.base_model import get_base_model


def do_model_pred(input_shape, arch, nlayers):
    x = Input(shape=input_shape)
    fmaps, map_size = get_base_model(x, arch, nlayers)
    model = Model(inputs=x, outputs=fmaps)

    x_in = np.random.normal(size=(2, ) + input_shape)
    pred = model.predict(x_in)
    stride = 32
    if arch == "cspdarknet_tiny":
        assert len(map_size) == 1
        assert len(pred) == 2
    else:
        assert len(map_size) == 2
        assert len(pred) == 3

    if arch == 'vgg':
        stride = 16

    # assert pred 0
    assert pred[0].shape[0] == 2
    assert pred[0].shape[2] == input_shape[-1] / stride
    assert pred[0].shape[3] == input_shape[-1] / stride

    # assert pred 1
    assert pred[1].shape[0] == 2
    assert pred[1].shape[2] == input_shape[-1] / (stride / 2)
    assert pred[1].shape[3] == input_shape[-1] / (stride / 2)

    # assert pred 2
    # YOLO v4 Tiny only has two heads(stride 32 and 16, no 8)
    if arch != "cspdarknet_tiny":
        assert pred[2].shape[0] == 2
        assert pred[2].shape[2] == input_shape[-1] / (stride / 4)
        assert pred[2].shape[3] == input_shape[-1] / (stride / 4)


def test_all_base_models():
    # let's give a large coef on loss
    for arch in ['resnet', 'vgg', 'squeezenet', 'darknet', 'mobilenet_v1', 'mobilenet_v2',
                 'googlenet', 'cspdarknet', 'efficientnet_b0', 'cspdarknet_tiny',
                 'cspdarknet_tiny_3l', 'wrong_net']:
        for nlayers in [10, 16, 18, 19, 34, 50, 53, 101]:
            if arch in ['squeezenet', 'googlenet', 'mobilenet_v1',
                        'efficientnet_b0', 'cspdarknet_tiny',
                        'cspdarknet_tiny_3l'] and nlayers > 10:
                # Use mobilenet_v2 to test nlayers invariant
                continue
            resnet_flag = (arch == 'resnet' and nlayers not in [10, 18, 34, 50, 101])
            vgg_flag = (arch == 'vgg' and nlayers not in [16, 19])
            darknet_flag = (arch in ['darknet', 'cspdarknet'] and nlayers not in [19, 53])

            if resnet_flag or vgg_flag or darknet_flag or arch == 'wrong_net':
                with pytest.raises((ValueError, NotImplementedError)):
                    do_model_pred((3, 64, 64), arch, nlayers)
            else:
                do_model_pred((3, 64, 64), arch, nlayers)
