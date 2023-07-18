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
"""test yolo arch builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, LeakyReLU
from nvidia_tao_tf1.cv.yolo_v3.architecture.yolo_arch import YOLO


def test_arch():
    it = Input(shape=(3, 64, 32), name="Input")
    model = YOLO(it,
                 'resnet', 18,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 freeze_blocks=[0],
                 freeze_bn=None,
                 arch_conv_blocks=2,
                 qat=True)
    assert model.get_layer('conv1').trainable is False
    assert model.get_layer('encoded_detections').output_shape[-2:] == (126, 91)

    model = YOLO(it,
                 'resnet', 18,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 freeze_blocks=[0],
                 freeze_bn=None,
                 arch_conv_blocks=2,
                 qat=True)
    assert model.get_layer('conv1').trainable is False
    assert model.get_layer('encoded_detections').output_shape[-2:] == (126, 91)
    for layer in model.layers:
        assert type(layer) != LeakyReLU

    model = YOLO(it,
                 'darknet', 19,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 freeze_blocks=None,
                 freeze_bn=None,
                 arch_conv_blocks=2,
                 qat=False,
                 force_relu=True)

    assert model.get_layer('conv1').trainable is True
    for layer in model.layers:
        assert type(layer) != LeakyReLU
