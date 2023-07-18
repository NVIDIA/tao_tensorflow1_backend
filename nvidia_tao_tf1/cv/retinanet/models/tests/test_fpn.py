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
"""test retinanet FPN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
import pytest
from nvidia_tao_tf1.cv.retinanet.models.fpn import FPN


@pytest.mark.slow
@pytest.mark.parametrize("bb, nlayers",
                         [('mobilenet_v2', None),
                          ('resnet', 18),
                          ('resnet', 10),
                          ('resnet', 34),
                          ('resnet', 50),
                          ('resnet', 101),
                          ('vgg', 16),
                          ('vgg', 19),
                          ('mobilenet_v1', 10),
                          ('googlenet', None)])
def test_fpn(bb, nlayers):
    input_tensor = Input(shape=(3, 512, 512))
    fpn = FPN(input_tensor, bb,
              nlayers=nlayers,
              use_batch_norm=True,
              use_pooling=False,
              freeze_bn=False,
              use_bias=False,
              data_format='channels_first',
              dropout=None,
              all_projections=True)
    feature_map_list = fpn.generate(
        feature_size=256,
        kernel_regularizer=None)
    assert len(feature_map_list) == 5
    p1, p2, p3, p4, p5 = feature_map_list

    shape1 = p1.get_shape().as_list()
    shape2 = p2.get_shape().as_list()
    shape3 = p3.get_shape().as_list()
    shape4 = p4.get_shape().as_list()
    shape5 = p5.get_shape().as_list()

    assert len(shape1) == 4
    assert shape1[1] == shape2[1] == shape3[1] == shape4[1] == shape5[1]
    assert shape1[-1] == 2 * shape2[-1]
    assert shape2[-1] == 2 * shape3[-1]
    assert shape3[-1] == 2 * shape4[-1]
    assert shape4[-1] == 2 * shape5[-1]
