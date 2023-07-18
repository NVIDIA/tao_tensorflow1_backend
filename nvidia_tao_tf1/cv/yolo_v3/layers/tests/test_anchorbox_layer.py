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
"""test NMS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.models import Model
import numpy as np

from nvidia_tao_tf1.cv.yolo_v3.layers.yolo_anchor_box_layer import YOLOAnchorBox


def test_anchorbox_multibatch():
    x = Input(shape=(3, 2, 2))
    img_size = 416.0
    anchor_size = np.array([(116, 90), (156, 198), (373, 326)]) / img_size
    y = YOLOAnchorBox(anchor_size=anchor_size)(x)
    model = Model(inputs=x, outputs=y)

    # See sample details in test_input_encoder
    fmap = np.random.normal(size=(2, 3, 2, 2))
    expected = '''np.array(   [[[  0.,   0.,  90., 116.],
                                 [  0.,   0., 198., 156.],
                                 [  0.,   0., 326., 373.],
                                 [  0., 208.,  90., 116.],
                                 [  0., 208., 198., 156.],
                                 [  0., 208., 326., 373.],
                                 [208.,   0.,  90., 116.],
                                 [208.,   0., 198., 156.],
                                 [208.,   0., 326., 373.],
                                 [208., 208.,  90., 116.],
                                 [208., 208., 198., 156.],
                                 [208., 208., 326., 373.]],


                               [ [  0.,   0.,  90., 116.],
                                 [  0.,   0., 198., 156.],
                                 [  0.,   0., 326., 373.],
                                 [  0., 208.,  90., 116.],
                                 [  0., 208., 198., 156.],
                                 [  0., 208., 326., 373.],
                                 [208.,   0.,  90., 116.],
                                 [208.,   0., 198., 156.],
                                 [208.,   0., 326., 373.],
                                 [208., 208.,  90., 116.],
                                 [208., 208., 198., 156.],
                                 [208., 208., 326., 373.]]])'''

    expected = eval(expected)
    expected = np.concatenate([expected, np.zeros((2, 12, 2)) + 208.0], axis=-1)
    pred = model.predict(fmap) * img_size
    assert np.max(np.abs(pred - expected)) < 1e-5

    x = Input(shape=(3, None, None))
    y = YOLOAnchorBox(anchor_size=anchor_size)(x)
    model = Model(inputs=x, outputs=y)
    pred = model.predict(fmap) * img_size
    assert np.max(np.abs(pred - expected)) < 1e-5
