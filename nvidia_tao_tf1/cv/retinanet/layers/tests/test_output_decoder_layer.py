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
"""test output decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.models import Model
import numpy as np

from nvidia_tao_tf1.cv.retinanet.layers.output_decoder_layer import DecodeDetections


def test_output_decoder_no_compression():
    x = Input(shape=(2, 15))
    y = DecodeDetections(top_k=2, nms_max_output_size=5, img_height=300, img_width=300)(x)
    model = Model(inputs=x, outputs=y)

    encoded_val = '''np.array(
      [[[  0.        ,   1.        ,   0.        ,  -2.46207404,
         -5.01084082, -21.38983255, -20.27411479,   0.25      ,
          0.5       ,   0.96124919,   0.96124919,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
        [  1.        ,   0.        ,   0.        ,  -1.07137391,
         -2.54451304,  -3.64782921,  -7.11356512,   0.25      ,
          0.5       ,   0.62225397,   1.24450793,   0.1       ,
          0.1       ,   0.2       ,   0.2       ]]]
        )'''
    encoded_val = eval(encoded_val)[:, :, :]
    expected = '''np.array(
        [[[  1.     ,   1.     , 127.67308, 148.65036, 130.63277,
         151.04959],
        [  1.     ,   0.     ,   0.     ,   0.     ,   0.     ,
           0.     ]]])'''
    expected = eval(expected)
    pred = model.predict(encoded_val)
    assert np.max(abs(pred - expected)) < 1e-5


def test_output_decoder_compression():
    x = Input(shape=(10, 15))
    y = DecodeDetections(top_k=5, nms_max_output_size=15, img_height=300, img_width=300)(x)
    model = Model(inputs=x, outputs=y)
    encoded_val = '''np.array(
       [[
       [  0.        ,   1.        ,   0.        ,   4.36584869,
          0.26784348,  -1.88672624,  -8.81819805,   0.05      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  0.        ,   1.        ,   0.        ,   3.56231825,
          0.26784348,  -1.88672624,  -8.81819805,   0.15      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  0.        ,   0.        ,   1.        ,   2.75878782,
          0.26784348,  -1.88672624,  -8.81819805,   0.25      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  1.        ,   0.        ,   0.        ,   1.95525739,
          0.26784348,  -1.88672624,  -8.81819805,   0.35      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  1.        ,   0.        ,   0.        ,   1.15172695,
          0.26784348,  -1.88672624,  -8.81819805,   0.45      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  1.        ,   0.        ,   0.        ,   0.34819652,
          0.26784348,  -1.88672624,  -8.81819805,   0.55      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  1.        ,   0.        ,   0.        ,  -0.45533391,
          0.26784348,  -1.88672624,  -8.81819805,   0.65      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  1.        ,   0.        ,   0.        ,  -1.25886435,
          0.26784348,  -1.88672624,  -8.81819805,   0.75      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  1.        ,   0.        ,   0.        ,  -2.06239478,
          0.26784348,  -1.88672624,  -8.81819805,   0.85      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ],
       [  1.        ,   0.        ,   0.        ,  -2.86592521,
          0.26784348,  -1.88672624,  -8.81819805,   0.95      ,
          0.5       ,   1.24450793,   0.62225397,   0.1       ,
          0.1       ,   0.2       ,   0.2       ]]])'''
    encoded_val = eval(encoded_val)[:, :, :]
    expected = '''np.array(
                  [[[  1., 1., 227.77, 166.17693, 473.4848, 172.46394],
                    [  2., 1., 204.19827, 166.17693, 408.7723, 172.46394],
                    [  1., 0., 0., 0., 0., 0.],
                    [  1., 0., 0., 0., 0., 0.],
                    [  1., 0., 0., 0., 0., 0.]]])'''
    expected = eval(expected)
    pred = model.predict(encoded_val)
    assert np.max(abs(pred - expected)) < 1e-5
