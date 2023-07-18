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

from nvidia_tao_tf1.cv.yolo_v3.layers.nms_layer import NMSLayer


def test_NMS_truncate():
    x = Input(shape=(5, 7))
    y = NMSLayer(output_size=3)(x)
    model = Model(inputs=x, outputs=y)

    # See sample details in test_input_encoder
    encoded_val = '''np.array([[[ 1., 2., 3., 4., 1., 0., 0.],
                                [ 35., 36., 37., 38., 0.5, 0., 0.],
                                [ 15., 16., 17., 18., 0.7, 0., 0.],
                                [ 25., 26., 27., 28., 0.6, 0., 0.],
                                [ 5., 6., 7., 8., 0.8, 0., 0.]]])'''
    encoded_val = eval(encoded_val)

    # normalized
    encoded_val[..., :4] = encoded_val[..., :4] / 50.0

    expected = '''np.array( [[[ 0. ,  1. ,  1. ,  2. ,  3. ,  4. ],
                              [ 0. ,  0.8,  5. ,  6. ,  7. ,  8. ],
                              [ 0. ,  0.7, 15. , 16. , 17. , 18. ]]] )'''

    expected = eval(expected)

    # normalize
    expected[..., -4:] = expected[..., -4:] / 50.0
    pred = model.predict(encoded_val)
    assert np.max(np.abs(pred - expected)) < 1e-5


def test_NMS_padding():
    x = Input(shape=(5, 7))
    y = NMSLayer(output_size=10)(x)
    model = Model(inputs=x, outputs=y)

    # See sample details in test_input_encoder
    encoded_val = '''np.array([[[ 1., 2., 3., 4., 1., 0., 0.],
                                [ 35., 36., 37., 38., 0.5, 0., 0.],
                                [ 15., 16., 17., 18., 0.7, 0., 0.],
                                [ 25., 26., 27., 28., 0.6, 0., 0.],
                                [ 5., 6., 7., 8., 0.8, 0., 0.]]])'''
    encoded_val = eval(encoded_val)
    # normalized
    encoded_val[..., :4] = encoded_val[..., :4] / 50.0
    expected = '''np.array( [[[ 0. ,  1. ,  1. ,  2. ,  3. ,  4. ],
                              [ 0. ,  0.8,  5. ,  6. ,  7. ,  8. ],
                              [ 0. ,  0.7, 15. , 16. , 17. , 18. ],
                              [ 0. ,  0.6, 25. , 26. , 27. , 28. ],
                              [ 0. ,  0.5, 35. , 36. , 37. , 38. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ]]] )'''

    expected = eval(expected)
    # normalize
    expected[..., -4:] = expected[..., -4:] / 50.0
    pred = model.predict(encoded_val)
    assert np.max(np.abs(pred - expected)) < 1e-5


def test_NMS_nms():
    x = Input(shape=(5, 7))
    y = NMSLayer(output_size=10)(x)
    model = Model(inputs=x, outputs=y)

    # See sample details in test_input_encoder
    encoded_val = '''np.array([[[ 1., 2., 3., 4., 1., 0., 0.],
                                [ 35., 36., 37., 38., 0.5, 0., 0.],
                                [ 5.3, 6.3, 7., 8., 0.7, 0., 0.],
                                [ 25., 26., 27., 28., 0.6, 0., 0.],
                                [ 5., 6., 7., 8., 0.8, 0., 0.]]])'''
    encoded_val = eval(encoded_val)
    # normalized
    encoded_val[..., :4] = encoded_val[..., :4] / 50.0
    expected = '''np.array( [[[ 0. ,  1. ,  1. ,  2. ,  3. ,  4. ],
                              [ 0. ,  0.8,  5. ,  6. ,  7. ,  8. ],
                              [ 0. ,  0.6, 25. , 26. , 27. , 28. ],
                              [ 0. ,  0.5, 35. , 36. , 37. , 38. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                              [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ]]] )'''

    expected = eval(expected)
    # normalize
    expected[..., -4:] = expected[..., -4:] / 50.0
    pred = model.predict(encoded_val)
    assert np.max(np.abs(pred - expected)) < 1e-5
