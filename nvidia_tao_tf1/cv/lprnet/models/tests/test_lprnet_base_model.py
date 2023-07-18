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
"""test lprnet base model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf
from nvidia_tao_tf1.cv.lprnet.models.lprnet_base_model import LPRNetbaseline


correct_nlayers_list = [10, 18]


@pytest.mark.parametrize("nlayers",
                         correct_nlayers_list)
def test_lprnet_base_model(nlayers):
    input_layer = tf.keras.Input(shape=(3, 48, 96))
    output_layer = LPRNetbaseline(nlayers=nlayers)(input_layer, trainable=False)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.predict(np.random.randn(1, 3, 48, 96))


def test_wrong_lprnet_base_model():
    with pytest.raises(AssertionError):
        LPRNetbaseline(nlayers=99)
