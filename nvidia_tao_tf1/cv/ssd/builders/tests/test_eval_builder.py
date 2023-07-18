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
"""test ssd eval builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Reshape
from keras.models import Model
import pytest
from nvidia_tao_tf1.cv.ssd.builders import eval_builder


@pytest.fixture
def test_model():
    x = Input(shape=(3, 40, 40))
    y = Reshape(target_shape=(300, 16))(x)
    model = Model(inputs=x, outputs=y)
    return model


def test_decoded_output(test_model):
    model = eval_builder.build(test_model)
    assert len(model.outputs) == 1
    assert model.outputs[0].shape[1] == 200
    assert model.outputs[0].shape[2] == 6


def test_decoded_output_with_encoded_prediction(test_model):
    model = eval_builder.build(test_model,
                               include_encoded_pred=True)
    assert len(model.outputs) == 2
    assert model.outputs[0].shape[1] == 300
    assert model.outputs[0].shape[2] == 16
