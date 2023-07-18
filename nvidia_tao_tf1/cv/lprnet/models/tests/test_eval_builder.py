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
"""test lprnet eval builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import tensorflow as tf
from nvidia_tao_tf1.cv.lprnet.models import eval_builder


@pytest.fixture
def test_model():
    x = tf.keras.Input(shape=(24, 36))
    model = tf.keras.Model(inputs=x, outputs=x)
    return model


def test_decoded_output(test_model):
    eval_model = eval_builder.build(test_model)

    assert len(eval_model.outputs) == 2
    assert eval_model.outputs[0].shape[-1] == 24
    assert eval_model.outputs[1].shape[-1] == 24
