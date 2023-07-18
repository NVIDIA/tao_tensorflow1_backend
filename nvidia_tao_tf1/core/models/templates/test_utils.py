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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras import backend as K
from keras.layers import Activation
from keras.layers import Add
from keras.layers import Dense
from keras.layers import ELU
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import ReLU

import numpy as np
import nvidia_tao_tf1.core.models.templates.utils as utils
import pytest


cloning_tests = [
    # no inputs --> return with placeholders
    ([(32,)], [], 1),
    # one input --> return with tensor
    ([(32,)], [(10, 32)], 0),
    # two inputs --> return with tensors
    ([(32,), (32,)], [(10, 32), (10, 32)], 0),
    # model has two inputs, but inputs only has one. --> throw error
    ([(32,), (32,)], [(10, 32)], None),
    # model has one input, but inputs has two. --> throw error
    ([(32,)], [(10, 32), (10, 32)], None),
]


@pytest.mark.parametrize("copy_weights", [True, False])
@pytest.mark.parametrize(
    "model_inputs_shape, inputs_shape, expected_placeholders", cloning_tests
)
def test_clone_model(
    copy_weights, model_inputs_shape, inputs_shape, expected_placeholders
):
    """Test cloning a model."""
    # Add the first and last element of model_inputs. This ensures that whether there are
    # one or two inputs provided, all of them are used.
    model_inputs = [Input(shape=shape) for shape in model_inputs_shape]
    inputs = [
        K.random_uniform_variable(shape=shape, low=0, high=1) for shape in inputs_shape
    ]
    if inputs == []:
        inputs = None

    middle_layer = Add()([model_inputs[0], model_inputs[-1]])
    model = keras.models.Model(inputs=model_inputs, outputs=Dense(32)(middle_layer))

    if expected_placeholders is not None:
        new_model = utils.clone_model(model, inputs=inputs, copy_weights=copy_weights)
        num_placeholders = len(
            [
                l
                for l in new_model.layers
                if (("is_placeholder" in dir(l)) and (l.is_placeholder is True))
            ]
        )
        assert num_placeholders == expected_placeholders
        if copy_weights:
            for old_layer, new_layer in zip(model.layers, new_model.layers):
                old_weights = old_layer.get_weights()
                new_weights = new_layer.get_weights()
                for old_w, new_w in zip(old_weights, new_weights):
                    np.testing.assert_array_equal(old_w, new_w)
    else:
        with pytest.raises(ValueError):
            new_model = utils.clone_model(model, inputs=inputs)


activation_test_cases = [
    ("relu", {}, Activation),
    ("relu-n", {"max_value": 6.0}, ReLU),
    ("lrelu", {"alpha": 0.2}, LeakyReLU),
    ("elu", {"alpha": 1.0}, ELU),
]


@pytest.mark.parametrize(
    "activation_type, activation_kwargs, expected_object_type", activation_test_cases
)
def test_add_activation(activation_type, activation_kwargs, expected_object_type):
    """Test that add_activation returns correct object instances."""
    activation_layer = utils.add_activation(activation_type, **activation_kwargs)
    assert isinstance(activation_layer, expected_object_type)


test_cases = [(27, "ab"), (201, "gt"), (0, "a")]


@pytest.mark.parametrize("key, expected_id", test_cases)
def test_SUBBLOCK_IDS(key, expected_id):
    """Test SUBBLOCK_IDS to return expected ID string."""
    subblock_ids = utils.SUBBLOCK_IDS()
    assert subblock_ids[key] == expected_id


def test_update_regularizers():
    """Test that update_regularizers works as advertised."""
    shape = (32,)
    inputs = Input(shape=shape)
    outputs = Dense(
        units=32,
        kernel_regularizer=keras.regularizers.l2(1.0),
        bias_regularizer=keras.regularizers.l1(2.0),
    )(inputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    assert len(model.losses) == 2

    updated_model = utils.update_regularizers(
        model, kernel_regularizer=None, bias_regularizer=None
    )

    assert updated_model.losses == []
