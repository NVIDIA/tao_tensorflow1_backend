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

"""Test the Keras backend changes related to mixed-precision implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine import Layer
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Input
from keras.models import Model
from keras.regularizers import l1
from keras.regularizers import l2
import numpy as np
from numpy.testing import assert_allclose
import pytest
import tensorflow as tf


WEIGHT_SHAPE = (3, 3, 3)
DATA_SHAPE = (24, 3, 16, 32)
BATCH_NORM_VARIABLE_NAMES = ("beta", "gamma", "moving_mean", "moving_variance")


def _get_tf_weight_by_name(name):
    """Get Tensorflow variable based on a partial variable name."""
    candidates = [
        var
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        if name in var.name
    ]
    assert len(candidates) == 1, "Use unique variable names."
    return candidates[0]


@pytest.mark.usefixtures("clear_session")
def test_layer_add_weight_fp32():
    """Test that keras.engine.Layer.add_weight is correctly patched for fp32 mode."""
    K.set_floatx("float32")
    # Create a layer and a weight.
    fp32_layer = Layer()
    keras_weight = fp32_layer.add_weight(
        name="fp32_mode_weight", shape=WEIGHT_SHAPE, initializer="ones"
    )
    # For fp32, returned weight has to match the backend variable.
    backend_variable = _get_tf_weight_by_name("fp32_mode_weight")
    assert backend_variable == keras_weight, "add_weight returned an unknown tensor."
    # Get the values and verify data type and shape.
    sess = K.get_session()
    np_keras_weight = sess.run(keras_weight)
    assert np_keras_weight.dtype == np.float32
    assert_allclose(np_keras_weight, np.ones(WEIGHT_SHAPE, dtype=np.float32))


@pytest.mark.usefixtures("clear_session")
def test_layer_add_weight_fp16():
    """Test that keras.engine.Layer.add_weight is correctly patched for fp16 mode."""
    K.set_floatx("float16")
    # Create a layer and a weight.
    fp16_layer = Layer()
    keras_weight = fp16_layer.add_weight(
        name="fp16_mode_weight", shape=WEIGHT_SHAPE, initializer="ones"
    )
    # For fp16, returned weight shall not match the backend variable.
    backend_variable = _get_tf_weight_by_name("fp16_mode_weight")
    assert backend_variable != keras_weight, "add_weight returned a raw variable."
    # Get the values and verify data type and shape.
    sess = K.get_session()
    np_keras_weight = sess.run(keras_weight)
    np_backend_variable = sess.run(backend_variable)
    assert np_keras_weight.dtype == np.float16
    assert_allclose(np_keras_weight, np.ones(WEIGHT_SHAPE, dtype=np.float16))
    # In mixed-precision training, backend variables are created in float32.
    assert np_backend_variable.dtype == np.float32
    assert_allclose(np_backend_variable, np.ones(WEIGHT_SHAPE, dtype=np.float32))


@pytest.mark.usefixtures("clear_session")
@pytest.mark.parametrize("data_type", ["float16", "float32"])
def test_batch_normalization(data_type):
    """Test that patched build and call are in use in BatchNormalization layer."""
    # Set backend precision.
    K.set_floatx(data_type)

    # Create dummy data.
    np_ones = np.ones(DATA_SHAPE, dtype=data_type)

    # Build the training graph.
    K.set_learning_phase(1)
    # Placeholder for input data.
    train_input = tf.placeholder(dtype=data_type, shape=DATA_SHAPE, name="train_data")
    input_layer = Input(tensor=train_input, name="train_input_layer")
    # Add one batch normalization layer.
    bn_out = BatchNormalization(axis=1, name="batchnorm_layer")(input_layer)
    # Get the model and its output.
    model = Model(inputs=input_layer, outputs=bn_out, name="dummy_model")
    train_output = model(train_input)

    # Build inference graph.
    K.set_learning_phase(0)
    infer_input = tf.placeholder(dtype=data_type, shape=DATA_SHAPE, name="infer_data")
    infer_output = model(infer_input)

    # Verify that all backend variables were created as float32_ref.
    for variable_name in BATCH_NORM_VARIABLE_NAMES:
        # Get backend variable by name
        var = _get_tf_weight_by_name(variable_name)
        # tf.float32_ref object does not exist, serialize and text compare.
        assert "float32_ref" in str(
            var.dtype
        ), "BatchNormalization created wrong variable dtype."

    # Verify that training and inference outputs follow Keras floatx setting.
    assert bn_out.dtype == data_type, "Wrong training output data type."
    assert infer_output.dtype == data_type, "Wrong inference output data type."

    # Infer with BN initial moving mean and variance (shall not modify the input).
    sess = K.get_session()
    net_out = sess.run(infer_output, feed_dict={infer_input: np_ones})
    assert_allclose(net_out, np_ones, atol=1e-3)

    # Build cost and optimizer.
    K.set_learning_phase(1)
    cost = tf.reduce_sum(train_output)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0)
    train_op = optimizer.minimize(loss=cost)
    # Run one step of training.
    _, net_out = sess.run([train_op, train_output], feed_dict={train_input: np_ones})
    # Verify that BN removed mean -> all outputs should be zeros.
    assert_allclose(net_out, np.zeros_like(net_out, dtype=data_type), atol=1e-4)


@pytest.mark.usefixtures("clear_session")
@pytest.mark.parametrize("data_type", ["float16", "float32"])
def test_regularizers(data_type):
    # Set backend precision.
    K.set_floatx(data_type)

    # Use very small weights (will round to zero in non-patched fp16 implementation).
    l1_regularizer = l1(1e-9)
    l2_regularizer = l2(1e-9)

    # Create a convolutional model.
    K.set_learning_phase(1)
    train_input = tf.placeholder(dtype=data_type, shape=DATA_SHAPE, name="train_data")
    input_layer = Input(tensor=train_input, name="train_input_layer")
    conv1_out = Conv2D(
        1,
        (3, 3),
        data_format="channels_first",
        kernel_regularizer=l1_regularizer,
        name="convolutional_layer_1",
    )(input_layer)
    conv2_out = Conv2D(
        1,
        (3, 3),
        data_format="channels_first",
        kernel_regularizer=l2_regularizer,
        name="convolutional_layer_2",
    )(conv1_out)

    # Get the model and regularization losses.
    model = Model(inputs=input_layer, outputs=conv2_out, name="dummy_model")
    reg_losses = model.losses

    # Get the regularization losses with dummy input.
    np_ones = np.ones(DATA_SHAPE, dtype=data_type)
    sess = K.get_session()
    loss_values = sess.run(reg_losses, feed_dict={train_input: np_ones})

    # Verify regularization loss data types.
    assert [loss.dtype for loss in loss_values] == [
        np.float32,
        np.float32,
    ], "Regularization loss dtype shall match backend variable dtype (always float32)."

    # Verify that regularization loss is not zero.
    assert np.all(np.array(loss_values) > 1e-10), "Regularization loss is zero."
