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
"""Tests for the Weighted Momementum Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.learning_rate_schedules import ConstantLearningRateSchedule
from nvidia_tao_tf1.core.training import enable_deterministic_training
from nvidia_tao_tf1.cv.bpnet.optimizers.weighted_momentum_optimizer import \
    WeightedMomentumOptimizer


def _set_seeds(seed):
    """Set seeds for reproducibility.

    Args:
        seed (int): random seed value.
    """
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


@pytest.fixture(scope="module", autouse=True)
def determinism():
    enable_deterministic_training()
    _set_seeds(42)


class SimpleConvModel(object):
    """Simple 2-layer convolutional model to test with."""
    def __init__(self, initial_value=0.125):
        """__init__ method.

        Args:
            intial_value (float): weights initializer value.
        """
        self.initial_value = initial_value
        self._built = False
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    @property
    def trainable_vars(self):
        """Return trainable variables."""
        return [self.W1, self.b1, self.W2, self.b2]

    def _build(self, x):
        """Build weights.

        Args:
            x (tensor): input tensor to the model.
        """
        if self._built:
            return

        # Build W1 and b1.
        in_channels = x.shape[1]  # NCHW
        out_channels = 7
        W1_initial_value = np.ones([3, 3, in_channels, out_channels],
                                   dtype=np.float32)
        W1_initial_value *= self.initial_value

        self.W1 = tf.Variable(initial_value=W1_initial_value, trainable=True)
        self.b1 = tf.Variable(initial_value=np.zeros([out_channels],
                                                     dtype=np.float32),
                              trainable=True)

        # Build W2 and b2.
        in_channels = out_channels
        out_channels = 13
        W2_initial_value = np.ones([5, 5, in_channels, out_channels],
                                   dtype=np.float32)
        W2_initial_value *= self.initial_value

        self.W2 = tf.Variable(initial_value=W2_initial_value, trainable=True)
        self.b2 = tf.Variable(initial_value=np.zeros([out_channels],
                                                     dtype=np.float32),
                              trainable=True)

    def __call__(self, x):
        """Call method.

        Args:
            x (tensor): input tensor to the model.

        Returns:
            y (tensor): output tensor.
        """
        self._build(x)
        # h = W1*x + b1
        h = tf.nn.conv2d(input=x,
                         filters=self.W1,
                         strides=1,
                         padding="VALID",
                         data_format="NCHW")
        h = tf.nn.bias_add(value=h, bias=self.b1, data_format="NCHW")
        # y = W2*h + b2
        y = tf.nn.conv2d(input=h,
                         filters=self.W2,
                         strides=1,
                         padding="VALID",
                         data_format="NCHW")
        y = tf.nn.bias_add(value=y, bias=self.b2, data_format="NCHW")

        return y


class TestScenarios(object):
    """Tests with different weighting cases."""

    NUM_STEPS = 3
    LEARNING_RATE = 1e-5

    @staticmethod
    def get_loss(model):
        """Loss tensor to optimize.

        Args:
            model (SimpleConvModel): model to get predictions to compute loss.

        Returns:
            loss (tensor): compputed loss based on given gt.
        """
        x = tf.ones([3, 5, 16, 32])  # NCHW.
        y = model(x)

        y_true = tf.ones_like(y)

        loss = tf.reduce_sum(
            input_tensor=tf.compat.v1.squared_difference(y_true, y))

        return loss

    def _init_helper(self, weight_default_value=1.0):
        """Intialize the model, loss and optimizer.

        Args:
            weight_default_value (float): default weight value to be used to
                initialize WeightedMomentumOptimizer.
        """

        # Reset the graph so the variable names remain consistent
        tf.compat.v1.reset_default_graph()

        # Get model
        model = SimpleConvModel()

        # Get losses
        loss = self.get_loss(model)

        # Get LR scheduler
        lr_scheduler = ConstantLearningRateSchedule(self.LEARNING_RATE)

        # Get weighted momentum optimizer
        weighted_optimizer = WeightedMomentumOptimizer(
            lr_scheduler, weight_default_value=weight_default_value)

        return model, loss, weighted_optimizer

    def test_zero_weight_all_variables(self):
        """Test that if zero weight is applied to every variable, nothing changes."""

        model, loss, weighted_optimizer = self._init_helper(
            weight_default_value=0.0)
        weighted_optimizer.build()
        min_op = weighted_optimizer.minimize(loss, var_list=None)

        model_var_fetches = model.trainable_vars

        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            # Run training for a few steps. Variables should never be updated.
            initial_values = session.run(model_var_fetches)
            for _ in range(self.NUM_STEPS):
                session.run(min_op)
            final_values = session.run(model_var_fetches)

        for old, new in zip(initial_values, final_values):
            np.testing.assert_array_equal(old, new)
            assert not np.isnan(old).any()

    @pytest.fixture
    def regularly_optimized_values(self):
        """First train a model without any gradient weigting, ie. weight=1.0."""

        model, loss, weighted_optimizer = self._init_helper(
            weight_default_value=1.0)
        weighted_optimizer.build()
        min_op = weighted_optimizer.minimize(loss, var_list=None)

        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            # Run training for a few steps.
            for _ in range(self.NUM_STEPS):
                session.run(min_op)
            final_values = session.run(model.trainable_vars)

        tf.compat.v1.reset_default_graph()

        return final_values

    def test_grad_weights_dict_with_zero_weights(self):
        """Test if setting zero weights using the grad_weights_dict yields unchanged varaibles."""

        model, loss, weighted_optimizer = self._init_helper(
            weight_default_value=1.0)
        model_var_fetches = model.trainable_vars
        grad_weights_dict = {}
        for var in model_var_fetches:
            grad_weights_dict[var.op.name] = 0.0
        weighted_optimizer.set_grad_weights_dict(grad_weights_dict)

        weighted_optimizer.build()
        min_op = weighted_optimizer.minimize(loss, var_list=None)

        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            # Run training for a few steps. Variables should never be updated.
            initial_values = session.run(model_var_fetches)
            for _ in range(self.NUM_STEPS):
                session.run(min_op)
            final_values = session.run(model_var_fetches)

        for old, new in zip(initial_values, final_values):
            np.testing.assert_array_equal(old, new)
            assert not np.isnan(old).any()

    def test_grad_weights_dict_with_altzero_weights(self):
        """Test if setting alternate zero weights using the grad_weights_dict yields
        alternate unchanged variables."""

        model, loss, weighted_optimizer = self._init_helper(
            weight_default_value=1.0)
        model_var_fetches = model.trainable_vars
        grad_weights_dict = {}
        for idx, var in enumerate(model_var_fetches):
            grad_weights_dict[var.op.name] = 0.0 if idx % 2 == 0 else 0.5
        weighted_optimizer.set_grad_weights_dict(grad_weights_dict)

        weighted_optimizer.build()
        min_op = weighted_optimizer.minimize(loss, var_list=None)

        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            # Run training for a few steps. Variables should never be updated.
            initial_values = session.run(model_var_fetches)
            for _ in range(self.NUM_STEPS):
                session.run(min_op)
            final_values = session.run(model_var_fetches)

        for idx, (old, new) in enumerate(zip(initial_values, final_values)):
            if idx % 2 == 0:
                np.testing.assert_array_equal(old[idx], new[idx])
            else:
                assert not np.isclose(old[idx], new[idx]).any()
            assert not np.isnan(old).any()

    # TODO: @vpraveen: re-enable this test after you understand
    # why identity test is failing.
    @pytest.mark.skipif(
        os.getenv("RUN_ON_CI", "0") == "1",
        reason="Temporarily skipping from CI execution."
    )
    def test_training_with_idenity_weight_matches_default_training(
            self, regularly_optimized_values):
        """Test training with var grad weights set to 1.0.

        Args:
            regularly_optimized_values (pytest.fixture): to train a model
                without any gradient weigting for comparison.
        """

        model, loss, weighted_optimizer = self._init_helper(
            weight_default_value=1.0)
        model_var_fetches = model.trainable_vars
        grad_weights_dict = {}
        for var in model_var_fetches:
            grad_weights_dict[var.op.name] = 1.0
        weighted_optimizer.set_grad_weights_dict(grad_weights_dict)

        weighted_optimizer.build()
        min_op = weighted_optimizer.minimize(loss, var_list=None)

        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            # Run training for a few steps.
            for _ in range(self.NUM_STEPS):
                session.run(min_op)
            weighted_final_values = session.run(model.trainable_vars)

        for expected, actual in zip(regularly_optimized_values,
                                    weighted_final_values):
            np.testing.assert_array_equal(
                expected,
                actual)
            assert not np.isnan(expected).any()
