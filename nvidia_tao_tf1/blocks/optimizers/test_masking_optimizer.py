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

"""Tests for the MaskingOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.learning_rate_schedules import ConstantLearningRateSchedule
from nvidia_tao_tf1.blocks.optimizers.gradient_descent_optimizer import (
    GradientDescentOptimizer,
)
from nvidia_tao_tf1.blocks.optimizers.masking_optimizer import MaskingOptimizer
from nvidia_tao_tf1.blocks.optimizers.test_fixtures import get_random_boolean_mask
from nvidia_tao_tf1.core.training import enable_deterministic_training


@pytest.fixture(scope="module", autouse=True)
def determinism():
    enable_deterministic_training()


def test_apply_gradients_forwards_args():
    optimizer = mock.Mock()
    masking_optimizer = MaskingOptimizer(optimizer)

    grads_and_vars = mock.Mock()
    global_step = mock.Mock()
    name = mock.Mock()
    masking_optimizer.apply_gradients(
        grads_and_vars=grads_and_vars, global_step=global_step, name=name
    )

    optimizer.apply_gradients.assert_called_once_with(
        grads_and_vars=grads_and_vars, global_step=global_step, name=name
    )


class EarlyExitException(Exception):
    pass


def test_compute_gradients_forwards_args():
    optimizer = mock.Mock()
    optimizer.compute_gradients.side_effect = EarlyExitException
    masking_optimizer = MaskingOptimizer(optimizer)

    loss = mock.Mock()
    var_list = mock.Mock()
    kwargs = {"some_keyword": mock.Mock()}

    with mock.patch.object(masking_optimizer, "_build_masks"), pytest.raises(
        EarlyExitException
    ):
        masking_optimizer.compute_gradients(loss=loss, var_list=var_list, **kwargs)

    optimizer.compute_gradients.assert_called_once_with(
        loss=loss, var_list=var_list, **kwargs
    )


def _get_optimizer(lr):
    return GradientDescentOptimizer(
        learning_rate_schedule=ConstantLearningRateSchedule(lr)
    )


class SimpleConvModel(object):
    """Simple 2-layer convolutional model to test with."""

    def __init__(self, initial_value=0.125):
        """Constructor."""
        self.initial_value = initial_value
        self._built = False
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    @property
    def trainable_vars(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _build(self, x):
        if self._built:
            return

        # Build W1 and b1.
        in_channels = x.shape[1]  # NCHW
        out_channels = 7
        W1_initial_value = np.ones([3, 3, in_channels, out_channels], dtype=np.float32)
        W1_initial_value *= self.initial_value

        self.W1 = tf.Variable(initial_value=W1_initial_value, trainable=True)
        self.b1 = tf.Variable(
            initial_value=np.zeros([out_channels], dtype=np.float32), trainable=True
        )

        # Build W2 and b2.
        in_channels = out_channels
        out_channels = 13
        W2_initial_value = np.ones([5, 5, in_channels, out_channels], dtype=np.float32)
        W2_initial_value *= self.initial_value

        self.W2 = tf.Variable(initial_value=W2_initial_value, trainable=True)
        self.b2 = tf.Variable(
            initial_value=np.zeros([out_channels], dtype=np.float32), trainable=True
        )

    def __call__(self, x):
        """Call method."""
        self._build(x)
        # h = W1*x + b1
        h = tf.nn.conv2d(
            input=x, filters=self.W1, strides=1, padding="VALID", data_format="NCHW"
        )
        h = tf.nn.bias_add(value=h, bias=self.b1, data_format="NCHW")
        # y = W2*h + b2
        y = tf.nn.conv2d(
            input=h, filters=self.W2, strides=1, padding="VALID", data_format="NCHW"
        )
        y = tf.nn.bias_add(value=y, bias=self.b2, data_format="NCHW")

        return y


class TestMaskingScenarios(object):
    """Tests with some masking happening."""

    NUM_STEPS = 3
    LEARNING_RATE = 1e-5

    @staticmethod
    def get_loss(model):
        """Loss tensor to optimize."""
        x = tf.ones([3, 5, 16, 32])  # NCHW.
        y = model(x)

        y_true = tf.ones_like(y)

        loss = tf.reduce_sum(input_tensor=tf.compat.v1.squared_difference(y_true, y))

        return loss

    def test_mask_all_variables(self):
        """Test that if a null mask is applied to every variable, nothing changes."""
        model = SimpleConvModel()
        loss = self.get_loss(model)

        optimizer = _get_optimizer(lr=self.LEARNING_RATE)
        masking_optimizer = MaskingOptimizer(
            optimizer=optimizer, mask_initial_value=0.0
        )

        min_op = masking_optimizer.minimize(loss=loss, var_list=None)

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
        """First train a model without any gradient masking."""
        model = SimpleConvModel()
        loss = self.get_loss(model)

        optimizer = _get_optimizer(lr=self.LEARNING_RATE)

        min_op = optimizer.minimize(loss=loss, var_list=None)

        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            # Run training for a few steps.
            for _ in range(self.NUM_STEPS):
                session.run(min_op)
            final_values = session.run(model.trainable_vars)

        tf.compat.v1.reset_default_graph()

        return final_values

    def test_training_with_no_op_masks_matches_maskless_training(
        self, regularly_optimized_values
    ):
        """Test training with masks set to 1.0."""
        model = SimpleConvModel()
        loss = self.get_loss(model)

        optimizer = _get_optimizer(lr=self.LEARNING_RATE)
        masking_optimizer = MaskingOptimizer(
            optimizer=optimizer, mask_initial_value=1.0
        )

        min_op = masking_optimizer.minimize(loss=loss, var_list=None)

        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            # Run training for a few steps.
            for _ in range(self.NUM_STEPS):
                session.run(min_op)
            masked_final_values = session.run(model.trainable_vars)

        for expected, actual in zip(regularly_optimized_values, masked_final_values):
            np.testing.assert_array_equal(expected, actual)
            assert not np.isnan(expected).any()

    def test_masks_are_applied_during_training(self):
        """Test masking at the element level."""
        model = SimpleConvModel()
        loss = self.get_loss(model)

        optimizer = _get_optimizer(lr=self.LEARNING_RATE)
        masking_optimizer = MaskingOptimizer(
            optimizer=optimizer, mask_initial_value=1.0
        )

        min_op = masking_optimizer.minimize(loss=loss, var_list=None)

        trainable_vars, grad_masks = zip(*masking_optimizer.vars_and_grad_masks)

        # Define some explicit assign ops for the masks.
        assign_ops = []
        fine_grained_masks = []
        for mask in grad_masks:
            mask_value = get_random_boolean_mask(shape=mask.shape)
            fine_grained_masks.append(mask_value)
            mask_value = mask_value.astype(mask.dtype.as_numpy_dtype)
            assign_ops.append(tf.compat.v1.assign(mask, mask_value))

        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            initial_values = session.run(trainable_vars)
            # Set the gradient masks to the values defined above.
            session.run(assign_ops)
            # Train for a few steps.
            for _ in range(self.NUM_STEPS):
                session.run(min_op)
            final_values = session.run(trainable_vars)

        # Now, check that the variable elements whose corresponding mask element was 0 did not
        # budge, whereas those whose mask element was 1 did.
        for old, new, mask in zip(initial_values, final_values, fine_grained_masks):
            zero_indices = np.where(fine_grained_masks == 0.0)
            one_indices = np.where(fine_grained_masks == 1.0)
            # These should have been frozen.
            np.testing.assert_array_equal(old[zero_indices], new[zero_indices])
            # These should have changed.
            assert not np.isclose(old[one_indices], new[one_indices]).any()
