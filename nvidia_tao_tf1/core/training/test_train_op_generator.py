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

"""Tests for TrainOpGenerator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


from nvidia_tao_tf1.core.training.train_op_generator import TrainOpGenerator


class TestTrainOpGenerator(object):
    """Test TrainOpGenerator class."""

    INITIAL_VALUE = 10.0
    INCREMENT = 0.1
    DECREMENT = 1.0

    def _get_generator(self, cost_scaling_enabled):
        """Setup a training op generator object."""
        return TrainOpGenerator(
            cost_scaling_enabled=cost_scaling_enabled,
            cost_scaling_init=self.INITIAL_VALUE,
            cost_scaling_inc=self.INCREMENT,
            cost_scaling_dec=self.DECREMENT,
        )

    def test_cost_scaling(self):
        """Verify dynamic cost scaling."""
        # Create a minimal graph.
        data = tf.Variable(
            initial_value=100.0, dtype=tf.float32, name="data", trainable=False
        )
        weight = tf.Variable(initial_value=1.0, dtype=tf.float32, name="weight")
        graph_out = weight * data
        target = 2.0
        # Create L1 cost.
        cost = tf.abs(graph_out - target)
        # Create SGD optimizer.
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        # Create the training op.
        generator = self._get_generator(cost_scaling_enabled=True)
        train_op = generator.get_train_op(optimizer, cost)
        # Create initializers.
        init_ops = [
            var.initializer
            for var in tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
            )
        ]

        # Initialize and run training.
        with tf.compat.v1.Session() as session:
            # Initialize all variables.
            session.run(init_ops)
            # Verify that cost scaling is properly initialized.
            cost_scaling_exponent = session.run("cost_scaling_exponent:0")
            np.testing.assert_allclose(cost_scaling_exponent, self.INITIAL_VALUE)

            # Run one training iteration.
            session.run(train_op)
            # Verify that scaling exponent is increased.
            cost_scaling_exponent = session.run("cost_scaling_exponent:0")
            expected_value = self.INITIAL_VALUE + self.INCREMENT
            np.testing.assert_allclose(cost_scaling_exponent, expected_value)

            # Store the weight after 1st iteration.
            weight_step_1 = session.run(weight)

            # Make data non-finite, which triggers cost scaling backoff.
            session.run(tf.compat.v1.assign(data, np.inf))
            # Run another step.
            session.run(train_op)
            # Verify that scaling exponent is decreased.
            cost_scaling_exponent = session.run("cost_scaling_exponent:0")
            expected_value = self.INITIAL_VALUE + self.INCREMENT - self.DECREMENT
            np.testing.assert_allclose(cost_scaling_exponent, expected_value)

            # Get the weight after 2nd iteration, make sure bad gradient was not applied.
            weight_step_2 = session.run(weight)
            np.testing.assert_allclose(weight_step_1, weight_step_2)

    def test_without_cost_scaling(self):
        """Verify that train op generator returns a good op, when cost scaling is disabled."""
        # Create dummy optimizer and cost.
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        cost = tf.Variable(initial_value=1.0, dtype=tf.float32, name="dummy_cost")
        # Create training op generator and the op itself.
        generator = self._get_generator(cost_scaling_enabled=False)
        train_op = generator.get_train_op(optimizer, cost)
        # Create initializers.
        init_ops = [
            var.initializer
            for var in tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
            )
        ]
        # Initialize and make sure that the training op works OK.
        with tf.compat.v1.Session() as session:
            session.run(init_ops)
            cost_step_0 = session.run(cost)
            session.run(train_op)
            cost_step_1 = session.run(cost)
            assert cost_step_1 < cost_step_0, "Optimizer increased the cost"
