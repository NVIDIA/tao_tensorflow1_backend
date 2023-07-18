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

"""TrainOpGenerator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from keras import backend as K
import tensorflow as tf

logger = logging.getLogger(__name__)


class TrainOpGenerator(object):
    """TrainOpGenerator class.

    TrainOpGenerator contains parameters for dynamic cost scaling required in mixed-precision
    training. It creates a TF op that includes the adaptation logic for dynamic cost scaling.

    The cost scaling feature can be disabled through parameters and in this case the generator
    returns a plain train op by calling optimizer.minimize.
    """

    def __init__(
        self,
        cost_scaling_enabled,
        cost_scaling_init,
        cost_scaling_inc,
        cost_scaling_dec,
    ):
        """Setup a train op generator.

        Args:
            cost_scaling_enabled (bool): Enable or disable dynamic cost scaling.
            cost_scaling_init (float): Initial value for cost scaling exponent.
            cost_scaling_inc (float): Added to scaling exponent if gradients are OK.
            cost_scaling_dec (float): Subtracted from scaling exponent if gradients overflow.
        """
        # Store the parameters.
        self.cost_scaling_enabled = cost_scaling_enabled
        self.cost_scaling_init = cost_scaling_init
        self.cost_scaling_inc = cost_scaling_inc
        self.cost_scaling_dec = cost_scaling_dec

        # Sanity check: allow user to train float16 without cost scaling, but give a warning.
        if K.floatx() == "float16" and not self.cost_scaling_enabled:
            logger.warning(
                "Cost scaling is disabled while mixed-precision training is enabled."
            )

    def get_train_op(self, optimizer, total_cost, var_list=None):
        """Return a train op with or without cost scaling.

        Args:
            optimizer (horovod.tensorflow.DistributedOptimizer): TF-compatible optimizer object.
            total_cost (float32 tf.Tensor): Scalar cost value used for computing gradients.
            var_list (list<tf.Variable>): Variables to update to minimize loss. If None, defaults
                to the list of variables collected in the graph under the key
                GraphKeys.TRAINABLE_VARIABLES.
        """
        if self.cost_scaling_enabled:
            return self._get_train_op_with_cost_scaling(optimizer, total_cost, var_list)

        return self._get_train_op_without_cost_scaling(optimizer, total_cost, var_list)

    def _get_train_op_without_cost_scaling(self, optimizer, total_cost, var_list):
        """Return a train op without cost scaling.

        Args:
            optimizer (horovod.tensorflow.DistributedOptimizer): TF-compatible optimizer object.
            total_cost (float32 tf.Tensor): Scalar cost value used for computing gradients.
            var_list (list<tf.Variable>): Variables to update to minimize loss. If None, defaults
                to the list of variables collected in the graph under the key
                GraphKeys.TRAINABLE_VARIABLES.
        """
        global_step = tf.compat.v1.train.get_or_create_global_step()
        return optimizer.minimize(
            loss=total_cost, global_step=global_step, var_list=var_list
        )

    def _get_train_op_with_cost_scaling(self, optimizer, total_cost, var_list):
        """Return a train op with cost scaling.

        Args:
            optimizer (horovod.tensorflow.DistributedOptimizer): TF-compatible optimizer object.
            total_cost (float32 tf.Tensor): Scalar cost value used for computing gradients.
            var_list (list<tf.Variable>): Variables to update to minimize loss. If None, defaults
                to the list of variables collected in the graph under the key
                GraphKeys.TRAINABLE_VARIABLES.
        """
        # Create a persistent cost scaling exponent.
        cost_scaling_exponent = tf.Variable(
            initial_value=self.cost_scaling_init,
            dtype=tf.float32,
            name="cost_scaling_exponent",
            trainable=False,
        )
        # Log the number of discarded gradients.
        bad_grad_counter = tf.Variable(
            initial_value=0, dtype=tf.int64, name="bad_grad_counter", trainable=False
        )

        # Multiply the total cost by 2^X.
        cost_multiplier = 2.0 ** cost_scaling_exponent
        inverse_multiplier = 1.0 / cost_multiplier
        scaled_total_cost = total_cost * cost_multiplier

        # Add tensorboard summaries.
        tf.compat.v1.summary.scalar("scaled_total_cost", scaled_total_cost)
        tf.compat.v1.summary.scalar("cost_scaling_exponent", cost_scaling_exponent)
        tf.compat.v1.summary.scalar("bad_grad_counter", bad_grad_counter)

        # Get the gradient tensors with the scaled cost.
        grads_and_vars = optimizer.compute_gradients(
            loss=scaled_total_cost, var_list=var_list
        )

        # Bring the gradient scale back to original (divide by 2^X).
        grads_and_vars = [
            (grad * inverse_multiplier, var)
            for grad, var in grads_and_vars
            if grad is not None
        ]

        # Check that gradients are finite.
        grad_ok = tf.reduce_all(
            input_tensor=tf.stack(
                [
                    tf.reduce_all(input_tensor=tf.math.is_finite(grad))
                    for grad, var in grads_and_vars
                ]
            )
        )

        # When gradients are not OK, apply zeros to maintain Horovod multi-GPU sync.
        zero_grads_and_vars = [
            (tf.zeros_like(var), var) for grad, var in grads_and_vars
        ]

        # Get global step.
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Create a conditional training op.
        train_op = tf.cond(
            # Condition is the finiteness of the gradients.
            pred=grad_ok,
            # Finite gradients -> increase scaling and apply gradients.
            true_fn=lambda: tf.group(
                tf.compat.v1.assign_add(cost_scaling_exponent, self.cost_scaling_inc),
                optimizer.apply_gradients(
                    grads_and_vars=grads_and_vars, global_step=global_step
                ),
            ),
            # Invalid gradients -> decrease scaling and apply zero-gradients.
            false_fn=lambda: tf.group(
                tf.compat.v1.assign_add(bad_grad_counter, 1),
                tf.compat.v1.assign_add(cost_scaling_exponent, -self.cost_scaling_dec),
                optimizer.apply_gradients(
                    grads_and_vars=zero_grads_and_vars, global_step=global_step
                ),
            ),
        )
        return train_op
