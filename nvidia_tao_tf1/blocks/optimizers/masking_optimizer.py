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

"""Optimizer that wraps another tf.Optimizer and applies gradient masks.

Potential areas of improvement / TODO:
    * It would nice to be able to provide masks separately from the optimizer, as that would
      allow us to delegate potentially complex mask definitions (see the test file for an
      illustration of why this current design is slightly less geared towards this scenario).
      However, it is crucial that the (grad, var) pairs be correctly defined, which at this point
      is done in the `minimize` when we either get the `var_list` as input or just get all
      trainable variables in the graph. If this is done outside, then users have to make sure
      gradient masks and trainable variables are given in corresponding orders.
    * Come up with a way of capturing dependencies between different variables (and therefore masks)
      This is required when e.g. pruning channel-wise (instead of element-wise), and a layer
      has both kernel and bias, or is followed by a batch normalization layer.
      Note that it's also a little fuzzy at this point whether this responsibility belongs here
      or somewhere else (e.g. in the mechanism that actually does the pruning).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.optimizers.optimizer import Optimizer
from nvidia_tao_tf1.core.coreobject import save_args


class MaskingOptimizer(Optimizer):
    """Optimizer that wraps another nvidia_tao_tf1.blocks.Optimizer and applies gradient masks."""

    @save_args
    def __init__(self, optimizer, mask_initial_value=1.0):
        """Constructor.

        Args:
            optimizer (nvidia_tao_tf1.blocks.Optimizer): Original optimizer this one wraps.
            mask_initial_value (float): Initial value for the masks. The default of 1.0 amounts
                to the mask being a (mathematical) no-op.
        """
        super(MaskingOptimizer, self).__init__(
            learning_rate_schedule=optimizer.learning_rate_schedule
        )
        self._optimizer = optimizer
        self._mask_initial_value = mask_initial_value
        # These will be updated later on at graph building time.
        self._optimizer_built = False
        self._grad_masks = []
        self._var_list = []

    def build(self):
        """Build the optimizer."""
        self._optimizer.build()
        self._optimizer_built = True

    @property
    def vars_and_grad_masks(self):
        """Return a handle on the trainable variables and their corresponding masks.

        Returns:
            (list): Returns a list of (variable, gradient_mask) tuples.

        Raises:
            RuntimeError: If the gradient masks / variables have yet to be defined.
        """
        if len(self._grad_masks) == 0 or len(self._var_list) == 0:
            raise RuntimeError(
                "Please call `minimize` or `compute_gradients` beforehand."
            )
        return list(zip(self._var_list, self._grad_masks))

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients.

        Args:
            grads_and_vars (list): List of (gradient, variable) pairs as returned by
                `compute_gradients()`.
            global_step (tf.Variable): Optional variable to increment by one after the variables
                have been updated.
            name (str): Optional name for the returned operation. Default to the name passed to the
                constructor.

        Returns:
            (tf.Operation): An operation that applies the specified gradients. If `global_step`
                was not `None`, that operation also increments `global_step`.
        """
        return self._optimizer.apply_gradients(
            grads_and_vars=grads_and_vars, global_step=global_step, name=name
        )

    def compute_gradients(self, loss, var_list=None, **kwargs):
        """Compute gradients and apply gradient masks.

        Args:
            loss (tf.Tensor): A tensor containing the value to compute gradients for.
            var_list (list): Optional list or tuple of `tf.Variable` to compute gradients for.
                Defaults to the list of variables collected in the graph under the key
                `GraphKeys.TRAINABLE_VARIABLES`.
            gate_gradients: How to gate the computation of gradients. Can be
                `tf.compat.v1.<GATE_NONE, GATE_OP, GATE_GRAPH>`.
            aggregation_method: Specifies the method used to combine gradient terms. Valid values
                are defined in the class `AggregationMethod`.
            colocate_gradients_with_ops (bool): If `True`, try colocating gradients with the
                corresponding op.
            grad_loss (tf.Tensor): Optional. A tensor holding the gradient computed for loss.

        Returns:
            A list of (gradient, variable) pairs. Variable is always present, but gradient can
                be `None`.
        """
        self._build_masks(var_list=var_list)

        if not self._optimizer_built:
            self.build()

        # Compute gradients as you would normally.
        grads_and_vars = self._optimizer.compute_gradients(
            loss=loss, var_list=var_list, **kwargs
        )

        # Apply the masks.
        masked_grads_and_vars = []
        for i, (grad, var) in enumerate(grads_and_vars):
            masked_grad = None
            if grad is not None:
                masked_grad = grad * self._grad_masks[i]
            masked_grads_and_vars.append((masked_grad, var))

        self._grads_and_vars = grads_and_vars
        self._masked_grads_and_vars = masked_grads_and_vars

        return masked_grads_and_vars

    def minimize(
        self,
        loss,
        global_step=None,
        var_list=None,
        gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        name=None,
        grad_loss=None,
    ):
        """Minimize op.

        Args:
            loss (tf.Tensor): A tensor containing the value to minimize.
            global_step (tf.Variable): Optional variable to increment by one after the variables
                have been updated.
            var_list (list): Optional list or tuple of `tf.Variable` objects to update when
                minimizing the `loss`. Defaults to the list of variables collected in the graph
                under the key `GraphKeys.TRAINABLE_VARIABLES`.
            gate_gradients: How to gate the computation of gradients. Can be
                `tf.compat.v1.<GATE_NONE, GATE_OP, GATE_GRAPH>`.
            aggregation_method: Specifies the method used to combine gradient terms. Valid values
                are defined in the class `AggregationMethod`.
            colocate_gradients_with_ops (bool): If `True`, try colocating gradients with the
                corresponding op.
            name (str): Optional named for the returned operation.
            grad_loss (tf.Tensor): Optional. A tensor holding the gradient computed for loss.

        Returns:
            (tf.Operation): Op that updates the variables in `var_list`. If `global_step` was not
                `None`, that operation also increments `global_step`.
        """
        # Compute the masked gradients.
        grads_and_vars = self.compute_gradients(
            loss=loss,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )
        # Apply the masked gradients.
        optimize_op = self.apply_gradients(
            grads_and_vars=grads_and_vars, global_step=global_step, name=name
        )

        return optimize_op

    def _build_masks(self, var_list=None):
        """Helper that defines the masks associated with the variables' gradients."""
        # Reset.
        self._grad_masks = []

        if var_list is None:
            var_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
            )

        with tf.compat.v1.variable_scope("grad_mask", reuse=tf.compat.v1.AUTO_REUSE):
            # This scope allows us to reuse gradient masks that may already have been defined.
            # This is useful in e.g. the context of multi-task training, where each task may have
            # its own optimizer on a different set of variables, but some of which are common. For
            # those variables that are common (e.g. belong to the "feature extractor"), we want
            # to reuse the same gradient masks (which may also turn out to save on memory usage).
            # For those variables that are not common, the AUTO_REUSE results in the creation of
            # a new gradient mask.
            for var in var_list:
                initial_value = self._mask_initial_value * np.ones(
                    var.shape.as_list(), dtype=var.dtype.as_numpy_dtype
                )
                # Give the mask variable a name. The string manipulations are necessary for TF not
                # to complain.
                mask_name = var.name[: var.name.find(":")]
                self._grad_masks.append(
                    tf.compat.v1.get_variable(
                        name=mask_name,
                        dtype=var.dtype.base_dtype,
                        initializer=initial_value,
                        trainable=False,
                    )
                )
        self._var_list = var_list

    @property
    def learning_rate_tensor(self):
        """Handle on the learning rate tensor."""
        return self._optimizer.learning_rate_tensor

    @property
    def learning_rate_schedule(self):
        """Handle on the learning rate schedule.

        Returns:
            (LearningRateSchedule)
        """
        return self._optimizer.learning_rate_schedule
