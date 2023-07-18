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
"""Weighted Momentum Optimizer with suport for layerwise gradient weighting/masking."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.optimizers.optimizer import Optimizer
from nvidia_tao_tf1.core.coreobject import save_args

import numpy as np
import tensorflow as tf


class WeightedMomentumOptimizer(Optimizer):
    """
    WeightedMomentumOptimizer class.

    Has suport for layerwise gradient weighting/masking for kernels and biases.
    """

    @save_args
    def __init__(self,
                 learning_rate_schedule,
                 grad_weights_dict=None,
                 weight_default_value=1.0,
                 momentum=0.9,
                 use_nesterov=False,
                 **kwargs):
        """__init__ method.

        learning_rate_schedule (LearningRateSchedule): The object from which we obtain the
            learning rate scalar tensor.
        momentum (float): A float value or a constant float tensor. The momentum factor. The method
            falls back into gradient descend optimizer when momentum is set to 0.
        use_nesterov (bool): If True, use the Nesterov momentum.
        """
        super(WeightedMomentumOptimizer,
              self).__init__(learning_rate_schedule=learning_rate_schedule,
                             **kwargs)
        self._learning_rate_schedule = learning_rate_schedule
        self._momentum = momentum
        self._use_nesterov = use_nesterov
        self._grad_weights_dict = grad_weights_dict if grad_weights_dict is not None else {}
        self._weight_default_value = weight_default_value
        self._optimizer_built = False

    def set_grad_weights_dict(self, grad_weights_dict):
        """Build the optimizer."""
        self._grad_weights_dict = grad_weights_dict

    def build(self):
        """Build the optimizer."""

        self._learning_rate_tensor = self._learning_rate_schedule.get_tensor()
        self._optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=self._learning_rate_tensor,
            momentum=self._momentum,
            use_nesterov=self._use_nesterov,
        )
        self._optimizer_built = True

        if not tf.executing_eagerly():
            self._distribute()

    @property
    def vars_and_grad_weights(self):
        """Return a handle on the trainable variables and their corresponding weights.

        Returns:
            (list): Returns a list of (variable, gradient_weights) tuples.

        Raises:
            RuntimeError: If the gradient weights / variables have yet to be defined.
        """
        if len(self._grad_weights) == 0 or len(self._var_list) == 0:
            raise RuntimeError(
                "Please call `minimize` or `compute_gradients` beforehand.")
        return list(zip(self._var_list, self._grad_weights))

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
        return self._optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                               global_step=global_step,
                                               name=name)

    def compute_gradients(self, loss, var_list=None, **kwargs):
        """Compute gradients and apply gradient weights.

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
        self._build_weights(var_list=var_list)

        if not self._optimizer_built:
            self.build()

        # Compute gradients as you would normally.
        grads_and_vars = self._optimizer.compute_gradients(loss=loss,
                                                           var_list=var_list,
                                                           **kwargs)
        # Apply the weights.
        weighted_grads_and_vars = []
        for i, (grad, var) in enumerate(grads_and_vars):
            weighted_grad = None
            if grad is not None:
                weighted_grad = grad * self._grad_weights[i]
            weighted_grads_and_vars.append((weighted_grad, var))

        self._grads_and_vars = grads_and_vars
        self._weighted_grads_and_vars = weighted_grads_and_vars

        return weighted_grads_and_vars

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
        # Compute the weighted gradients.
        grads_and_vars = self.compute_gradients(
            loss=loss,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )
        # Apply the weighted gradients.
        optimize_op = self.apply_gradients(grads_and_vars=grads_and_vars,
                                           global_step=global_step,
                                           name=name)

        return optimize_op, grads_and_vars, grads_and_vars

    def _build_weights(self, var_list=None):
        """Helper that defines the weights associated with the variables' gradients."""
        # Reset.
        self._grad_weights = []

        if var_list is None:
            var_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

        with tf.compat.v1.variable_scope("grad_weight",
                                         reuse=tf.compat.v1.AUTO_REUSE):
            # This scope allows us to reuse gradient weights that may already have been defined.
            # This is useful in e.g. the context of multi-task training, where each task may have
            # its own optimizer on a different set of variables, but some of which are common. For
            # those variables that are common (e.g. belong to the "feature extractor"), we want
            # to reuse the same gradient weights (which may also turn out to save on memory usage).
            # For those variables that are not common, the AUTO_REUSE results in the creation of
            # a new gradient weight.
            for var in var_list:
                # Give the weight variable a name. The string manipulations are necessary for TF not
                # to complain.
                weight_name = var.name[:var.name.find(":")]
                initial_value = np.ones(var.shape.as_list(),
                                        dtype=var.dtype.as_numpy_dtype)
                if weight_name in self._grad_weights_dict:
                    initial_value = self._grad_weights_dict[
                        weight_name] * initial_value
                else:
                    initial_value = self._weight_default_value * initial_value

                self._grad_weights.append(
                    tf.compat.v1.get_variable(
                        name=weight_name,
                        dtype=var.dtype.base_dtype,
                        initializer=initial_value,
                        trainable=False,
                    ))
        self._var_list = var_list
