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

"""Cost auto weight hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_function_parameters import (
    build_target_class_list
)
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer


def build_cost_auto_weight_hook(cost_function_config, steps_per_epoch):
    """Build a CostAutoWeightHook based on proto.

    Arguments:
        cost_function_config: CostFunctionConfig.
        steps_per_epoch (int): Number of steps per epoch.
    Returns:
        A CostAutoWeightHook instance.
    """
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0")

    if not cost_function_config.target_classes :
        raise ValueError("CostFunctionConfig should have at least one class")

    for target_class in cost_function_config.target_classes:
        if not target_class.objectives :
            raise ValueError("CostFunctionConfig.target_classes should have at least one "
                             "objective")

    return CostAutoWeightHook(build_target_class_list(cost_function_config),
                              cost_function_config.enable_autoweighting,
                              cost_function_config.min_objective_weight,
                              cost_function_config.max_objective_weight,
                              steps_per_epoch)


class CostAutoWeightHook(tf.estimator.SessionRunHook):
    """Class for computing objective auto weighting and total cost."""

    def __init__(self, target_classes, enable_autoweighting,
                 min_objective_weight, max_objective_weight, steps_per_epoch):
        """__init__ method.

        Compute normalized initial values for class and objective weights based on parameters.
        Create objective auto weighting variables and update ops. Add update ops to lists for
        execution on epoch begin/end callbacks.

        Args:
            target_classes: A list of TargetClass objects.
            enable_autoweighting (bool): Whether auto weighting is enabled.
            min_objective_weight (float): Minimum objective cost weight is clamped to this value.
            max_objective_weight (float): Maximum objective cost weight is clamped to this value.
            steps_per_epoch (int): Number of steps per epoch.
        """
        self.target_classes = target_classes
        self.enable_autoweighting = enable_autoweighting
        self.min_objective_weight = min_objective_weight
        self.max_objective_weight = max_objective_weight
        self.steps_per_epoch = steps_per_epoch
        self.steps_counter = 0
        self.debug = False

        # Initialize lists of callback update ops.
        self.on_epoch_begin_updates = []
        self.on_epoch_end_updates = []

        # Stored values for testing purposes.
        self._before_run_values = []
        self._after_run_values = []

        self._init_target_class_weights()
        self._init_objective_weights()

    def _init_target_class_weights(self):
        # Compute sum of class weight initial values for normalization.
        target_class_weight_sum = 0.
        for target_class in self.target_classes:
            target_class_weight_sum += target_class.class_weight

        # Initialize class weights (currently constant).
        self.target_class_weights = {}
        for target_class in self.target_classes:
            # Normalize initial value.
            init_val = target_class.class_weight / target_class_weight_sum
            self.target_class_weights[target_class.name] = tf.constant(init_val, dtype=tf.float32)

    def _init_objective_weights(self):
        # Initialize objective weighting.
        self.cost_sums = {}
        self.objective_weights = {}
        for target_class in self.target_classes:
            # Compute objective weight sum for normalization.
            objective_weight_sum = 0.
            for objective in target_class.objectives:
                objective_weight_sum += objective.initial_weight

            self.cost_sums[target_class.name] = {}
            self.objective_weights[target_class.name] = {}
            for objective in target_class.objectives:
                # Create cost sum variables.
                with tf.variable_scope('cost_sums'):
                    init = tf.constant(0., dtype=tf.float32)
                    name = '%s-%s' % (target_class.name, objective.name)
                    var = tf.get_variable(name, initializer=init, trainable=False)
                    self.cost_sums[target_class.name][objective.name] = var
                    # Reset the value at the beginning of every epoch.
                    self.on_epoch_begin_updates.append(tf.assign(ref=var, value=init))

                # Create objective weight variables.
                with tf.variable_scope('objective_weights'):
                    # Normalize initial value.
                    if objective_weight_sum:
                        init_val = objective.initial_weight / objective_weight_sum
                    else:
                        init_val = 0.0
                    init = tf.constant(init_val, dtype=tf.float32)
                    name = '%s-%s' % (target_class.name, objective.name)
                    var = tf.get_variable(name, initializer=init, trainable=False)
                    self.objective_weights[target_class.name][objective.name] = var

            if self.enable_autoweighting:
                # Construct objective weight update op.
                # Step 1: compute objective weights and their sum based on objective cost
                # means over the last epoch.
                weights = {}
                sum_weights = 0.
                for objective in target_class.objectives:
                    # Note: cost sums are actually sums of minibatch means, so in principle we
                    # should divide them by the number of minibatches per epoch, but since we're
                    # normalizing the weights, division by the number of minibatches cancels out.
                    # Note 2: for multi-GPU, we need to average over all GPUs in order to keep
                    # the weights in sync. Each process will compute the same updates, so
                    # there's no need to broadcast the results. Allreduce computes a sum of
                    # means so we should divide the result by the number of GPUs, but again
                    # the division cancels out due to normalization.
                    obj_mean = self.cost_sums[target_class.name][objective.name]
                    obj_mean = distribution.get_distributor().allreduce(obj_mean)
                    # Compute 1/obj_mean. If obj_mean is 0, result is 0.
                    oo_obj_mean = tf.where(tf.equal(obj_mean, 0.), obj_mean, 1. / obj_mean)
                    weights[objective.name] = objective.weight_target * oo_obj_mean
                    sum_weights += weights[objective.name]

                # Step 2: compute weight normalizer.
                # Note: in case sum_weights is 0, we will retain the old weights so the value of
                # nrm doesn't matter.
                nrm = tf.where(tf.equal(sum_weights, 0.), 0., 1. / sum_weights)

                # Step 3: compute normalized objective weights and schedule weight update op.
                for objective in target_class.objectives:
                    w = weights[objective.name] * nrm
                    w = tf.maximum(w, self.min_objective_weight)
                    w = tf.minimum(w, self.max_objective_weight)

                    # If weight sum is 0, setting objective weight does not make sense ->
                    # retain old value.
                    oldw = self.objective_weights[target_class.name][objective.name]
                    w = tf.where(tf.equal(sum_weights, 0.), oldw, w)
                    # Schedule objective weight update op to be executed at the end of each epoch.
                    op = tf.assign(self.objective_weights[target_class.name][objective.name], w)
                    if self.debug:
                        op = tf.Print(op, [op], "objective weight gpu %d - %s - %s = " %
                                      (distribution.get_distributor().local_rank(),
                                       target_class.name, objective.name))
                    self.on_epoch_end_updates.append(op)

    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        Args:
            session: A TensorFlow Session that has been created.
            coord: A Coordinator object which keeps track of all threads.
        """
        self.session = session

    def before_run(self, run_context):
        """Called before each call to run().

        Run epoch begin updates before the first step of each epoch.

        Args:
            run_context: A SessionRunContext object.
        """
        if self.steps_counter == 0:
            # Store value for testing purposes.
            self._before_run_values = self.session.run(self.on_epoch_begin_updates)

    def after_run(self, run_context, run_values):
        """Called after each call to run().

        Run epoch end updates after the last step of each epoch.

        Args:
            run_context: A SessionRunContext object.
            run_values: A SessionRunValues object.
        """
        self.steps_counter += 1
        if self.steps_counter == self.steps_per_epoch:
            self.steps_counter = 0
            # Store value for testing purposes.
            self._after_run_values = self.session.run(self.on_epoch_end_updates)

    def cost_combiner_func(self, component_costs):
        """Cost function.

        Args:
            component_costs: Per target class per objective cost means over a minibatch.
        Returns:
            Total minibatch cost.
        """
        # Target_classes in component_costs must be present in cost_function_parameters.
        # Cost_function_parameters must not have additional target_classes.
        assert {target_class.name for target_class in self.target_classes} ==\
            set(component_costs)
        # Compute a weighted sum of component costs.
        total_cost = 0.0
        costs = {}
        for target_class in self.target_classes:
            # Objectives in component_costs must be present in cost_function_parameters.
            # Cost_function_parameters must not have additional objectives.
            assert {objective.name for objective in target_class.objectives} ==\
                set(component_costs[target_class.name])
            costs[target_class.name] = {}
            for objective in target_class.objectives:
                # Average cost over minibatch and spatial dimensions.
                mean_cost = component_costs[target_class.name][objective.name]
                # Accumulate per class per objective cost, and total_cost.
                # Control dependency needed since total_cost doesn't depend on the cost sum
                # variables and thus they wouldn't get updated otherwise.
                op = tf.assign_add(self.cost_sums[target_class.name][objective.name], mean_cost)
                with tf.control_dependencies([op]):
                    cost = mean_cost * self.target_class_weights[target_class.name] *\
                        self.objective_weights[target_class.name][objective.name]
                    costs[target_class.name][objective.name] = cost
                    total_cost += cost

        # Compute and visualize percentage of how much each component contributes to the total cost.
        if Visualizer.enabled:
            for target_class in self.target_classes:
                for objective in target_class.objectives:
                    percentage = 100. * costs[target_class.name][objective.name] / total_cost
                    tf.summary.scalar('cost_percentage_%s_%s' % (target_class.name, objective.name),
                                      percentage)
                    tf.summary.scalar('cost_autoweight_%s_%s' % (target_class.name, objective.name),
                                      self.objective_weights[target_class.name][objective.name])

        return total_cost
