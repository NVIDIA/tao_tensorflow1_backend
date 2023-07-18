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

"""Cost function config parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Objective(object):
    """Objective parameters."""

    def __init__(self, name, initial_weight, weight_target):
        """Constructor.

        Args:
            name (str): Name of the objective.
            initial_weight (float): Initial weight that will be assigned to this objective's cost.
            weight_target (float): Target weight that will be assigned to this objective's cost.

        Raises:
            ValueError: On invalid input args.
        """
        if name is None or name == "":
            raise ValueError("cost_function_parameters.Objective: name must be set.")
        if initial_weight < 0.0:
            raise ValueError("cost_function_parameters.Objective: initial_weight must be >= 0.")
        if weight_target < 0.0:
            raise ValueError("cost_function_parameters.Objective: weight_target must be >= 0.")

        self.name = name
        self.initial_weight = initial_weight
        self.weight_target = weight_target


class TargetClass(object):
    """Target class parameters."""

    def __init__(self, name, class_weight, coverage_foreground_weight, objectives):
        """Constructor.

        Args:
            name (str): Name of the target class.
            class_weight (float): Weight assigned to this target class's cost (all objectives
                combined).
            coverage_foreground_weight (float): Relative weight associated with the cost of cells
                where there is a foreground instance (i.e. the presence of what this TargetClass
                represents). Value should be in the range ]0., 1.[.
            objectives (list): Each item is a cost_function_parameters.Objective instance which
                contains the cost configuration options for this target class's objectives.

        Raises:
            ValueError: On invalid input args.
        """
        if name is None or name == "":
            raise ValueError("cost_function_parameters.TargetClass: name must be set.")
        if class_weight <= 0.0:
            raise ValueError("cost_function_parameters.TargetClass: class_weight must be > 0.")
        if coverage_foreground_weight <= 0.0 or coverage_foreground_weight >= 1.0:
            raise ValueError("cost_function_parameters.TargetClass: coverage_foreground_weight "
                             "must be in ]0., 1.[.")
        # Check that the sum of all objectives' weights for this target class is positive.
        initial_weight_sum = sum([objective.initial_weight for objective in objectives])
        weight_target_sum = sum([objective.weight_target for objective in objectives])
        if initial_weight_sum <= 0.0:
            raise ValueError("cost_function_parameters.objectives: Sum of initial_weight values "
                             "must be > 0.")
        if weight_target_sum <= 0.0:
            raise ValueError("cost_function_parameters.objectives: Sum of target_weight values "
                             "must be > 0.")

        self.name = name
        self.class_weight = class_weight
        self.coverage_foreground_weight = coverage_foreground_weight
        self.objectives = objectives


def build_target_class_list(cost_function_config):
    """Build a list of TargetClasses based on proto.

    Arguments:
        cost_function_config: CostFunctionConfig.
    Returns:
        A list of TargetClass instances.
    """
    target_classes = []
    for target_class in cost_function_config.target_classes:

        objectives = []
        for objective in target_class.objectives:

            objectives.append(Objective(objective.name,
                                        objective.initial_weight,
                                        objective.weight_target))

        target_classes.append(TargetClass(target_class.name,
                                          target_class.class_weight,
                                          target_class.coverage_foreground_weight,
                                          objectives))
    return target_classes


def get_target_class_names(cost_function_config):
    """Return a list of target class names.

    Args:
        cost_function_config (cost_function_pb2.CostFunctionConfig): proto message.

    Returns:
        List of target class names (str).
    """
    return [target_class.name for target_class in cost_function_config.target_classes]
