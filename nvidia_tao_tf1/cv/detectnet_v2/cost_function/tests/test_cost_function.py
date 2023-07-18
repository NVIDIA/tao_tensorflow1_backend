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

"""Test cost functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from six.moves import range
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_auto_weight_hook import (
    build_cost_auto_weight_hook,
    CostAutoWeightHook
)
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_function_parameters import (
    build_target_class_list,
    get_target_class_names,
    Objective,
    TargetClass
)
from nvidia_tao_tf1.cv.detectnet_v2.model.utilities import get_class_predictions
from nvidia_tao_tf1.cv.detectnet_v2.objectives.objective_set import build_objective_set
from nvidia_tao_tf1.cv.detectnet_v2.proto.cost_function_config_pb2 import CostFunctionConfig
from nvidia_tao_tf1.cv.detectnet_v2.proto.model_config_pb2 import ModelConfig
from nvidia_tao_tf1.cv.detectnet_v2.proto.visualizer_config_pb2 import VisualizerConfig
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer


verbose = False


def _weighted_BCE_cost(target, pred, weight):
    """Elementwise weighted BCE cost."""
    EPSILON = 1e-05
    BCE = -(target * np.log(pred + EPSILON) + (1.0 - target) * np.log(1.0 - pred + EPSILON))
    weight_vec_for_one = weight * np.ones_like(target)
    weight_vec_for_zero = (1.0 - weight) * np.ones_like(target)
    weights_tensor = np.where(target > 0.5, weight_vec_for_one, weight_vec_for_zero)
    return weights_tensor * BCE


def _weighted_L1_cost(target, pred, weight):
    """Weighted L1 cost."""
    weight = np.ones_like(target) * weight
    dist = np.abs(pred - target)
    return np.multiply(weight, dist)


class TestCostFunction:
    """Test cost functions."""

    def _compute_expected_total_cost(self, cost_function_config, target_dict, predictions_dict,
                                     target_class_weights, objective_weights, cost_means):
        # Compute expected total cost value.
        total_cost = 0.0
        for target_class in cost_function_config.target_classes:
            # Targets.
            cov_target = target_dict[target_class.name]['cov']
            cov_norm_target = target_dict[target_class.name]['cov_norm']
            bbox_target = target_dict[target_class.name]['bbox']
            # Predictions.
            cov_pred = predictions_dict[target_class.name]['cov']
            bbox_pred = predictions_dict[target_class.name]['bbox']
            # Compute costs.
            cov_cost = np.mean(_weighted_BCE_cost(cov_target, cov_pred,
                                                  target_class.coverage_foreground_weight))
            bbox_cost = np.mean(_weighted_L1_cost(bbox_target, bbox_pred, cov_norm_target))
            # Sum per target, per objective costs.
            cost_means[target_class.name]['cov'] += cov_cost
            cost_means[target_class.name]['bbox'] += bbox_cost
            # Accumulate total cost.
            cov_cost *= objective_weights[target_class.name]['cov']
            bbox_cost *= objective_weights[target_class.name]['bbox']
            total_cost += target_class_weights[target_class.name] * (cov_cost + bbox_cost)

        return total_cost

    def _compute_expected_updated_weights(self, cost_function_config, current_weights,
                                          cost_means):
        updated_weights = []
        for target_class in cost_function_config.target_classes:
            # Compute objective weights and their sum.
            weights = {}
            sum_weights = 0.
            for objective in target_class.objectives:
                o = cost_means[target_class.name][objective.name]
                if o != 0.:
                    o = objective.weight_target / o
                weights[objective.name] = o
                sum_weights += o

            # Compute weight normalizer.
            nrm = 1.0 / sum_weights

            # Update objective weights.
            for objective in target_class.objectives:
                # If coverage cost is 0, setting objective weight does not make sense ->
                # retain old value.
                if sum_weights == 0.:
                    w = current_weights[target_class.name][objective.name]
                else:
                    w = max(weights[objective.name] * nrm,
                            cost_function_config.min_objective_weight)
                    w = min(weights[objective.name] * nrm,
                            cost_function_config.max_objective_weight)
                updated_weights.append(w)

        return updated_weights

    def _get_model_config(self):
        """Get a valid model config."""
        model_config = ModelConfig()
        model_config.num_layers = 18
        model_config.objective_set.bbox.scale = 35.
        model_config.objective_set.bbox.offset = 0.5
        model_config.objective_set.cov.MergeFrom(ModelConfig.CovObjective())
        return model_config

    def _get_cost_function_config(self):
        """Get a valid cost function config."""
        cost_function_config = CostFunctionConfig()
        cost_function_config.enable_autoweighting = True
        cost_function_config.max_objective_weight = 0.9999
        cost_function_config.min_objective_weight = 0.0001
        return cost_function_config

    def test_cost_function(self):
        """Test cost function."""
        config = VisualizerConfig()
        config.enabled = False
        Visualizer.build_from_config(config)

        model_config = self._get_model_config()

        cost_function_config = self._get_cost_function_config()

        # Add 'cat' class.
        cat = cost_function_config.target_classes.add()
        cat.name = 'cat'
        cat.class_weight = 1.
        cat.coverage_foreground_weight = .9
        # Add 'cov' objective for 'cat'.
        cat_cov = cat.objectives.add()
        cat_cov.name = 'cov'
        cat_cov.initial_weight = 4.
        cat_cov.weight_target = 2.
        # Add 'bbox' objective for 'cat'.
        cat_bbox = cat.objectives.add()
        cat_bbox.name = 'bbox'
        cat_bbox.initial_weight = 1.
        cat_bbox.weight_target = 1.

        # Add 'dog' class.
        dog = cost_function_config.target_classes.add()
        dog.name = 'dog'
        dog.class_weight = 3.0
        dog.coverage_foreground_weight = 0.5
        # Add 'cov' objective for 'dog'.
        dog_cov = dog.objectives.add()
        dog_cov.name = 'cov'
        dog_cov.initial_weight = 1.
        dog_cov.weight_target = 1.
        # Add 'bbox' objective for 'dog'.
        dog_bbox = dog.objectives.add()
        dog_bbox.name = 'bbox'
        dog_bbox.initial_weight = 3.
        dog_bbox.weight_target = 3.

        # Expected initial weight values after normalization.
        expected_target_class_weights = {'cat': 0.25, 'dog': 0.75}
        expected_objective_weights = {'cat': {'cov': 0.8, 'bbox': 0.2},
                                      'dog': {'cov': 0.25, 'bbox': 0.75}}

        batch_size = 1
        num_classes = len(cost_function_config.target_classes)
        grid_height = 4
        grid_width = 4
        num_epochs = 5
        num_batches_per_epoch = 2

        cost_auto_weight_hook = build_cost_auto_weight_hook(cost_function_config,
                                                            num_batches_per_epoch)

        input_height, input_width = 16, 16
        output_height, output_width = 1, 1
        objective_set = build_objective_set(model_config.objective_set, output_height, output_width,
                                            input_height, input_width)

        target_classes = build_target_class_list(cost_function_config)

        # Construct ground truth tensor.
        target_dict = {}
        for target_class in cost_function_config.target_classes:
            target_dict[target_class.name] = {}
            for objective in objective_set.objectives:
                target = np.zeros((batch_size, objective.num_channels, grid_height, grid_width),
                                  dtype=np.float32)
                target[:, :, 1:3, 1:3] = 1.0
                target_dict[target_class.name][objective.name] = target

        # Construct prediction tensors.
        cov_pred = np.zeros((batch_size, num_classes, 1, grid_height, grid_width),
                            dtype=np.float32)
        bbox_pred = np.zeros((batch_size, num_classes, 4, grid_height, grid_width),
                             dtype=np.float32)

        with tf.Session() as session:
            cost_auto_weight_hook.after_create_session(session, coord=None)

            session.run(tf.global_variables_initializer())

            # Emulate a few epochs of training.
            for epoch in range(num_epochs):
                # Begin epoch. This clears auto weighting related variables.
                cost_auto_weight_hook.before_run(run_context=None)
                cost_sums = cost_auto_weight_hook._before_run_values
                # Check that cost_sums are all zeroes.
                np.testing.assert_equal(cost_sums, [0., 0., 0., 0.])

                # Emulate a few minibatches.
                expected_cost_means = {}
                for target_class in cost_function_config.target_classes:
                    expected_cost_means[target_class.name] = {'cov': 0., 'bbox': 0.}

                for batch in range(num_batches_per_epoch):
                    # Emulate network learning: Predictions close in on targets on every iteration.
                    v = float(epoch*num_batches_per_epoch+batch) / \
                        float(num_epochs*num_batches_per_epoch-1)
                    cov_pred[:, :, :, 1:3, 1:3] = v
                    bbox_pred[:, :, :, 1:3, 1:3] = 11.0 - 10.0 * v

                    predictions_dict = get_class_predictions(
                        {'cov': cov_pred, 'bbox': bbox_pred},
                        [t.name for t in cost_function_config.target_classes])

                    # Compute minibatch cost. Accumulates objective costs.
                    def cost_func(y_true, y_pred):
                        component_costs = objective_set.compute_component_costs(y_true, y_pred,
                                                                                target_classes)
                        return cost_auto_weight_hook.cost_combiner_func(component_costs)
                    total_cost = session.run(cost_func(target_dict, predictions_dict))
                    # Check that total cost matches expectation.
                    expected_total_cost = \
                        self._compute_expected_total_cost(cost_function_config,
                                                          target_dict, predictions_dict,
                                                          expected_target_class_weights,
                                                          expected_objective_weights,
                                                          expected_cost_means)
                    if verbose:
                        print("epoch %s batch %s total_cost: computed %s expected %s" %
                              (epoch, batch, total_cost, expected_total_cost))
                    np.testing.assert_almost_equal(total_cost, expected_total_cost)

                    # End batch. This computes updated objective weights at the end of an epoch.
                    cost_auto_weight_hook.after_run(run_context=None, run_values=None)
                    updated_weights = cost_auto_weight_hook._after_run_values

                # Check that updated objective weights match expectation.
                expected_updated_weights = \
                    self._compute_expected_updated_weights(cost_function_config,
                                                           expected_objective_weights,
                                                           expected_cost_means)
                if verbose:
                    print("epoch %s updated weights: computed %s expected %s" %
                          (epoch, updated_weights, expected_updated_weights))
                np.testing.assert_almost_equal(updated_weights, expected_updated_weights)

                # Update weights for the next epoch
                expected_objective_weights = {'cat': {'cov': expected_updated_weights[0],
                                                      'bbox': expected_updated_weights[1]},
                                              'dog': {'cov': expected_updated_weights[2],
                                                      'bbox': expected_updated_weights[3]}}


class TestCostFunctionParameters:
    """Test cost function parameters."""

    def test_build_target_class_list(self):
        """Test cost function config parsing."""
        config = CostFunctionConfig()

        # Default values should generate an empty list.
        ret = build_target_class_list(config)
        assert ret == []

        # Add a class and an objective, but forget to set objective's weight_target.
        c = config.target_classes.add()
        c.name = "cat"
        c.class_weight = 0.5
        c.coverage_foreground_weight = 0.75
        o = c.objectives.add()
        o.name = "mouse"
        o.initial_weight = 0.25
        with pytest.raises(ValueError):
            # o.weight_target is not set, so it will default to 0, which is an illegal value.
            build_target_class_list(config)

        # This config should pass.
        o.weight_target = 1.0
        ret = build_target_class_list(config)
        assert len(ret) == 1
        assert ret[0].name == "cat"
        assert ret[0].class_weight == 0.5
        assert ret[0].coverage_foreground_weight == 0.75
        assert len(ret[0].objectives) == 1
        assert ret[0].objectives[0].initial_weight == 0.25
        assert ret[0].objectives[0].weight_target == 1.0

        # Add a second objective but forget to set its name.
        o2 = c.objectives.add()
        o2.initial_weight = 0.25
        o2.weight_target = 1.0
        with pytest.raises(ValueError):
            # o2.name is not set, so it will default to 0, which is an illegal value.
            build_target_class_list(config)

        # Fix the problem and check that the result is ok.
        o2.name = "bird"
        ret = build_target_class_list(config)
        assert len(ret[0].objectives) == 2

    def test_get_target_class_names(self):
        """Test cost function config parsing."""
        config = CostFunctionConfig()
        ret = get_target_class_names(config)
        assert not ret

        c = config.target_classes.add()
        c.name = "cat"
        ret = get_target_class_names(config)
        assert len(ret) == 1
        assert ret[0] == "cat"

        c2 = config.target_classes.add()
        c2.name = "dog"
        ret = get_target_class_names(config)
        assert len(ret) == 2
        assert ret[0] == "cat"
        assert ret[1] == "dog"


def test_build_cost_auto_weight_hook():
    """Test CostAutoWeightHook creation."""
    config = CostFunctionConfig()

    # Default values should not pass.
    with pytest.raises(ValueError):
        build_cost_auto_weight_hook(config, 1)

    # Add a class.
    c = config.target_classes.add()
    c.name = "cat"
    c.class_weight = 0.5
    c.coverage_foreground_weight = 0.75

    # No objectives should not pass.
    with pytest.raises(ValueError):
        build_cost_auto_weight_hook(config, 1)

    # Add an objective.
    o = c.objectives.add()
    o.name = "mouse"
    o.initial_weight = 0.25
    o.weight_target = 1.0

    # A valid config should pass.
    ret = build_cost_auto_weight_hook(config, 1)
    assert isinstance(ret, CostAutoWeightHook)

    with pytest.raises(ValueError):
        # steps_per_epoch == 0 should fail.
        build_cost_auto_weight_hook(config, 0)


class TestObjective(object):
    """Test cost_function_parameters.Objective."""

    @pytest.mark.parametrize(
        "name,initial_weight,weight_target",
        [("level_3", -1.1, 0.5),
         ("level_3", 0.5, -1.1),
         ("", 0.5, 0.6),
         (None, 0.1, 0.2)])
    def test_objective_value_error(self, name, initial_weight, weight_target):
        """Test that a ValueError is raised on invalid inputs."""
        with pytest.raises(ValueError):
            Objective(name=name, initial_weight=initial_weight, weight_target=weight_target)


class TestTargetClass(object):
    """Test cost_function_parameters.TargetClass."""

    VALID_OBJECTIVES = [Objective(name="number_1", initial_weight=0.5, weight_target=0.5),
                        Objective(name="number_2", initial_weight=0.2, weight_target=0.1)]

    # The sum of initial_weight is 0.0.
    INVALID_OBJECTIVES_1 = [Objective(name="number_1", initial_weight=0.0, weight_target=1.0),
                            Objective(name="number_2", initial_weight=0.0, weight_target=1.0)]

    # The sum of weight_target is 0.0.
    INVALID_OBJECTIVES_2 = [Objective(name="number_1", initial_weight=1.0, weight_target=0.0),
                            Objective(name="number_2", initial_weight=1.0, weight_target=0.0)]

    @pytest.mark.parametrize(
        "name,class_weight,coverage_foreground_weight,objectives",
        [
         ("", 20., 0.1, VALID_OBJECTIVES),
         (None, 20., 0.1, VALID_OBJECTIVES),
         ("cat", -0.5, 0.75, VALID_OBJECTIVES),
         ("cat", 0.5, -0.5, VALID_OBJECTIVES),
         ("cat", 0.5, 1.1, VALID_OBJECTIVES),
         ("cat", 0.5, 0.5, INVALID_OBJECTIVES_1),
         ("cat", 0.5, 0.5, INVALID_OBJECTIVES_2)
         ])
    def test_target_class_value_error(
            self, name, class_weight,
            coverage_foreground_weight, objectives):
        """Test that TargetClass raises a ValueError on invalid values."""
        with pytest.raises(ValueError):
            TargetClass(
                name=name,
                class_weight=class_weight,
                coverage_foreground_weight=coverage_foreground_weight,
                objectives=objectives)
