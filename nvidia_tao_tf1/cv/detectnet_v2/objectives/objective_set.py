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

"""Objective set class and builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from nvidia_tao_tf1.cv.detectnet_v2.objectives.build_objective import build_objective


def build_objective_set(objective_set_config,
                        output_height, output_width,
                        input_height, input_width):
    """Construct the model output Objectives.

    Args:
        objective_set_config: The ObjectiveSet configuration proto
        output_height, output_width: Shape of the DNN output tensor.
        input_height, input_width: Shape of the DNN input tensor.

    Returns:
        ObjectiveSet
    """
    objective_names = list(
        objective_set_config.DESCRIPTOR.fields_by_name.keys())

    objectives = []
    for objective_name in objective_names:
        if objective_set_config.HasField(objective_name):
            objectives.append(build_objective(objective_name,
                                              output_height,
                                              output_width,
                                              input_height,
                                              input_width,
                                              getattr(objective_set_config, objective_name)))
    assert objectives, "Model config needs to contain at least one objective"

    # Add the normalized coverage objective
    objectives.append(build_objective('cov_norm',
                                      output_height,
                                      output_width,
                                      input_height,
                                      input_width,
                                      None))

    return ObjectiveSet(objectives)


def get_head_input(model, input_layer_name):
    """Get an output tensor from model based on a layer name search string.

    Args:
        model (Keras.Model): Model from where to look for the input tensor.
        input_layer_name (string): Layer name search string. If empty, last
            layer of model is used.

    Returns:
        The unique tensor whose name contains the input name.

    Raises:
        AssertionError: When a unique tensor is not found.
    """
    if input_layer_name:
        input_layers = [l for l in model.layers if input_layer_name in l.name]
        assert len(input_layers) == 1, \
            "Did not find a unique input matching '%s'. Found %s." % \
            (input_layer_name, [l.name for l in input_layers])
        input_tensor = input_layers[0].output
    else:
        # Input layer name was not given, default to last layer of model.
        input_tensor = model.layers[-1].output

    return input_tensor


class ObjectiveSet(object):
    """Class for sets of objectives."""

    def __init__(self, objectives):
        """Constructor.

        Args:
            objectives: (list<Objective>) List of the Objectives.
        """
        self.objectives = objectives

        # Form list of learnable objectives for convenience
        self.learnable_objectives = [o for o in self.objectives if o.learnable]

    def compute_component_costs(self, y_true, y_pred, target_classes, loss_masks=None):
        """Per target class per objective cost function.

        Args:
            y_true: Ground truth images dictionary.
            y_pred: Network predictions dictionary.
            target_classes: A list of TargetClass instances.
            loss_masks (nested dict): [target_class_name][objective_name]. The leaf values are the
                corresponding loss masks (tf.Tensor) for a batch of frames.
        Returns:
            Dictionary of cost components indexed by target class name and objective name.
        """
        # Compute cost for each target class and objective.
        component_costs = {}
        for target_class in target_classes:
            assert target_class.name in y_true
            assert target_class.name in y_pred

            component_costs[target_class.name] = \
                self.get_objective_costs(
                    y_true, y_pred, target_class, loss_masks)

        return component_costs

    def get_objective_costs(self, y_true, y_pred, target_class, loss_masks=None):
        """Cost per objective for a given target class.

        Args:
            y_true: Ground truth tensor dictionary.
            y_pred: Prediction tensor dictionary.
            target_class: (TargetClass) for which to create the cost.
            loss_masks (nested dict): [target_class_name][objective_name]. The leaf values are the
                corresponding loss masks (tf.Tensor) for a batch of frames.

        Returns:
            objective_costs: Dictionary of per objective scalar cost tensors.
        """
        if loss_masks is None:
            loss_masks = dict()
        objective_costs = dict()

        for objective in self.learnable_objectives:
            # TODO(@williamz): Should loss_masks have been pre-populated with 1.0?
            if target_class.name in loss_masks and objective.name in loss_masks[target_class.name]:
                loss_mask = loss_masks[target_class.name][objective.name]
            else:
                loss_mask = 1.0
            objective_cost = objective.cost(y_true[target_class.name],
                                            y_pred[target_class.name],
                                            target_class,
                                            loss_mask=loss_mask)
            objective_costs[objective.name] = objective_cost

        return objective_costs

    def construct_outputs(self, model, num_classes, data_format,
                          kernel_regularizer, bias_regularizer):
        """Construct the output heads for predicting the objectives.

        For every objective, check whether the model already has a matching output.
        In case the output is not found, construct the corresponding DNN head and
        return it. In case a matching output is found in the model, return the existing
        output (pretrained models may already contain the outputs).

        Args:
            model: Model to which the outputs are added.
            num_classes: The number of model target classes.
            data_format: Order of the dimensions. Set to 'channels_first'.
            kernel_regularizer: Keras regularizer to be applied to convolution kernels.
            bias_regularizer: Keras regularizer to be applied to biases.

        Returns:
            outputs: List of output tensors for a set of objectives.
        """
        outputs = []
        for objective in self.learnable_objectives:
            # Check if model already has the output.
            matching_outputs = [
                o for o in model.outputs if objective.name in o.name]
            # We should not find multiple tensors whose name matches a single objective.
            assert len(matching_outputs) < 2, \
                "Ambiguous model output names: %s. Objective name %s." % \
                ([o.name for o in model.outputs], objective.name)
            if matching_outputs:
                output = matching_outputs[0]
            elif objective.template:
                output = objective.dnn_head_from_template(
                    model=model,
                    num_classes=num_classes,
                    data_format=data_format,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer)
            else:
                input_tensor = get_head_input(
                    model, objective.input_layer_name)
                output = objective.dnn_head(num_classes=num_classes,
                                            data_format=data_format,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer)(input_tensor)
            outputs.append(output)

        return outputs

    def predictions_to_absolute(self, predictions):
        """Convert predictions from model output space to the absolute image space.

        Args:
            predictions: Dictionary of model output space predictions of shape
            (num_samples, num_classes, num_channels, output_height, output_width).

        Returns:
            absolute_predictions: Dictionary of predictions tensors in the image space.
            The shape of the tensors remains unchanged.
        """
        absolute_predictions = dict()

        for objective in self.learnable_objectives:
            prediction = predictions[objective.name]
            prediction = objective.predictions_to_absolute(prediction)
            absolute_predictions[objective.name] = prediction

        return absolute_predictions

    def transform_predictions(self, predictions, matrices=None):
        """Transform predictions by applying transformation matrices.

        Args:
            predictions: Dictionary of predictions of shape
            (num_samples, num_classes, num_channels, output_height, output_width).
            matrices: A tensor of 3x3 transformation matrices, shape (num_samples, 3, 3).
            Matrices are applied to the predictions sample-wise.

        Returns:
            transformed_predictions: Dictionary of transformed predictions tensor. The shape
            of the tensors remains unchanged.
        """
        transformed_predictions = dict()

        for objective in self.learnable_objectives:
            prediction = predictions[objective.name]
            prediction = objective.transform_predictions(prediction, matrices)
            transformed_predictions[objective.name] = prediction

        return transformed_predictions

    def generate_ground_truth_tensors(self, bbox_rasterizer, batch_labels):
        """Generate ground truth tensors.

        Args:
            bbox_rasterizer (BboxRasterizer): Instance of the BboxRasterizer class that will handle
                label-to-rasterizer-arg translation and provide the target_gradient() methods with
                the necessary inputs, as well as perform the final call to the SDK's rasterizer.
            batch_labels (list): Each element is a dict of target features (each a tf.Tensor).

        Returns:
            target_tensors (dict): [target_class_name][objective_name] rasterized ground truth
                tensor.
        """
        target_tensors = defaultdict(dict)
        if isinstance(batch_labels, list):
            # Corresponds to old (DefaultDataloader) path.
            # Get necessary info to compute target gradients from based on the labels.
            batch_bbox_rasterizer_input = [
                bbox_rasterizer.get_target_gradient_info(item) for item in batch_labels
            ]
            batch_gradient_info = [item.gradient_info for item in batch_bbox_rasterizer_input]
        else:
            # Implicitly assumes here it is a Bbox2DLabel.
            # Get necessary info to compute target gradients from based on the labels.
            batch_bbox_rasterizer_input = bbox_rasterizer.get_target_gradient_info(batch_labels)
            # Retrieve gradient info.
            batch_gradient_info = batch_bbox_rasterizer_input.gradient_info

        for objective in self.objectives:
            # Now compute the target gradients.
            if isinstance(batch_labels, list):
                batch_gradients = [objective.target_gradient(item) for item in batch_gradient_info]
            else:
                batch_gradients = objective.target_gradient(batch_gradient_info)
            # Call the rasterizer.
            target_tensor = \
                bbox_rasterizer.rasterize_labels(
                    batch_bbox_rasterizer_input=batch_bbox_rasterizer_input,
                    batch_gradients=batch_gradients,
                    num_gradients=objective.num_channels,
                    gradient_flag=objective.gradient_flag)
            # Slice per-class targets out of the rasterized target tensor
            for class_index, target_class_name in enumerate(bbox_rasterizer.target_class_names):
                target_tensors[target_class_name][objective.name] = target_tensor[:, class_index]

        return target_tensors
