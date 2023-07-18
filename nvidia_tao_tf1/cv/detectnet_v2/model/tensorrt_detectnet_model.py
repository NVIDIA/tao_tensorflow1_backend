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

"""A wrapper for a TensorRT (TRT) DriveNet engine.

The engine can be a FP32, FP16, or a calibrated INT8 engine.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from nvidia_tao_tf1.core.export import load_tensorrt_engine
from nvidia_tao_tf1.cv.detectnet_v2.model.detectnet_model import GridboxModel
from nvidia_tao_tf1.cv.detectnet_v2.objectives.objective_set import build_objective_set


def unravel_dimensions(dims):
    """Unravel dimensions to c,h,w."""
    if len(dims) == 3:
        height, width = (dims[1], dims[2])
    elif len(dims) == 4 and dims[0] not in [-1, None]:
        height, width = (dims[2], dims[3])
    else:
        raise NotImplementedError(
            "Unhandled shape: {shape} for dimensions {dims}".format(
                shape=len(dims), dims=dims)
        )
    return height, width


class TensorRTGridboxModel(GridboxModel):
    """A wrapper class for a TensorRT (TRT) DriveNet engine.

    Provides interfaces for evaluation that match those of the GridboxModel class.
    This allows the TRT engine to be used with the standard evaluation and inference scripts.
    """

    def load_model_weights(self, model_file, **kwargs):
        """Load a TensorRT engine for inference.

        Args:
            model_file (str): TensorRT engine filename.
        """
        self._engine = load_tensorrt_engine(model_file)

        for binding in self._engine._engine:
            if self._engine._engine.binding_is_input(binding):
                input_dims = self._engine._engine.get_binding_shape(binding)
            else:
                output_dims = self._engine._engine.get_binding_shape(binding)
        self.max_batch_size = self._engine._engine.max_batch_size
        self.input_height, self.input_width = unravel_dimensions(input_dims)
        self.output_height, self.output_width = unravel_dimensions(output_dims)
        self.num_output_classes = len(self.target_class_names)

        self.objective_set = build_objective_set(self.objective_set_config,
                                                 self.output_height,
                                                 self.output_width,
                                                 self.input_height,
                                                 self.input_width)

        self.constructed = True
        self._prediction_placeholders = None
        self.ground_truth_placeholders = None

    def load_model(self, *args, **kwargs):
        """Not implemented."""
        raise NotImplementedError("Loading TensorRT engine with metadata is not implemented.")

    def build_training_graph(self, *args, **kwargs):
        """Not implemented."""
        raise NotImplementedError("Training a TensorRT engine is not implemented.")

    @property
    def num_params(self):
        """Get number of parameters from TensorRT evaluate."""
        # TODO: @vpraveen Need to figure out how to get the number of
        # params from a TensorRT engine if at all possible.
        return 0

    def _get_prediction_placeholders(self, inputs):
        """Create placeholders for the prediction tensors.

        Args
            inputs: Dataset input tensors to be used for validation.

        Returns:
            predictions (list): Each element is a tf.placeholder of the correct dtype and shape.
        """
        predictions = []

        batch_size = int(inputs.shape[0])

        # Create a placeholder for each output of the TensorRT engine.
        for objective in self.objective_set.learnable_objectives:
            shape = [batch_size, self.num_output_classes, objective.num_channels,
                     self.output_height, self.output_width]
            predictions.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=shape,
                               name='output_'+objective.name))

        return predictions

    def build_validation_graph(self, inputs, ground_truth_tensors,
                               target_classes, cost_combiner_func):
        """Set up the TensorRT engine for validation.

        Args:
            inputs: Dataset input tensors to be used for validation.
            ground_truth_tensors (dict): [target_class_name][objective_name] -> tf.Tensor.
            target_classes (list): A list of TargetClass instances.
            cost_combiner_func: A function that takes in a dictionary of objective costs,
                and total cost by computing a weighted sum of the objective costs.
        """
        # Predictions are done outside the TensorFlow graph using the TensorRT engine. For this
        # reason, replace prediction and ground truth tensors with placeholders. They will be
        # fed after doing inference using the engine. Otherwise, construct the validation graph
        # normally using the GridboxModel code.
        self._prediction_placeholders = self._get_prediction_placeholders(inputs)

        # Replace model predictions with the prediction placeholders.
        self.keras_model = lambda x: self._prediction_placeholders

        # Build the validation graph using the code from GridboxModel.
        super(TensorRTGridboxModel, self).build_validation_graph(
            inputs,
            ground_truth_tensors,
            target_classes,
            cost_combiner_func)

    def print_model_summary(self):
        """Print a summary of the TensorRT engine."""
        outputs = [o.name for o in self.objective_set.learnable_objectives]
        print('Outputs of the TensorRT engine are:', outputs)

    def prune(self, *args, **kwargs):
        """Not implemented."""
        raise NotImplementedError("Pruning a TensorRT engine is not implemented.")

    def _reshape_trt_predictions(self, predictions):
        """TRT flattens the prediction arrays, reshape to the correct shape.

        Args:
            predictions: Dictionary of numpy arrays containing TRT inference results. Keys are
            output names of the Caffe/UFF model and values are the prediction numpy arrays.

        Returns:
            reshaped_predictions: A list of reshaped numpy arrays.
        """
        reshaped_predictions = []

        for objective in self.objective_set.learnable_objectives:
            # Map TRT output names to output names used in the graph.
            trt_output = objective.name.replace('output_', '')
            prediction = next(value
                              for key, value in six.iteritems(predictions) if trt_output in key)

            batch_size = np.shape(prediction)[0]

            expected_shape = [batch_size, self.num_output_classes, objective.num_channels,
                              self.output_height, self.output_width]

            reshaped_predictions.append(prediction.reshape(expected_shape))

        return reshaped_predictions

    def get_predictions_feed_dict(self, images):
        """Get a prediction tensors dictionary for validating/evaluating the model."""
        # Get one batch of images and ground truths for validation.
        predictions = self._engine.infer(images)

        # Reshape TRT predictions to the shape expected by evaluation code.
        predictions = self._reshape_trt_predictions(predictions)

        # Match numeric predictions and ground truth labels with the corresponding tensors to
        # run the validation graph.
        predictions_feed_dict = dict()

        for tensor, prediction in zip(self._prediction_placeholders, predictions):
            predictions_feed_dict[tensor] = prediction

        return predictions_feed_dict

    @staticmethod
    def get_session_config():
        """Retrieve a TensorFlow session config.

        Returns:
            config (tf.compat.v1.ConfigProto): Retrive tensorflow config
                with GPU options set.
        """
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.33
        )
        config = tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            device_count={'GPU': 0, 'CPU': 1}
        )
        return config
