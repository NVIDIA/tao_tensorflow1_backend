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

"""Test DetectNet V2 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras
from keras import backend as K
import numpy as np
import pytest
from six.moves import zip
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_function_parameters import (
    get_target_class_names
)
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import build_model
from nvidia_tao_tf1.cv.detectnet_v2.proto.model_config_pb2 import ModelConfig
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.detectnet_v2.training.training_proto_utilities import (
    build_optimizer,
    build_regularizer,
    build_train_op_generator
)
from nvidia_tao_tf1.cv.detectnet_v2.training.utilities import setup_keras_backend
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer

update_regularizers_test_cases = [
    # (regularization_type, regularization_weight)
    ("L1", 0.25),
    ("L2", 0.5),
    ("NO_REG", 0.99)]


def check_layer_weights(test_layer, ref_layer):
    """Helper for checking the weights of two layers match.

    Args:
        test_layer, ref_layer (Keras.Layer): Layers to be checked.

    Raises:
        AssertionError if the layers weights are not the same.
    """
    test_weights = test_layer.get_weights()
    ref_weights = ref_layer.get_weights()
    assert len(test_weights) == len(ref_weights), \
        "Wrong length of loaded model weights at layer %s." % test_layer
    for test_weight, ref_weight in zip(test_weights, ref_weights):
        np.testing.assert_almost_equal(test_weight, ref_weight)


class TestGridbox:
    """Test DetectNet V2 model methods, e.g. loading and updating regularizers."""

    @pytest.fixture(scope='class')
    def model_file_path(self):
        return 'dummy_model.hdf5'

    @pytest.fixture(scope='class', params=[
        ModelConfig.TrainingPrecision.FLOAT16,
        ModelConfig.TrainingPrecision.FLOAT32])
    def experiment_spec(self, request):
        spec_proto = load_experiment_spec()
        spec_proto.model_config.num_layers = 10
        spec_proto.model_config.training_precision.backend_floatx = request.param
        return spec_proto

    @pytest.fixture(scope='class')
    def gridbox_model(self, experiment_spec):
        K.clear_session()

        # Prepare model initialization parameters.
        width = 224
        height = 224
        shape = (3, width, height)

        # Setup the backend with correct computation precision
        setup_keras_backend(experiment_spec.model_config.training_precision, is_training=True)

        # Set up regularization.
        kernel_regularizer, bias_regularizer = build_regularizer(
                experiment_spec.training_config.regularizer)

        # Construct a gridbox_model.
        target_class_names = get_target_class_names(experiment_spec.cost_function_config)
        gridbox_model = build_model(experiment_spec.model_config, target_class_names)

        gridbox_model.construct_model(shape, kernel_regularizer, bias_regularizer,
                                      pretrained_weights_file=None)

        return gridbox_model

    def test_load_model_weights(self, gridbox_model, experiment_spec, model_file_path):
        """Test loading weights of a saved DetectNet V2 model."""
        gridbox_model.keras_model.save(model_file_path)

        # Load the gridbox_model from file.
        target_class_names = get_target_class_names(experiment_spec.cost_function_config)
        gridbox_model_reloaded = build_model(experiment_spec.model_config, target_class_names)

        gridbox_model_reloaded.load_model_weights(model_file_path)

        # Give gridbox_model a new name for readability.
        gridbox_model_expected = gridbox_model

        assert gridbox_model_expected.num_layers == gridbox_model_reloaded.num_layers
        assert gridbox_model_expected.input_height == gridbox_model_reloaded.input_height
        assert gridbox_model_expected.input_width == gridbox_model_reloaded.input_width
        assert gridbox_model_expected.input_num_channels == \
            gridbox_model_reloaded.input_num_channels
        assert gridbox_model_expected.output_height == gridbox_model_reloaded.output_height
        assert gridbox_model_expected.output_width == gridbox_model_reloaded.output_width

        # Test the loaded models layers weights match the original ones.
        assert len(gridbox_model_expected.keras_model.layers) \
            == len(gridbox_model_reloaded.keras_model.layers)
        for ref_layer in gridbox_model_expected.keras_model.layers:
            test_layer = gridbox_model_reloaded.keras_model.get_layer(name=ref_layer.name)
            check_layer_weights(test_layer, ref_layer)

        # Remove the temporary file.
        os.remove(model_file_path)

    def test_update_regularizer(self, gridbox_model, experiment_spec):
        """Test update regularizer function to override the previous model regs."""
        previous_model = gridbox_model.keras_model
        regularizer_config = experiment_spec.training_config.regularizer
        kernel_regularizer, bias_regularizer = build_regularizer(regularizer_config)
        gridbox_model.update_regularizers(kernel_regularizer, bias_regularizer)
        model_config = gridbox_model.keras_model.get_config()
        for r_layer, l_config in zip(gridbox_model.keras_model.layers, model_config['layers']):
            if r_layer in [keras.layers.convolutional.Conv2D,
                           keras.layers.core.Dense,
                           keras.layers.DepthwiseConv2D]:
                if hasattr(r_layer, 'kernel_regularizer'):
                    assert l_config['config']['kernel_regularizer'] == kernel_regularizer
                if hasattr(r_layer, 'bias_regularizer'):
                    assert l_config['config']['bias_regularizer'] == bias_regularizer
            test_layer = previous_model.get_layer(name=r_layer.name)
            check_layer_weights(test_layer, r_layer)

    def test_model_regularizers(self, gridbox_model, experiment_spec):
        """Test that the regularizers specified in spec are present in the DetectNet V2 model."""
        # Set up regularization.
        kernel_regularizer, bias_regularizer = build_regularizer(
                experiment_spec.training_config.regularizer)

        for layer in gridbox_model.keras_model.layers:

            if 'kernel_regularizer' in layer.get_config():
                assert layer.kernel_regularizer.l1 == kernel_regularizer.l1
                assert layer.kernel_regularizer.l2 == kernel_regularizer.l2

            if 'bias_regularizer' in layer.get_config():
                assert layer.bias_regularizer.l1 == bias_regularizer.l1
                assert layer.bias_regularizer.l2 == bias_regularizer.l2

    def test_build_training_graph(self, gridbox_model, experiment_spec, model_file_path, mocker):
        """Test building training graph."""
        # Set up optimizer.
        optimizer = build_optimizer(experiment_spec.training_config.optimizer, 1.0)
        train_op_generator = build_train_op_generator(experiment_spec.training_config.cost_scaling)

        # Build the Visualizer.
        visualizer_config = experiment_spec.training_config.visualizer
        visualizer_config.enabled = True
        Visualizer.build_from_config(visualizer_config)

        # Build training graph.
        K.set_learning_phase(0)
        dummy_input = tf.zeros((1, 3, gridbox_model.input_height, gridbox_model.input_width),
                               dtype=K.floatx())
        # Mock gradient computation to be able to check how it was called.
        wrapped = optimizer.compute_gradients
        compute_gradients_mock = mocker.patch.object(tf.train.AdamOptimizer, 'compute_gradients',
                                                     wraps=wrapped)
        # Set one layer as non-trainable to check optimizer behavior.
        gridbox_model.keras_model.layers[1].trainable = False

        gridbox_model.build_training_graph(inputs=dummy_input, ground_truth_tensors=None,
                                           optimizer=optimizer, target_classes=[],
                                           cost_combiner_func=lambda *args: 0.0,
                                           train_op_generator=train_op_generator)

        # Check that optimizer is called with correct variables.
        var_list = []
        for call_arg in compute_gradients_mock.call_args:
            if 'var_list' in call_arg:
                var_list = call_arg['var_list']

        for layer in gridbox_model.keras_model.layers:
            for weight in layer.weights:
                # All trainable layers params should be mentioned, except for BN moving averages.
                if layer.trainable and 'moving_' not in weight.name:
                    assert weight in var_list, "Trainable weight was not passed to optimizer."
                else:
                    assert weight not in var_list, "Non-trainable weight was passed to optimizer."

    def test_objective_names(self, gridbox_model):
        """Test objective_names."""
        assert gridbox_model.objective_names == set(['bbox', 'cov'])

    def test_output_layer_names(self, gridbox_model):
        """Test output_layer_names."""
        assert gridbox_model.output_layer_names == ['output_bbox', 'output_cov']
