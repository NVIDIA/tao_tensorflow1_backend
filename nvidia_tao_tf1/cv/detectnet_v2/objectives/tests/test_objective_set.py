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

"""Test ObjectiveSet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import numpy as np
import pytest
from six.moves import zip
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.label_filter.test_label_filter import get_dummy_labels
from nvidia_tao_tf1.cv.detectnet_v2.objectives.objective_set import build_objective_set
from nvidia_tao_tf1.cv.detectnet_v2.proto.cost_function_config_pb2 import CostFunctionConfig
from nvidia_tao_tf1.cv.detectnet_v2.proto.model_config_pb2 import ModelConfig
from nvidia_tao_tf1.cv.detectnet_v2.proto.visualizer_config_pb2 import VisualizerConfig
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer import BboxRasterizer
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer_config import BboxRasterizerConfig
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer


@pytest.fixture(scope="module")
def objective_set():
    """Build ObjectiveSet."""
    objective_set = ModelConfig.ObjectiveSet()
    objective_set.bbox.input = "dropout"
    objective_set.bbox.scale = 1
    objective_set.bbox.offset = 1
    objective_set.cov.MergeFrom(ModelConfig.CovObjective())
    objective_set.cov.input = "dropout"

    input_height, input_width = 16, 16
    output_height, output_width = 1, 1

    visualizer_config = VisualizerConfig()
    visualizer_config.enabled = False
    Visualizer.build_from_config(visualizer_config)

    objective_set = build_objective_set(objective_set, output_height, output_width,
                                        input_height, input_width)

    return objective_set


@pytest.fixture(scope="module")
def bbox_rasterizer():
    """Define a BboxRasterizer to use for the tests."""
    bbox_rasterizer_config = BboxRasterizerConfig(deadzone_radius=0.5)
    bbox_rasterizer_config['car'] = BboxRasterizerConfig.TargetClassConfig(
        cov_center_x=0.5, cov_center_y=0.5, cov_radius_x=0.8, cov_radius_y=0.7, bbox_min_radius=0.5
    )
    bbox_rasterizer_config['person'] = BboxRasterizerConfig.TargetClassConfig(
        cov_center_x=0.5, cov_center_y=0.5, cov_radius_x=0.8, cov_radius_y=0.7, bbox_min_radius=0.5
    )
    bbox_rasterizer = BboxRasterizer(
        input_width=16, input_height=16, output_height=1, output_width=1,
        target_class_names=['car', 'person'], bbox_rasterizer_config=bbox_rasterizer_config,
        target_class_mapping={'pedestrian': 'person', 'automobile': 'car'})

    return bbox_rasterizer


@pytest.fixture()
def dummy_predictions(objective_set):
    """Return dict in which keys are objective names and values valid but dummy predictions."""
    output_dims = [
        objective.num_channels for objective in objective_set.learnable_objectives]
    predictions = {o.name: tf.ones((1, 1, dims, 1, 1)) for o, dims in
                   zip(objective_set.learnable_objectives, output_dims)}

    return predictions


def test_build_objective_set(objective_set):
    """Test building an ObjectiveSet."""
    objective_names = {
        objective.name for objective in objective_set.objectives}
    learnable_objective_names = {objective.name for objective in
                                 objective_set.learnable_objectives}

    expected = set(["cov", "bbox"])

    assert learnable_objective_names == expected
    expected.add("cov_norm")

    assert objective_names == expected


def test_get_objective_costs(objective_set):
    """Test computing cost per objective for an ObjectiveSet."""
    y_true = {'car': {objective.name: tf.ones(
        (1, 1)) for objective in objective_set.objectives}}
    # Prediction equals the ground truth. Expected cost is zero.
    y_pred = y_true
    target_class = CostFunctionConfig.TargetClass()
    target_class.name = 'car'

    objective_costs = objective_set.get_objective_costs(
        y_true, y_pred, target_class)

    with tf.Session() as session:
        objective_costs = session.run(objective_costs)

        for objective in objective_set.learnable_objectives:
            assert objective_costs[objective.name] == 0.


def test_construct_outputs(objective_set):
    """Check that outputs are created for each objective and that they are in expected order."""
    inputs = keras.layers.Input(shape=(2, 1, 1))
    outputs = keras.layers.Dropout(0.0)(inputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    kernel_regularizer = bias_regularizer = keras.regularizers.l1(1.)
    num_classes = 3
    outputs = objective_set.construct_outputs(model,
                                              num_classes=num_classes,
                                              data_format='channels_first',
                                              kernel_regularizer=kernel_regularizer,
                                              bias_regularizer=bias_regularizer)

    assert len(outputs) == len(objective_set.learnable_objectives)
    expected_output_dims = [4, 1, 1]

    for objective, output, output_dims in zip(objective_set.learnable_objectives, outputs,
                                              expected_output_dims):
        assert objective.name in output.name
        assert keras.backend.int_shape(output) == (
            None, num_classes * output_dims, 1, 1)


def test_construct_outputs_no_matching_input(objective_set):
    """Check that constructing output fails if there is no matching input is found."""
    inputs = keras.layers.Input(shape=(2, 1, 1))
    outputs = keras.layers.Dropout(0.0, name='ropout_fail')(inputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    with pytest.raises(AssertionError):
        outputs = objective_set.construct_outputs(model,
                                                  num_classes=1,
                                                  data_format='channels_first',
                                                  kernel_regularizer=None,
                                                  bias_regularizer=None)


def test_construct_outputs_multiple_matching_inputs(objective_set):
    """Check that constructing output fails if there are multiple matching inputs."""
    inputs = keras.layers.Input(shape=(2, 1, 1))
    outputs = keras.layers.Dropout(0.0, name='dropout_1')(inputs)
    outputs = keras.layers.Dropout(0.0, name='dropout_2')(outputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    with pytest.raises(AssertionError):
        outputs = objective_set.construct_outputs(model,
                                                  num_classes=1,
                                                  data_format='channels_first',
                                                  kernel_regularizer=None,
                                                  bias_regularizer=None)


def test_construct_outputs_default_input(objective_set):
    """Check that constructing output is OK when inputs are not specified."""
    inputs = keras.layers.Input(shape=(2, 1, 1))
    outputs = keras.layers.Dropout(0.0, name='dropout_1')(inputs)
    outputs = keras.layers.Dropout(0.0, name='dropout_2')(outputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    for objective in objective_set.learnable_objectives:
        objective.input_layer_name = ''
    outputs = objective_set.construct_outputs(model,
                                              num_classes=1,
                                              data_format='channels_first',
                                              kernel_regularizer=None,
                                              bias_regularizer=None)


def _check_transformed_predictions(objective_set, original_predictions, transformed_predictions,
                                   transformation, additional_inputs):
    """Compare transformed predictions to the result of applying a given transformation in place."""
    with tf.Session() as session:
        for objective in objective_set.learnable_objectives:
            original_prediction = original_predictions[objective.name]
            inputs = [original_prediction] + additional_inputs
            expected = session.run(getattr(objective, transformation)(*inputs))

            transformed_prediction = session.run(
                transformed_predictions[objective.name])

            np.testing.assert_allclose(transformed_prediction, expected)


def test_predictions_to_absolute(objective_set, dummy_predictions):
    """Test converting all objectives to the absolute coordinate space."""
    absolute_predictions = objective_set.predictions_to_absolute(
        dummy_predictions)

    _check_transformed_predictions(objective_set, dummy_predictions, absolute_predictions,
                                   'predictions_to_absolute', [])


def test_transform_predictions(objective_set, dummy_predictions):
    """Test transforming all predictions."""
    num_samples = [int(list(dummy_predictions.values())[0].shape[0])]
    matrices = tf.eye(3, batch_shape=num_samples)

    transformed_predictions = objective_set.transform_predictions(
        dummy_predictions, matrices)

    _check_transformed_predictions(objective_set, dummy_predictions, transformed_predictions,
                                   'transform_predictions', [matrices])


def test_generate_ground_truth_tensors(objective_set, bbox_rasterizer):
    """Test that generate_ground_truth_tensors sets up the correct tensors.

    Note: this does not test the result of running the rasterization op.

    Args:
        objective_set (ObjectiveSet): As defined by the module-wide fixture.
        bbox_rasterizer (BBoxRasterizer): As defined by the module-wide fixture.
    """
    batch_source_class_names = [['car', 'person', 'car'], ['person']]
    batch_other_attributes = [dict(), dict()]
    # First, define bbox coords for each frame.
    batch_other_attributes[0]['target/bbox_coordinates'] = \
        np.cast[np.float32](np.random.randn(3, 4))
    batch_other_attributes[1]['target/bbox_coordinates'] = \
        np.cast[np.float32](np.random.randn(1, 4))
    # Then, add depth related field.
    batch_other_attributes[0]['target/world_bbox_z'] = \
        np.array([12.3, 4.56, 7.89], dtype=np.float32)
    batch_other_attributes[1]['target/world_bbox_z'] = np.array(
        [10.11], dtype=np.float32)
    # Same with orientation.
    batch_other_attributes[0]['target/orientation'] = np.array(
        [0.1, 0.2, 0.3], dtype=np.float32)
    batch_other_attributes[1]['target/orientation'] = np.array(
        [0.4], dtype=np.float32)
    # Add augmentation matrices.
    batch_other_attributes[0]['frame/augmented_to_input_matrices'] = \
        np.cast[np.float32](np.random.randn(3, 3))
    batch_other_attributes[1]['frame/augmented_to_input_matrices'] = \
        np.cast[np.float32](np.random.randn(3, 3))
    # Add first order bw-poly coefficients.
    batch_other_attributes[0]['frame/bw_poly_coeff1'] = \
        np.array([0.0005], dtype=np.float32)
    batch_other_attributes[1]['frame/bw_poly_coeff1'] = \
        np.array([0.001], dtype=np.float32)
    batch_labels = \
        [get_dummy_labels(x[0], other_attributes=x[1]) for x in zip(
            batch_source_class_names, batch_other_attributes)]
    target_tensors = objective_set.generate_ground_truth_tensors(bbox_rasterizer=bbox_rasterizer,
                                                                 batch_labels=batch_labels)
    target_class_names = {'car', 'person'}
    assert set(target_tensors.keys()) == target_class_names
    for target_class_name in target_class_names:
        assert set(target_tensors[target_class_name].keys()) == \
            {'cov', 'bbox', 'cov_norm'}
