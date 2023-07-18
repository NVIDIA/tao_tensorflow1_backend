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

"""Test convolutional gated recurrent unit (GRU) custom Keras layer TRT export"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import keras

import numpy as np
from nvidia_tao_tf1.core.models.templates.conv_gru_2d_export import ConvGRU2DExport
from parameterized import parameterized
import pytest
import tensorflow as tf


# cases for is_stateful, sequence_length_in_frames
test_cases_1 = [[True, 1], [False, 2]]

# cases for is_stateful, sequence_length_in_frames, state_scaling
state_scaling_cases = [0.9, 1.0]
test_cases_2 = [
    test_cases_1[i] + [state_scaling_cases[i]] for i in range(len(test_cases_1))
]

# cases for is_stateful, sequence_length_in_frames, state_scaling, expected output
expected_output_cases = [0.99482181, 0.98481372]
test_cases_3 = [
    test_cases_2[i] + [expected_output_cases[i]] for i in range(len(test_cases_2))
]


class TestConvGRU2DExport(tf.test.TestCase):
    """Test convolutional gated recurrent unit (GRU) custom Keras TRT-export layer."""

    def setUp(self):
        """Set up the test fixture: construct a ConvGRU2DExport object."""
        super(TestConvGRU2DExport, self).setUp()

        # Shape: [channels, height, width]
        self.GRU_INPUT_SHAPE = [None, 3, 1, 2]

        # Set the initial state shape.
        # Shape: [number_channels in a state, height, width]
        self.INITIAL_STATE_SHAPE = [None, 4, 1, 2]
        self.SPATIAL_KERNEL_HEIGHT = 1
        self.SPATIAL_KERNEL_WIDTH = 1
        self.KERNEL_REGULARIZER = {
            "class_name": "L1L2",
            "config": {"l2": 0.1, "l1": 0.3},
        }
        self.BIAS_REGULARIZER = {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.1}}

    @parameterized.expand(test_cases_2)
    def test_convgru2d_export(
        self, is_stateful, sequence_length_in_frames, state_scaling
    ):
        """Test if convgru2dexport layer parameters are initialized as expected values."""
        convgru2d = ConvGRU2DExport(
            model_sequence_length_in_frames=sequence_length_in_frames,
            input_sequence_length_in_frames=sequence_length_in_frames,
            is_stateful=is_stateful,
            state_scaling=state_scaling,
            input_shape=self.GRU_INPUT_SHAPE,
            initial_state_shape=self.INITIAL_STATE_SHAPE,
            spatial_kernel_height=self.SPATIAL_KERNEL_HEIGHT,
            spatial_kernel_width=self.SPATIAL_KERNEL_WIDTH,
            kernel_regularizer=self.KERNEL_REGULARIZER,
            bias_regularizer=self.BIAS_REGULARIZER,
        )

        input_shape_expected = self.GRU_INPUT_SHAPE

        model_sequence_length_in_frames_expected = sequence_length_in_frames
        input_sequence_length_in_frames_expected = sequence_length_in_frames
        state_scaling_expected = state_scaling
        is_stateful_expected = is_stateful

        # Set the initial state shape.
        # Shape: [number_channels in a state, height, width].
        initial_state_shape_expected = self.INITIAL_STATE_SHAPE
        spatial_kernel_height_expected = self.SPATIAL_KERNEL_HEIGHT
        spatial_kernel_width_expected = self.SPATIAL_KERNEL_WIDTH

        kernel_regularizer = self.KERNEL_REGULARIZER
        bias_regularizer = self.BIAS_REGULARIZER

        assert input_shape_expected == convgru2d.rnn_input_shape
        # Test intitial state shape except the batch dimension.
        assert initial_state_shape_expected[1:] == convgru2d.initial_state_shape[1:]
        assert spatial_kernel_height_expected == convgru2d.spatial_kernel_height
        assert spatial_kernel_width_expected == convgru2d.spatial_kernel_width
        assert (
            model_sequence_length_in_frames_expected
            == convgru2d.model_sequence_length_in_frames
        )
        assert (
            input_sequence_length_in_frames_expected
            == convgru2d.input_sequence_length_in_frames
        )
        assert np.isclose(state_scaling_expected, convgru2d.state_scaling)
        assert is_stateful_expected == convgru2d.is_stateful
        assert kernel_regularizer["config"]["l1"] == convgru2d.kernel_regularizer.l1
        assert kernel_regularizer["config"]["l2"] == convgru2d.kernel_regularizer.l2
        assert bias_regularizer["config"]["l1"] == convgru2d.bias_regularizer.l1
        assert bias_regularizer["config"]["l2"] == convgru2d.bias_regularizer.l2

    @parameterized.expand(test_cases_2)
    def test_convgru2dexport_variables(
        self, is_stateful, sequence_length_in_frames, state_scaling
    ):
        """Test the trainable variables of the ConvGRU2DExport Layer.

        1) Test if the trainable variables are built with the correct shapes and names.
        2) Test if the number of trainable variables are correct.
        3) Test if the output shape is set correctly while upon calling the layer.
        """
        convgru2d = ConvGRU2DExport(
            model_sequence_length_in_frames=sequence_length_in_frames,
            input_sequence_length_in_frames=sequence_length_in_frames,
            is_stateful=is_stateful,
            state_scaling=state_scaling,
            input_shape=self.GRU_INPUT_SHAPE,
            initial_state_shape=self.INITIAL_STATE_SHAPE,
            spatial_kernel_height=self.SPATIAL_KERNEL_HEIGHT,
            spatial_kernel_width=self.SPATIAL_KERNEL_WIDTH,
            kernel_regularizer=self.KERNEL_REGULARIZER,
            bias_regularizer=self.BIAS_REGULARIZER,
        )

        variable_names_expected = set(
            [
                "conv_gru_2d_export/W_z:0",
                "conv_gru_2d_export/W_r:0",
                "conv_gru_2d_export/W_h:0",
                "conv_gru_2d_export/U_z:0",
                "conv_gru_2d_export/U_r:0",
                "conv_gru_2d_export/U_h:0",
                "conv_gru_2d_export/b_z:0",
                "conv_gru_2d_export/b_r:0",
                "conv_gru_2d_export/b_h:0",
            ]
        )

        inputs = keras.Input(
            shape=(
                self.GRU_INPUT_SHAPE[1],
                self.GRU_INPUT_SHAPE[2],
                self.GRU_INPUT_SHAPE[3],
            )
        )
        output = convgru2d(inputs)

        # Check variable names.
        weights_actual = convgru2d.weights
        variable_names_actual = {weight.name for weight in weights_actual}

        assert variable_names_expected == variable_names_actual

        # Check the number of trainable variables. This should be 9: 3 x U_, 3 X W_ and 3 X bias.
        assert len(convgru2d.trainable_weights) == 9

        # Check the shapes of input to state projections.
        # Shape [height, width, in_channels, out_channels].
        var_shape_expected = [
            self.SPATIAL_KERNEL_HEIGHT,
            self.SPATIAL_KERNEL_WIDTH,
            self.GRU_INPUT_SHAPE[1],
            self.INITIAL_STATE_SHAPE[1],
        ]

        for local_var_name in ["W_z", "W_r", "W_h"]:
            var_shape_actual = getattr(convgru2d, local_var_name).shape.as_list()
            assert var_shape_expected == var_shape_actual

        # Check the shapes of state to state projections.
        # Shape [height, width, in_channels, out_channels].
        var_shape_expected = [
            self.SPATIAL_KERNEL_HEIGHT,
            self.SPATIAL_KERNEL_WIDTH,
            self.INITIAL_STATE_SHAPE[1],
            self.INITIAL_STATE_SHAPE[1],
        ]

        for local_var_name in ["U_z", "U_r", "U_h"]:
            var_shape_actual = getattr(convgru2d, local_var_name).shape.as_list()
            assert var_shape_expected == var_shape_actual

        # Check the shapes of bias variables.
        # Shape [out_channels].
        var_shape_expected = [self.INITIAL_STATE_SHAPE[1]]

        for local_var_name in ["b_z", "b_r", "b_h"]:
            var_shape_actual = getattr(convgru2d, local_var_name).shape.as_list()
            assert var_shape_expected == var_shape_actual

        # Check the output shape. [batch_size, num_filters, grid_height, grid_width].
        output_shape_expected = [None] + list(self.INITIAL_STATE_SHAPE[1:])
        assert output_shape_expected == output.shape.as_list()

    @parameterized.expand(test_cases_3)
    def test_convgru2d_output_values_single_step(
        self, is_stateful, sequence_length_in_frames, state_scaling, expected_output
    ):
        """Test the value of the output tensor after calling a single step ConvGRU2DExport Layer."""
        # Set test case variables in the model
        self.MODEL_SEQUENCE_LENGTH_IN_FRAMES = sequence_length_in_frames
        self.INPUT_SEQUENCE_LENGTH_IN_FRAMES = sequence_length_in_frames

        with self.test_session() as sess:
            # A close-to-minimum sized GRU export.
            # This GRU will implement 1x1 operations on 2x2 grid. in-channels:1, out-channels:1 .
            convgru2d_minimal = ConvGRU2DExport(
                model_sequence_length_in_frames=self.MODEL_SEQUENCE_LENGTH_IN_FRAMES,
                input_sequence_length_in_frames=self.INPUT_SEQUENCE_LENGTH_IN_FRAMES,
                state_scaling=1.0,
                is_stateful=is_stateful,
                input_shape=[None, 1, 2, 2],
                initial_state_shape=[None, 1, 2, 2],
                spatial_kernel_height=1,
                spatial_kernel_width=1,
            )
            inputs = keras.Input(shape=(1, 2, 2))
            convgru2d_minimal(inputs)

            # Manually set the weights to specific values.
            value_for_projections_variables = np.ones([1, 1, 1, 1], dtype=np.float32)
            # Set the bias variable to zero first.
            value_for_bias_variables = np.zeros([1], dtype=np.float32)

            # Weights have 6 projection variables and 3 bias variables.
            convgru2d_minimal.set_weights(
                6 * [value_for_projections_variables] + 3 * [value_for_bias_variables]
            )

            # Present a ones input to the network.
            input_tensor = tf.constant(np.ones([1, 1, 2, 2], dtype=np.float32))

            if is_stateful:
                # First, we keep the state for the previous time step at 0.
                state_input_tensor = np.zeros([1, 1, 2, 2], dtype=np.float32)
                feed_dict = {convgru2d_minimal._initial_state: state_input_tensor}
            else:
                # For stateless model, there is no initial state but there are feature maps
                # from previous frames stored in _past_features
                past_features_tensor = np.zeros([1, 1, 2, 2], dtype=np.float32)
                feed_dict = {}
                for i in range(self.INPUT_SEQUENCE_LENGTH_IN_FRAMES - 1):
                    feed_dict[
                        convgru2d_minimal._past_features[i]
                    ] = past_features_tensor

            # 1) For the case of stateful (i.e. is_stateful==True):
            # The below sub-session should compute the value of
            # the GRU-operations output after time-step.
            # z will be 1x1x2x2 tensor, each element will equal sigmoid(1).
            # r will be 1x1x2x2 tensor, each element will equal sigmoid(1).
            # state_update_input will be 1x1x2x2 tensor, each element will equal tanh(1).
            # Then the output will be z * state_update_input,
            # a 1x1x2x2 tensor, each element will equal sigmoid(1) * tanh(1) ~ 0.55677 .
            #
            # 2) For the case of stateless (i.e. is_stateful==False):
            # Because the past feature equals zero, the output state from the first iteration would
            # be zero. The second iteration will be the same as the stateful case, so the output
            # is the same as the stateful case.
            output_tensor = convgru2d_minimal(input_tensor)
            output_value = sess.run(output_tensor, feed_dict=feed_dict)
            np.testing.assert_array_almost_equal(
                0.55677 * np.ones((1, 1, 2, 2)), output_value
            )

            # Now everything is the same as previous test, but the bias values will be 1.
            value_for_bias_variables = np.ones([1], dtype=np.float32)
            if is_stateful:
                # In addition, state for the previous time step will be 1. as well.
                state_input_tensor = np.ones([1, 1, 2, 2], dtype=np.float32)
                feed_dict = {convgru2d_minimal._initial_state: state_input_tensor}
            else:
                # For stateless model, there is no initial state but there are feature maps
                # from previous frames stored in _past_features
                past_features_tensor = np.ones([1, 1, 2, 2], dtype=np.float32)
                feed_dict = {}
                for i in range(self.INPUT_SEQUENCE_LENGTH_IN_FRAMES - 1):
                    feed_dict[
                        convgru2d_minimal._past_features[i]
                    ] = past_features_tensor

            # Weights have 6 projection variables and 3 bias variables.
            convgru2d_minimal.set_weights(
                6 * [value_for_projections_variables] + 3 * [value_for_bias_variables]
            )

            # 1) For the case of stateful (i.e. is_stateful==True):
            # The below sub-session should compute the value of
            # the GRU-operations output after time-step.
            # z will be 1x1x2x2 tensor, each element will equal sigmoid(1+1+1).
            # r will be 1x1x2x2 tensor, each element will equal sigmoid(1+1+1).
            # state_update_input will be 1x1x2x2 tensor, each element will equal:
            #                   tanh(1+sigmoid(3)+1) ~ 0.994564
            # Then the output will be z * state_update_input,
            # a 1x1x2x2 tensor, each element will equal:
            #                 ( (1-sigmoid(3)) * 1. +  sigmoid(3) * 0.994564 )
            #               = ( (1-0.95257413) * 1.) + 0.95257413 * 0.994564
            #               ~ 0.99482181 .
            #
            # 2) For the case of stateless (i.e. is_stateful==False):
            # In the first iteration,
            # z will be 1x1x2x2 tensor, each element will equal sigmoid(1+1)
            # r will be 1x1x2x2 tensor, each element will equal sigmoid(1+1)
            # h will be 1x1x2x2 tensor, each element will equal tanh(1+1)
            # state will be 1x1x2x2 tensor, each element will equal: sigmoid(2) * tanh(2)
            # In the second iteration,
            # z will be 1x1x2x2 tensor, each element will equal sigmoid(1+state+1)
            # r will be 1x1x2x2 tensor, each element will equal sigmoid(1+state+1)
            # h will be 1x1x2x2 tensor, each element will equal tanh(1+state*sigmoid(2+state)+1)
            # final state will be 1x1x2x2 tensor, each element will equal:
            # sigmoid(2+state)*math.tanh(2+state*sigmoid(2+state))+(1-sigmoid(2+state))*state
            # ~ 0.98481372 .

            output_value = sess.run(output_tensor, feed_dict=feed_dict)
            np.testing.assert_array_almost_equal(
                expected_output * np.ones((1, 1, 2, 2)), output_value
            )

    @parameterized.expand(test_cases_1)
    def test_convgru2d_export_model(self, is_stateful, sequence_length_in_frames):
        """Test if ConvGRU2D is constructed with is_export_model flag as expected.

        Resulting object must have is_export_model flag set as True.
        It should be stateful.
        Its initial state must have proper placeholder name.
        """
        keras.backend.clear_session()
        # Construct the layer with is_export_model flag.
        convgru2d_exporting = ConvGRU2DExport(
            model_sequence_length_in_frames=sequence_length_in_frames,
            input_sequence_length_in_frames=sequence_length_in_frames,
            state_scaling=1.0,
            input_shape=[None, 1, 2, 2],
            initial_state_shape=[None, 1, 2, 2],
            spatial_kernel_height=1,
            spatial_kernel_width=1,
            is_stateful=is_stateful,
        )
        inputs = keras.Input(shape=(1, 2, 2))
        convgru2d_exporting(inputs)
        # Test the corresponding boolean members.
        assert convgru2d_exporting.is_stateful == is_stateful
        assert convgru2d_exporting.is_export_model
        # Test the name of the state input for the export model.
        assert (
            convgru2d_exporting._initial_state.name
            == "conv_gru_2d_export/state_placeholder:0"
        )
        assert (
            len(convgru2d_exporting._past_features)
            == convgru2d_exporting.input_sequence_length_in_frames - 1
        )

    # This test is only used for stateful model
    @parameterized.expand([[True], [False]])
    def test_convgru2d_export_model_violations(self, is_stateful):
        """Test if the expected AssertionErrors are raised when tyring to construct improperly.

        An AssertionError will be raised if the constructed arguments are not
        compatible with is_export_model flag.
        """
        # Test that AssertionError is raised if model_sequence_lenght is not 1.
        with pytest.raises(AssertionError):
            ConvGRU2DExport(
                model_sequence_length_in_frames=2,
                input_sequence_length_in_frames=1,
                state_scaling=1.0,
                input_shape=[None, 1, 2, 2],
                initial_state_shape=[None, 1, 2, 2],
                spatial_kernel_height=1,
                spatial_kernel_width=1,
                is_stateful=is_stateful,
            )
        # Test that AssertionError is raised if input_sequence_lenght is not 1.
        with pytest.raises(AssertionError):
            ConvGRU2DExport(
                model_sequence_length_in_frames=1,
                input_sequence_length_in_frames=2,
                state_scaling=1.0,
                input_shape=[None, 1, 2, 2],
                initial_state_shape=[None, 1, 2, 2],
                spatial_kernel_height=1,
                spatial_kernel_width=1,
                is_stateful=is_stateful,
            )
