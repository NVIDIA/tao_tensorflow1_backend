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

"""Test convolutional gated recurrent unit (GRU) custom Keras layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import numpy as np
from nvidia_tao_tf1.core.models.templates.conv_gru_2d import ConvGRU2D
from parameterized import parameterized

import tensorflow as tf


class TestConvGRU2D(tf.test.TestCase):
    """Test convolutional gated recurrent unit (GRU) custom Keras layer."""

    def setUp(self):
        """Set up the test fixture: construct a ConvGRU2D object."""
        super(TestConvGRU2D, self).setUp()

        # Shape: [channels, height, width]
        self.GRU_INPUT_SHAPE = [None, 3, 1, 2]

        self.MODEL_SEQUENCE_LENGTH_IN_FRAMES = 2
        self.INPUT_SEQUENCE_LENGTH_IN_FRAMES = 2
        self.STATE_SCALING = 1.0
        self.IS_STATEFUL = False

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

        self.convgru2d = ConvGRU2D(
            model_sequence_length_in_frames=self.MODEL_SEQUENCE_LENGTH_IN_FRAMES,
            input_sequence_length_in_frames=self.INPUT_SEQUENCE_LENGTH_IN_FRAMES,
            state_scaling=self.STATE_SCALING,
            input_shape=self.GRU_INPUT_SHAPE,
            initial_state_shape=self.INITIAL_STATE_SHAPE,
            spatial_kernel_height=self.SPATIAL_KERNEL_HEIGHT,
            spatial_kernel_width=self.SPATIAL_KERNEL_WIDTH,
            kernel_regularizer=self.KERNEL_REGULARIZER,
            bias_regularizer=self.BIAS_REGULARIZER,
        )

    def test_convgru2d(self):
        """Test if convgru2d layer parameters are initialized as expected values."""
        input_shape_expected = self.GRU_INPUT_SHAPE

        model_sequence_length_in_frames_expected = self.MODEL_SEQUENCE_LENGTH_IN_FRAMES
        input_sequence_length_in_frames_expected = self.INPUT_SEQUENCE_LENGTH_IN_FRAMES
        state_scaling_expected = self.STATE_SCALING
        is_stateful_expected = self.IS_STATEFUL

        # Set the initial state shape.
        # Shape: [number_channels in a state, height, width].
        initial_state_shape_expected = self.INITIAL_STATE_SHAPE
        spatial_kernel_height_expected = self.SPATIAL_KERNEL_HEIGHT
        spatial_kernel_width_expected = self.SPATIAL_KERNEL_WIDTH

        kernel_regularizer = self.KERNEL_REGULARIZER
        bias_regularizer = self.BIAS_REGULARIZER

        assert input_shape_expected == self.convgru2d.rnn_input_shape
        assert (
            list(initial_state_shape_expected[1:])
            == self.convgru2d.initial_state_shape[1:]
        )
        assert spatial_kernel_height_expected == self.convgru2d.spatial_kernel_height
        assert spatial_kernel_width_expected == self.convgru2d.spatial_kernel_width
        assert (
            model_sequence_length_in_frames_expected
            == self.convgru2d.model_sequence_length_in_frames
        )
        assert (
            input_sequence_length_in_frames_expected
            == self.convgru2d.input_sequence_length_in_frames
        )
        assert np.isclose(state_scaling_expected, self.convgru2d.state_scaling)
        assert is_stateful_expected == self.convgru2d.is_stateful
        assert (
            kernel_regularizer["config"]["l1"] == self.convgru2d.kernel_regularizer.l1
        )
        assert (
            kernel_regularizer["config"]["l2"] == self.convgru2d.kernel_regularizer.l2
        )
        assert bias_regularizer["config"]["l1"] == self.convgru2d.bias_regularizer.l1
        assert bias_regularizer["config"]["l2"] == self.convgru2d.bias_regularizer.l2

    def test_convgru2d_variables(self):
        """Test the trainable variables of the ConvGRU2D Layer.

        1) Test if the trainable variables are built with the correct shapes and names.
        2) Test if the number of trainable variables are correct.
        3) Test if the output shape is set correctly while upon calling the layer.
        """
        variable_names_expected = set(
            [
                "conv_gru_2d/W_z:0",
                "conv_gru_2d/W_r:0",
                "conv_gru_2d/W_h:0",
                "conv_gru_2d/U_z:0",
                "conv_gru_2d/U_r:0",
                "conv_gru_2d/U_h:0",
                "conv_gru_2d/b_z:0",
                "conv_gru_2d/b_r:0",
                "conv_gru_2d/b_h:0",
            ]
        )

        inputs = keras.Input(
            shape=(
                self.GRU_INPUT_SHAPE[1],
                self.GRU_INPUT_SHAPE[2],
                self.GRU_INPUT_SHAPE[3],
            )
        )
        output = self.convgru2d(inputs)

        # Check variable names.
        weights_actual = self.convgru2d.weights
        variable_names_actual = {weight.name for weight in weights_actual}

        assert variable_names_expected == variable_names_actual

        # Check the number of trainable variables. This should be 9: 3 x U_, 3 X W_ and 3 X bias.
        assert len(self.convgru2d.trainable_weights) == 9

        # Check the shapes of input to state projections.
        # Shape [height, width, in_channels, out_channels].
        var_shape_expected = [
            self.SPATIAL_KERNEL_HEIGHT,
            self.SPATIAL_KERNEL_WIDTH,
            self.GRU_INPUT_SHAPE[1],
            self.INITIAL_STATE_SHAPE[1],
        ]

        for local_var_name in ["W_z", "W_r", "W_h"]:
            var_shape_actual = getattr(self.convgru2d, local_var_name).shape.as_list()
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
            var_shape_actual = getattr(self.convgru2d, local_var_name).shape.as_list()
            assert var_shape_expected == var_shape_actual

        # Check the shapes of bias variables.
        # Shape [out_channels].
        var_shape_expected = [self.INITIAL_STATE_SHAPE[1]]

        for local_var_name in ["b_z", "b_r", "b_h"]:
            var_shape_actual = getattr(self.convgru2d, local_var_name).shape.as_list()
            assert var_shape_expected == var_shape_actual

        # Check the output shape for the dimensions that can be inferred.
        # Shape [batch_size, num_filters, grid_height, grid_width].
        output_shape_expected = [
            None,
            self.INITIAL_STATE_SHAPE[1],
            None,
            self.INITIAL_STATE_SHAPE[3],
        ]
        assert output_shape_expected == output.shape.as_list()

    @parameterized.expand([["all"], ["last"]])
    def test_convgru2d_output_values_single_step(self, output_type):
        """Test the value of the output tensor after calling a single step ConvGRU2D Layer."""
        with self.test_session() as sess:
            # A close-to-minimum sized GRU.
            # This GRU will implement 1x1 operations on 2x2 grid. in-channels:1, out-channels:1 .
            convgru2d_minimal = ConvGRU2D(
                model_sequence_length_in_frames=1,
                input_sequence_length_in_frames=1,
                state_scaling=1.0,
                input_shape=[None, 1, 2, 2],
                initial_state_shape=[None, 1, 2, 2],
                spatial_kernel_height=1,
                spatial_kernel_width=1,
                output_type=output_type,
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

            # The below sub-session should compute the value of
            # the GRU-operations output after time-step.
            # z will be 1x1x2x2 tensor, each element will equal sigmoid(1).
            # r will be 1x1x2x2 tensor, each element will equal sigmoid(1).
            # state_update_input will be 1x1x2x2 tensor, each element will equal tanh(1).
            # Then the output will be z * state_update_input,
            # a 1x1x2x2 tensor, each element will equal sigmoid(1) * tanh(1) ~ 0.55677 .
            output_tensor = convgru2d_minimal(input_tensor)
            output_value = sess.run(output_tensor)
            np.testing.assert_array_almost_equal(
                0.55677 * np.ones((1, 1, 2, 2)), output_value
            )

            # Now everything is the same as previous test, but the bias values will be 1.
            value_for_bias_variables = np.ones([1], dtype=np.float32)
            # Weights have 6 projection variables and 3 bias variables.
            convgru2d_minimal.set_weights(
                6 * [value_for_projections_variables] + 3 * [value_for_bias_variables]
            )

            # The below sub-session should compute the value of
            # the GRU-operations output after time-step.
            # z will be 1x1x2x2 tensor, each element will equal sigmoid(1+1).
            # r will be 1x1x2x2 tensor, each element will equal sigmoid(1+1).
            # state_update_input will be 1x1x2x2 tensor, each element will equal tanh(1+1).
            # Then the output will be z * state_update_input,
            # a 1x1x2x2 tensor, each element will equal sigmoid(2) * tanh(2) ~ 0.849112675 .
            output_value = sess.run(output_tensor)
            np.testing.assert_array_almost_equal(
                0.849112675 * np.ones((1, 1, 2, 2)), output_value
            )

    @parameterized.expand([["all"], ["last"]])
    def test_convgru2d_output_values_2_steps(self, output_type):
        """Test the value of the output tensor after calling a 2-step ConvGRU2D Layer."""
        with self.test_session() as sess:
            # Now test the GRU, for 2 time steps inference.
            # This GRU will implement 1x1 operations on 2x2 grid. in-channels:1, out-channels:1.
            convgru2d_2steps = ConvGRU2D(
                model_sequence_length_in_frames=2,
                input_sequence_length_in_frames=2,
                state_scaling=1.0,
                input_shape=[None, 1, 2, 2],
                initial_state_shape=[None, 1, 2, 2],
                spatial_kernel_height=1,
                spatial_kernel_width=1,
                output_type=output_type,
            )
            inputs = keras.Input(shape=(1, 2, 2))
            convgru2d_2steps(inputs)

            # Manually set the weights to specific values.
            value_for_projections_variables = np.ones([1, 1, 1, 1], dtype=np.float32)
            # Set the bias variable to zero first.
            value_for_bias_variables = np.zeros([1], dtype=np.float32)

            # Weights have 6 projection variables and 3 bias variables.
            convgru2d_2steps.set_weights(
                6 * [value_for_projections_variables] + 3 * [value_for_bias_variables]
            )

            # Present a [ones, zeros] input to the network.
            # Note that the number of input tensors equals 2*sequence_length_in_frames.
            # First input sequence in the batch consists of tensors of ones.
            # Second input sequence in the batch consist of zero tensors.
            # The second input is included in order to check that the ordering of the
            # output states is correct.
            input_tensor = tf.constant(
                np.concatenate((np.ones([2, 1, 2, 2]), np.zeros([2, 1, 2, 2])), axis=0),
                dtype=tf.float32,
            )

            # The below sub-session should compute the value of
            # the GRU-operations output after time-step.
            # z_0 will be 1x1x2x2 tensor, each element will equal sigmoid(1).
            # r_0 will be 1x1x2x2 tensor, each element will equal sigmoid(1).
            # state_update_input_0 will be 1x1x2x2 tensor, each element will equal tanh(1).

            # Then the state_1 for the first sequence will be z_0 * state_update_input_0,
            #   a 2x2 tensor, where each element will equal sigmoid(1) * tanh(1) ~ 0.55677.
            # state_1 for the second sequence will be zero.
            expected_state_1_seq_1 = 0.55677 * np.ones((1, 1, 2, 2))
            expected_state_1_seq_2 = np.zeros((1, 1, 2, 2))

            # z_1 will be 1x1x2x2 tensor, each element will be
            # sigmoid(1 + 0.55677) ~ 0.825889 .

            # r_1 will be 1x1x2x2 tensor, each element will be
            # sigmoid(1 + 0.55677) ~ 0.825889.

            # state_update_input_1 will be 1x1x2x2 tensor, each element will equal
            # tanh(0.55677 * 0.825889 + 1) ~ 0.897619618.

            # state_2 for the first sequence will be z_1 * state_update_input_1 + (1-z_1) * state_0
            #   a 1x1x2x2 tensor, each element will equal
            #   0.825889 * 0.897619618 + (1.-0.825889) * 0.55677 ~ 0.8382739.
            # state_2 for the second sequence will be zero.
            expected_state_2_seq_1 = 0.8382739 * np.ones((1, 1, 2, 2))
            expected_state_2_seq_2 = np.zeros((1, 1, 2, 2))

            output_tensor = convgru2d_2steps(input_tensor)
            output_value = sess.run(output_tensor)

            if output_type == "last":
                np.testing.assert_array_almost_equal(
                    np.concatenate(
                        (expected_state_2_seq_1, expected_state_2_seq_2), axis=0
                    ),
                    output_value,
                )
            elif output_type == "all":
                np.testing.assert_array_almost_equal(
                    np.concatenate(
                        (
                            expected_state_1_seq_1,
                            expected_state_2_seq_1,
                            expected_state_1_seq_2,
                            expected_state_2_seq_2,
                        ),
                        axis=0,
                    ),
                    output_value,
                )

    @parameterized.expand([["all"], ["last"]])
    def test_convgru2d_output_values_2_times_stateful(self, output_type):
        """Test the value of the output tensor after calling a 1-step ConvGRU2D Layer 2 times."""
        with self.test_session() as sess:
            # Now test the GRU, for 2 time steps stateful inference with 2 single steps.
            # This GRU will implement 1x1 operations on 2x2 grid. in-channels:1, out-channels:1.
            convgru2d_2steps = ConvGRU2D(
                model_sequence_length_in_frames=1,
                input_sequence_length_in_frames=1,
                state_scaling=1.0,
                input_shape=[None, 1, 2, 2],
                initial_state_shape=[None, 1, 2, 2],
                spatial_kernel_height=1,
                spatial_kernel_width=1,
                is_stateful=True,
                output_type=output_type,
            )
            inputs = keras.Input(shape=(1, 2, 2))
            convgru2d_2steps(inputs)

            # Manually set the weights to specific values.
            value_for_projections_variables = np.ones([1, 1, 1, 1], dtype=np.float32)
            # Set the bias variable to zero first.
            value_for_bias_variables = np.zeros([1], dtype=np.float32)

            # Weights have 6 projection variables and 3 bias variables.
            convgru2d_2steps.set_weights(
                6 * [value_for_projections_variables] + 3 * [value_for_bias_variables]
            )

            # Present a [ones, zeros] input to the network.
            # Note that the number of input tensors equals 2.
            # First input sequence in the batch consists of tensors of ones.
            # Second input sequence in the batch consist of zero tensors.
            input_tensor = tf.constant(
                np.concatenate((np.ones([1, 1, 2, 2]), np.zeros([1, 1, 2, 2])), axis=0),
                dtype=tf.float32,
            )

            # The below sub-session should compute the value of
            # the GRU-operations output after time-step.
            # z_0 will be 1x1x2x2 tensor, each element will equal sigmoid(1).
            # r_0 will be 1x1x2x2 tensor, each element will equal sigmoid(1).
            # state_update_input_0 will be 1x1x2x2 tensor, each element will equal tanh(1).

            # Then the state_1 for the first sequence will be z_0 * state_update_input_0,
            #   a 2x2 tensor, where each element will equal sigmoid(1) * tanh(1) ~ 0.55677.
            # state_1 for the second sequence will be zero.
            expected_state_1_seq_1 = 0.55677 * np.ones((1, 1, 2, 2))
            expected_state_1_seq_2 = np.zeros((1, 1, 2, 2))

            # z_1 will be 1x1x2x2 tensor, each element will be
            # sigmoid(1 + 0.55677) ~ 0.825889 .

            # r_1 will be 1x1x2x2 tensor, each element will be
            # sigmoid(1 + 0.55677) ~ 0.825889.

            # state_update_input_1 will be 1x1x2x2 tensor, each element will equal
            # tanh(0.55677 * 0.825889 + 1) ~ 0.897619618.

            # state_2 for the first sequence will be z_1 * state_update_input_1 + (1-z_1) * state_0
            #   a 1x1x2x2 tensor, each element will equal
            #   0.825889 * 0.897619618 + (1.-0.825889) * 0.55677 ~ 0.8382739.
            # state_2 for the second sequence will be zero.
            expected_state_2_seq_1 = 0.8382739 * np.ones((1, 1, 2, 2))
            expected_state_2_seq_2 = np.zeros((1, 1, 2, 2))

            # Call the layer 2 times to check that the hidden state is correctly
            # saved after the first call.
            output_tensor_1 = convgru2d_2steps(input_tensor)
            output_value_1 = sess.run(output_tensor_1)
            output_value_2 = sess.run(output_tensor_1)

            np.testing.assert_array_almost_equal(
                np.concatenate(
                    (expected_state_1_seq_1, expected_state_1_seq_2), axis=0
                ),
                output_value_1,
            )

            np.testing.assert_array_almost_equal(
                np.concatenate(
                    (expected_state_2_seq_1, expected_state_2_seq_2), axis=0
                ),
                output_value_2,
            )

    @parameterized.expand([["last"], ["all"]])
    def test_convgru2d_stateful_gradients(self, output_type):
        """Test the value of the output tensor after calling a 2-step ConvGRU2D Layer."""
        with self.test_session() as sess:
            # Test the GRU for stateful training gradients.
            convgru2d_2steps = ConvGRU2D(
                model_sequence_length_in_frames=2,
                input_sequence_length_in_frames=2,
                state_scaling=1.0,
                input_shape=[None, 1, 2, 2],
                initial_state_shape=[2, 1, 2, 2],
                spatial_kernel_height=1,
                spatial_kernel_width=1,
                is_stateful=True,
                output_type=output_type,
            )
            inputs = keras.Input(shape=(1, 2, 2))
            convgru2d_2steps(inputs)

            # Manually set the weights to specific values.
            value_for_projections_variables = np.ones([1, 1, 1, 1], dtype=np.float32)
            # Set the bias variable to zero first.
            value_for_bias_variables = np.zeros([1], dtype=np.float32)

            # Weights have 6 projection variables and 3 bias variables.
            convgru2d_2steps.set_weights(
                6 * [value_for_projections_variables] + 3 * [value_for_bias_variables]
            )

            # Note that the number of input tensors  equals sequence_length_in_frames.
            input_tensor = tf.constant(np.ones([4, 1, 2, 2], dtype=np.float32))
            output_tensor = convgru2d_2steps(input_tensor)
            gradients = keras.backend.gradients(
                output_tensor, convgru2d_2steps.trainable_weights
            )

            # Take one forward pass, which assigns the state variables.
            sess.run(output_tensor)
            # Obtain the gradients. This breaks if the above state assignment changed the state
            # variable shape. Since we set the batch size to correct value upon construction,
            # the state variable shape remains the same, and no error is raised.
            gradient_values = sess.run(gradients)

            # The gradient values should be nonzero for all weights.
            assert all([float(g) != 0.0 for g in gradient_values])

    @parameterized.expand(
        [["all", False], ["all", True], ["last", False], ["last", True]]
    )
    def test_convgru2d_variable_input_size(self, output_type, is_stateful):
        """Test the ConvGRU2D Layer with variable input size.

        Test that the Conv2DGRU layer accepts an input tensor whose height and width
        dimensions are different to the dimensions given when the layer was created
        and outputs a tensor of correct size.
        """
        convgru2d_variable_input = ConvGRU2D(
            model_sequence_length_in_frames=self.MODEL_SEQUENCE_LENGTH_IN_FRAMES,
            input_sequence_length_in_frames=self.INPUT_SEQUENCE_LENGTH_IN_FRAMES,
            state_scaling=self.STATE_SCALING,
            input_shape=self.GRU_INPUT_SHAPE,
            initial_state_shape=self.INITIAL_STATE_SHAPE,
            spatial_kernel_height=self.SPATIAL_KERNEL_HEIGHT,
            spatial_kernel_width=self.SPATIAL_KERNEL_WIDTH,
            kernel_regularizer=self.KERNEL_REGULARIZER,
            bias_regularizer=self.BIAS_REGULARIZER,
            is_stateful=is_stateful,
            output_type=output_type,
        )

        inputs = keras.Input(
            shape=(
                self.GRU_INPUT_SHAPE[1],
                self.GRU_INPUT_SHAPE[2],
                self.GRU_INPUT_SHAPE[3],
            )
        )
        convgru2d_variable_input(inputs)

        batch_size = 4
        if output_type == "last":
            expected_batch_size = batch_size // self.INPUT_SEQUENCE_LENGTH_IN_FRAMES
        elif output_type == "all":
            expected_batch_size = batch_size
        inputs_shape = inputs.shape.as_list()
        input_tensor = tf.ones(
            [batch_size, inputs_shape[1], inputs_shape[2] + 2, inputs_shape[3] + 3]
        )

        output_tensor = convgru2d_variable_input(input_tensor)

        expected_output_shape = (
            expected_batch_size,
            self.INITIAL_STATE_SHAPE[1],
        ) + tuple(input_tensor.shape.as_list()[2:])

        with self.cached_session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_value = sess.run(output_tensor)

            assert output_value.shape == expected_output_shape

        # Test also compute_output_shape function.
        assert (
            convgru2d_variable_input.compute_output_shape(input_tensor.shape.as_list())
            == expected_output_shape
        )
