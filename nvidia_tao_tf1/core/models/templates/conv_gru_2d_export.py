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

"""A convolutional GRU layer for DriveNet gridbox TensorRT models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

from nvidia_tao_tf1.core.models.templates.conv_gru_2d import ConvGRU2D
import tensorflow as tf


class ConvGRU2DExport(ConvGRU2D):
    """TRT compatible class containing convolutional GRU operations on sequential tensors."""

    def __init__(
        self,
        model_sequence_length_in_frames,
        input_sequence_length_in_frames,
        state_scaling,
        input_shape,
        initial_state_shape,
        spatial_kernel_height,
        spatial_kernel_width,
        kernel_regularizer=None,
        bias_regularizer=None,
        is_stateful=True,
        name="conv_gru_2d_export",
        output_type="last",
        **kwargs
    ):
        """Constructor for ConvGRU2D export layer.

        Args:
            model_sequence_length_in_frames (int): How many steps the GRU will be tracked for
                gradient computation (starting from the last frame). That is, a stop_gradient
                operation will be applied just before the last model_sequence_length_in_frames
                steps of the input sequence.
            input_sequence_length_in_frames (int): Length of the sequence that is presented to the
                GRU in the minibatch. The GRU might, however, stop the gradient flow in the middle
                of the sequence. The gradients will propagate backwards through the last
                model_sequence_length_in_frames steps of the input.
            state_scaling (float): A constant scaling factor in range [0,1] in order to simulate
                an exponential decay.
            input_shape (list / tuple): Input tensor shape (N, C, H, W), where
                [N: batch_size * sequence_length_in_frames / None,
                 C: input_channels,
                 H: input_height,
                 W: input_width].
            initial_state_shape (list / tuple): Shape of the initial state (M, F, H, W), where
                [M: batch_size / None,
                 F: number_out_channels,
                 H: input_height,
                 W: input_width].
            spatial_kernel_height (int): Height of the convolution kernel within the GRU.
            spatial_kernel_width (int): Width of the convolution kernel within the GRU.
            kernel_regularizer (keras.regularizers.Regularizer instance):
                Regularizer to be applied to convolution kernels.
            bias_regularizer (keras.regularizers.Regularizer instance):
                Regularizer to be applied to biases.
            is_stateful (bool): Whether the GRU keeps track of state from minibatch to minibatch.
            name (str): Name of the layer.
            output_type (str): Whether to give an output for all frames in the sequences or only
                for the last frame. Unused in this layer since input and output consist of only
                single frame.
        Raises:
            AssertionError: If height and width for input and state shapes are not equal or
                if state_scaling is not in range [0, 1] or if input_sequence_length_in_frames is
                less than model_sequence_length_in_frames.
        """
        super(ConvGRU2DExport, self).__init__(
            model_sequence_length_in_frames,
            input_sequence_length_in_frames,
            state_scaling,
            input_shape,
            initial_state_shape,
            spatial_kernel_height,
            spatial_kernel_width,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            is_stateful=is_stateful,
            name=name,
            **kwargs
        )
        self.is_export_model = True
        self.is_stateful = is_stateful
        self.state_output_name = "state_output"
        self._set_initial_state()
        self._check_export_layer_sanity()

    def _check_export_layer_sanity(self):
        """For GRU/TRT export layer must be stateful and sequence lengths must be one.

        Raises:
            AssertionError: if either of the sequence length parameters differs from 1
                or if the layer is not stateful.
        """
        assert (
            self.input_sequence_length_in_frames == self.model_sequence_length_in_frames
        ), (
            "Input sequence length and model sequence length must be the same for the "
            "TRT export layer."
        )
        if self.is_stateful:
            assert (
                self.input_sequence_length_in_frames == 1
            ), "Input sequence length must be 1 for the TRT export layer."
            assert (
                self.model_sequence_length_in_frames == 1
            ), "Model sequence length must be 1 for the TRT export layer."

    def _set_initial_state(self):
        """Initialize the state for the minibatch.

        Note that initial_state refers to the initial state of the minibatch.
        For the export model, the initial state is a placeholder to be fed from
        outside.
        """
        self.state_input_name = "/".join([self.name, "state_placeholder"])
        # In the TRT model, initial state needs to be provided externally.
        if not hasattr(self, "_initial_state"):
            self._initial_state = tf.compat.v1.placeholder(
                K.floatx(),
                shape=[1] + self.initial_state_shape[1:],
                name=self.state_input_name,
            )
        if not hasattr(self, "_past_features"):
            self._past_feature_count = self.input_sequence_length_in_frames - 1
            self._past_features = []
            self._past_feature_names = []
            # Keep them in the order of time stamps, i.e. t-3, t-2, t-1
            for i in reversed(range(self._past_feature_count)):
                past_feature_name = "/".join([self.name, "past_feature", str(i + 1)])
                self._past_features.append(
                    tf.compat.v1.placeholder(
                        K.floatx(),
                        shape=self.past_feature_shape,
                        name=past_feature_name,
                    )
                )
                self._past_feature_names.append(past_feature_name)

    def call(self, input_tensor):
        """Call the GRU network.

        Args:
            input_tensor (N, C, H, W), where
                [N: batch_size,
                 C: number_input_channels,
                 H: grid_height,
                 W: grid_width]): input to the GRU.

        Returns:
            state: Final state of the GRU after input_sequence_length_in_frames cell operations,
                with shape (M, F, H, W), where
                [M: batch_size,
                 F: number_out_channels,
                 H: input_height,
                 W: input_width].
        """

        if self.is_stateful:
            # Set the initial state and the computation.
            state = tf.identity(self._initial_state)
            state = self._step(input_tensor, state, 0)
        else:
            # inputs_per_time contains [... past_feature/2, past_feature/1, current_feature]
            inputs_per_time = self._past_features + [input_tensor]
            state = None

            for step, instant_input in enumerate(inputs_per_time):
                state = self._step(instant_input, state, step)

        return state
