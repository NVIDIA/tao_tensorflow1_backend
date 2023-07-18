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

"""A convolutional GRU layer for DriveNet gridbox models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
import tensorflow as tf


def conv2d(x, W):
    """Convenience wrapper around 2d convolution."""
    return K.conv2d(x, W, strides=(1, 1), padding="same", data_format="channels_first")


class ConvGRU2D(Layer):
    """Class containing convolutional GRU operations on sequential tensors."""

    # Variable names of this layer, grouped according to their functions.
    INPUT_PROJECTION_VARIABLES_NAMES = ["W_z", "W_r", "W_h"]
    STATE_VARIABLES_NAMES = ["U_z", "U_r", "U_h"]
    BIAS_VARIABLES_NAMES = ["b_z", "b_r", "b_h"]

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
        is_stateful=False,
        name="conv_gru_2d",
        output_type="last",
        **kwargs
    ):
        """Constructor for ConvGRU2D layer.

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
                Batch_size needs to be specified only for stateful training.
            initial_state_shape (list / tuple): Shape of the initial state (M, F, H, W), where
                [M: batch_size or None,
                 F: number_out_channels,
                 H: input_height,
                 W: input_width].
                Batch size needs to be specified for stateful training.
            spatial_kernel_height (int): Height of the convolution kernel within the GRU.
            spatial_kernel_width (int): Width of the convolution kernel within the GRU.
            kernel_regularizer (keras.regularizers.Regularizer instance or keras.regularizers config
                dict): Regularizer to be applied to convolution kernels.
            bias_regularizer (keras.regularizers.Regularizer instance or keras.regularizers config
                dict): Regularizer to be applied to biases.
            is_stateful (bool): Whether the GRU keeps track of state from minibatch to minibatch.
            name (str): Name of the layer.
            output_type (str): Whether to give an output for all frames in the sequences or only
                for the last frame. Possible values: 'all' or 'last' (default).
        Raises:
            AssertionError: If height and width for input and state shapes are not equal or
                if state_scaling is not in range [0, 1] or if input_sequence_length_in_frames is
                less than model_sequence_length_in_frames.
        """
        super(ConvGRU2D, self).__init__(**kwargs)

        self.rnn_input_shape = input_shape
        self.model_sequence_length_in_frames = model_sequence_length_in_frames
        self.input_sequence_length_in_frames = input_sequence_length_in_frames
        assert (
            input_sequence_length_in_frames >= model_sequence_length_in_frames
        ), "The input sequence must be at least as long as the \
             sequence with gradient flow."
        self.is_stateful = is_stateful
        self.state_scaling = state_scaling
        assert 0 <= state_scaling <= 1, "state_scaling must lie in range [0, 1]."
        # Set the initial state shape.
        initial_state_shape = list(initial_state_shape)
        if initial_state_shape[0] is None:
            # Singleton batch dimension works for forward pass (stateful inference),
            # as the variable shape can be changed in assign op.
            initial_state_shape[0] = 1
        self.initial_state_shape = initial_state_shape
        assert tuple(input_shape[2:4]) == tuple(
            initial_state_shape[2:4]
        ), "Both height and width for input and state shapes must be equal."
        self.state_depth = initial_state_shape[1]
        self.spatial_kernel_height = spatial_kernel_height
        self.spatial_kernel_width = spatial_kernel_width
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.name = name
        self.state_output_name = "_".join(["state_output", self.name])
        self.output_type = output_type
        self.past_feature_shape = input_shape
        self._set_initial_state()

    def _set_initial_state(self):
        """Initialize the state for the minibatch.

        Note that initial_state refers to the initial state of the minibatch.
        For the first minibatch this will be zeros (assuming batch size 1).
        The initial state for the next minibatches will be set as the final state
        of the previous minibatch. This will also update the initial_state dimension
        according to the batch size of the training/evaluation.
        """
        if self.is_stateful:
            self._initial_state = tf.Variable(
                tf.zeros([1], dtype=K.floatx()), trainable=False
            )
        else:
            self._initial_state = tf.zeros([], dtype=K.floatx())

    def build(self, input_shape):
        """Create layer variables.

        Args:
            input_shape: Tensor shape as a list/tuple (N, C, H, W), where
                [N: batch_size * input_sequence_length_in_frames / None,
                 C: input_channels,
                 H: input_height,
                 W: input_width].
        """
        number_input_channels = input_shape[1]

        # Create variables here.
        # Input projection variables' shape is: [H, W, in_channels, out_channels].
        for var_name in self.INPUT_PROJECTION_VARIABLES_NAMES:
            temp_var = self.add_weight(
                name=var_name,
                shape=[
                    self.spatial_kernel_height,
                    self.spatial_kernel_width,
                    number_input_channels,
                    self.state_depth,
                ],
                initializer="glorot_uniform",
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
            setattr(self, var_name, temp_var)
        # State variables' shape is: [H, W, in_channels, out_channels].
        # Note that in_channels = out_channels = state_depth for those variables.
        for var_name in self.STATE_VARIABLES_NAMES:
            temp_var = self.add_weight(
                name=var_name,
                shape=[
                    self.spatial_kernel_height,
                    self.spatial_kernel_width,
                    self.state_depth,
                    self.state_depth,
                ],
                initializer="glorot_uniform",
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
            setattr(self, var_name, temp_var)
        # Bias variables' shape is: [out_channels].
        for var_name in self.BIAS_VARIABLES_NAMES:
            temp_var = self.add_weight(
                name=var_name,
                shape=[self.state_depth],
                initializer="zeros",
                trainable=True,
                regularizer=self.bias_regularizer,
            )
            setattr(self, var_name, temp_var)

        super(ConvGRU2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute layer's output shape.

        Args:
            input_shape: Tensor shape as a list/tuple (N, C, H, W), where
                [N: batch_size * input_sequence_length_in_frames / None,
                 C: input_channels,
                 H: input_height,
                 W: input_width].
        Returns:
            Output tensor shape as a tuple (M, F, H, W), where
                [M: batch_size / None,
                 F: number_out_channels,
                 H: input_height,
                 W: input_width].
        """
        if self.output_type == "last":
            batch_size = (
                input_shape[0] // self.input_sequence_length_in_frames
                if input_shape[0] is not None
                else None
            )
        elif self.output_type == "all":
            batch_size = input_shape[0]
        else:
            raise ValueError("output_type = {} not supported".format(self.output_type))
        return (batch_size, self.state_depth) + tuple(input_shape[2:])

    def _split_inputs_per_time(self, input_tensor):
        """Splits the input tensor into a per time list of tensors.

        Args:
            input_tensor (N, C, H, W), where
                [N: batch_size * input_sequence_length_in_frames,
                 C: number_input_channels,
                 H: grid_height,
                 W: grid_width]): input to the GRU.

        Returns:
            inputs_per_time: List of tensors each,
                with shape (M, F, H, W), where
                [M: batch_size,
                 F: number_out_channels,
                 H: input_height,
                 W: input_width].
        """
        inputs_per_time = [
            input_tensor[offset :: self.input_sequence_length_in_frames]
            for offset in range(self.input_sequence_length_in_frames)
        ]

        return inputs_per_time

    def call(self, input_tensor):
        """Call the GRU network.

        Args:
            input_tensor (N, C, H, W), where
                [N: batch_size * input_sequence_length_in_frames,
                 C: number_input_channels,
                 H: grid_height,
                 W: grid_width]): input to the GRU.

        Returns:
            state: Final state of the GRU after input_sequence_length_in_frames cell operations,
                with shape (M, F, H, W), where
                [M: batch_size * output_sequence_length,
                 F: number_out_channels,
                 H: input_height,
                 W: input_width],
                where output_sequence_length == 1 if self.output_type == 'last' and
                output_sequence_length == input_sequence_length if self.output_type = 'all'.
        """
        # model_sequence_length_in_frames is the number of last layers in the unrolled GRU,
        # upto which the gradients still back-propagate.
        model_sequence_length_in_frames = self.model_sequence_length_in_frames

        # state_depth is both number of input_channels and number of
        # output filters to an operation to a GRU.
        state_depth = self.state_depth

        # inputs_per_time is a list with a length of sequence_length_in_frames,
        # where each item is a tensor with shape [ batch_size,
        #                                          rnn_input_channels,
        #                                          height,
        #                                          width ].
        inputs_per_time = self._split_inputs_per_time(input_tensor)
        number_frames_per_sequence = len(inputs_per_time)

        # Compute the state shape dynamically from the input_tensor shape.
        input_shape = tf.shape(input=input_tensor)
        state_shape = tf.stack(
            [
                input_shape[0] // self.input_sequence_length_in_frames,
                state_depth,
                input_shape[2],
                input_shape[3],
            ]
        )

        if self.is_stateful:
            # Create a correct size tensor of zeros so that the initial state is
            # broadcasted to correct shape if its shape is [1].
            zero_state = tf.fill(state_shape, tf.constant(0.0, dtype=K.floatx()))
            state = zero_state + tf.identity(self._initial_state)
        else:
            state = tf.fill(state_shape, tf.constant(0.0, dtype=K.floatx()))

        output = []
        for step, instant_input in enumerate(inputs_per_time):
            state = self._step(instant_input, state, step)

            # Update the initial state for the next mini-batch.
            if step == model_sequence_length_in_frames - 1:
                state = self._update_network_state(state)

            if self.output_type == "all":
                output.append(state)

            # Stop the gradients before the extent of the model memory,
            # so that the gradients are utilized for the model sequence_length_in_frames frames.
            if step == number_frames_per_sequence - model_sequence_length_in_frames - 1:
                state = tf.stop_gradient(state)

        if self.output_type == "all":
            # Stack the outputs to a 4D tensor with shape
            # [M: batch_size * input_sequence_length_in_frames,
            #  F: number_out_channels,
            #  H: input_height,
            #  W: input_width]
            # such that in the first dimension first input_sequence_length_in_frames
            # values belong to the first sequence, the next input_sequence_length_in_frames
            # values belong to the second sequence, and so on.
            output_tensor = tf.stack(output, axis=0)
            output_tensor = tf.transpose(a=output_tensor, perm=(1, 0, 2, 3, 4))
            output_tensor = tf.reshape(
                output_tensor,
                (input_shape[0], state_depth, input_shape[2], input_shape[3]),
            )
            return output_tensor
        return state

    def _step(self, input_tensor, state, step):
        """Single time step update.

        Args:
            input_tensor (N, C, H, W), where
                [N: batch_size,
                 C: number_input_channels,
                 H: grid_height,
                 W: grid_width]): input to the GRU.
            state (N, D, H, W), where
                [N: batch_size,
                 F: number_out_channels,
                 H: grid_height,
                 W: grid_width]): GRU state tensor.
                 For stateless GRU layer, the initial state is zeros. Because zeros tensor
                 (or tf.fill) is not supported in TensorRT, a `None` state is used to represent
                 zero state. In this case, this function will simply not use the input variable
                 `state`.
            step (int): Time step index.
        """
        # Scale the state down to simulate the necessary leak.
        if state is not None:
            state = state * self.state_scaling
        # Convolutional GRU operations.
        z = conv2d(input_tensor, self.W_z)
        if state is not None:
            z += conv2d(state, self.U_z)
        z = K.bias_add(z, self.b_z, data_format="channels_first")
        z = K.sigmoid(z)

        r = conv2d(input_tensor, self.W_r)
        if state is not None:
            r += conv2d(state, self.U_r)
        r = K.bias_add(r, self.b_r, data_format="channels_first")
        r = K.sigmoid(r)

        h = conv2d(input_tensor, self.W_h)
        if state is not None:
            h += conv2d(tf.multiply(state, r), self.U_h)
        h = K.bias_add(h, self.b_h, data_format="channels_first")
        h = K.tanh(h)
        # Name the the output node operation to ease the TRT export.
        state_name = (
            self.state_output_name
            if step == self.input_sequence_length_in_frames - 1
            else None
        )
        if state is not None:
            state = tf.subtract(
                tf.multiply(z, h), tf.multiply((z - 1.0), state), name=state_name
            )
        else:
            state = tf.multiply(z, h, name=None)

        return state

    def _update_network_state(self, state):
        """Updates the state of the GRU depending on the statefulness.

        Args:
            state: Tensor with shape (M, F, H, W), where
                [M: batch_size,
                 F: number_out_channels,
                 H: input_height,
                 W: input_width].

        Returns:
            Updated state with the same shape.
        """
        # If the layer is stateful, update the initial state for the next mini-batch.
        if self.is_stateful:
            # Tf.assign does not propagate gradients, so do the assign op as a control dependency.
            # See https://github.com/tensorflow/tensorflow/issues/17735
            updated_state = tf.compat.v1.assign(
                ref=self._initial_state, value=state, validate_shape=False
            )
            with tf.control_dependencies([updated_state]):
                state = tf.identity(state)

        return state

    def get_config(self):
        """Get layer's configuration.

        Returns:
            The dict of configuration.
        """
        config = {
            "model_sequence_length_in_frames": self.model_sequence_length_in_frames,
            "input_sequence_length_in_frames": self.input_sequence_length_in_frames,
            "state_scaling": self.state_scaling,
            "input_shape": self.input_shape,
            "initial_state_shape": self.initial_state_shape,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "spatial_kernel_height": self.spatial_kernel_height,
            "spatial_kernel_width": self.spatial_kernel_width,
            "is_stateful": self.is_stateful,
            "output_type": self.output_type,
        }

        base_config = super(ConvGRU2D, self).get_config()
        base_config.update(config)
        return base_config

    @classmethod
    def from_config(cls, config):
        """Create convolutional GRU layer from its config.

        Args:
            config (dict) Layer config.

        Returns:
            A convolutional GRU layer.
        """
        # Handle backward compatibility.
        if "sequence_length_in_frames" in config:
            config.update(
                {
                    "model_sequence_length_in_frames": config[
                        "sequence_length_in_frames"
                    ],
                    "input_sequence_length_in_frames": config[
                        "sequence_length_in_frames"
                    ],
                    "state_scaling": 1.0,
                    "is_stateful": False,
                }
            )
            config.pop("sequence_length_in_frames")
        return super(ConvGRU2D, cls).from_config(config)
