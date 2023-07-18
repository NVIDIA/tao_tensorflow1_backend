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
"""Convolutional RNN Layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import keras
import keras.backend

import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.data_format import (
    CHANNELS_FIRST,
    DataFormat,
)
from nvidia_tao_tf1.core.decorators.arg_scope import add_arg_scope

logger = logging.getLogger(__name__)


def duple(t):
    """Returns `t` if it's a tuple, otherwise returns (`t`, `t`)."""
    if isinstance(t, (list, tuple)):
        if len(t) == 1:
            return t + t
        assert len(t) == 2
        return t
    return (t, t)


class RNNConv2dBase(keras.layers.Layer):
    """Base implementation of a recurrent convolutional layer."""

    @add_arg_scope
    def __init__(
        self,
        filters,
        kernel_size,
        num_backprop_timesteps=None,
        state_scaling=1.0,
        is_stateful=False,
        activation_type="relu",
        data_format=CHANNELS_FIRST,
        kernel_regularizer=None,
        bias_regularizer=None,
        space_time_kernel=False,
        seq_length_schedule=None,
        name=None,
        is_export_mode=False,
        **kwargs
    ):
        """
        Standard constructor for recurrent convolutional layers.

        This module expects the input to be either (B*T, C, H, W) or (B*T, H, W, C) depending
        on the value of `data_format`.

        Args:
            filters (int): The number of convolutional filters to produce as the output.
            kernel_size (int, tuple): The spatial size of the input filters.
                                      NOTE: If `space_time_kernel` is true, then this also applies
                                            to the recurrent convolution.
            num_backprop_timesteps (int): The number of sequence frames to backprop through time.
            state_scaling (float): Explicitly scale the input recurrent activations from the
                                   previous time step. Must be 0 < `state_scaling` <= 1.0.
            is_stateful (bool): Controls whether history is preserved across batches during
                                training.
            activation_type (str): Where applicable, specifies the activation type of the model.
            data_format (nvidia_tao_tf1.blocks.multi_source_loader.DataFormat): The format of
                                the input tensors. Must be either CHANNELS_FIRST or CHANNELS_LAST.
            kernel_regularizer (`regularizer`): regularizer for the kernels.
            bias_regularizer (`regularizer`): regularizer for the biases.
            space_time_kernel (bool): Whether the convolution over time also includes a spatial
                                      component. If true, `kernel_size` is used for this
                                      convolution operation too.
            seq_length_schedule (None, int, list): Provides a schedule mechanism for controlling
                the number of time steps that are processing.

                If `None` (default), then all time steps will be processed every batch.
                If `int`, the processed sequence length will increase by 1 for each multiple of
                          iterations of the specified value until the full input length is
                          processed. For example, if `seq_length_schedule = 100` and the full
                          sequence length is 5, then for the first 100 iterations, only the last
                          input will be processed. From iteration 100-199, the last two inputs
                          will be processed. By step 400, all inputs will be processed.
                If `list`, the number of inputs that will be processed is equal to the number of
                           elements in the list that the current iteration is greater than, plus 1.
            name (str): The name of the layer.
            is_export_mode (bool): Whether or not this layer operates in stateful inference mode.
        """
        super(RNNConv2dBase, self).__init__(name=name)

        self.num_backprop_timesteps = num_backprop_timesteps
        self.state_scaling = state_scaling
        if state_scaling <= 0.0 or state_scaling > 1.0:
            raise ValueError("State scaling must be in the range (0, 1]")
        self.is_stateful = is_stateful
        self.filters = filters
        self.kernel_size = duple(kernel_size)
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._data_format = data_format
        self._inital_state = None
        self.n_input_shape = None
        self.state_shape = None
        self.is_space_time_kernel = space_time_kernel
        self.activation_type = activation_type
        self.seq_length_schedule = seq_length_schedule
        self.is_export_mode = is_export_mode

    def get_config(self):
        """Get layer's configuration.

        Returns:
            dict: The layer configuration.
        """
        config = super(RNNConv2dBase, self).get_config()
        self.build_config(config)
        return config

    def build_config(self, config):
        """
        Appends configuration parameters for keras serialization.

        Args:
            config (dict): Dictionary to append parameters to.
        """
        config["num_backprop_timesteps"] = self.num_backprop_timesteps
        config["state_scaling"] = self.state_scaling
        config["is_stateful"] = self.is_stateful
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["kernel_regularizer"] = self.kernel_regularizer
        config["bias_regularizer"] = self.bias_regularizer
        config["data_format"] = str(self._data_format)
        config["n_input_shape"] = self.n_input_shape
        config["is_space_time_kernel"] = self.is_space_time_kernel
        config["activation_type"] = self.activation_type
        config["seq_length_schedule"] = self.seq_length_schedule
        config["state_shape"] = self.state_shape

    @property
    def state_input_name(self):
        """The name of the input state layer."""
        return self.name + "/state_input"

    def prepare_for_export(self):
        """Configures the layer for export mode."""
        print("Preparing RNN for export!!!")
        self.is_export_mode = True
        self.is_stateful = True

        if not self.state_shape:
            raise ValueError(
                "The state shape has not been initialized. Has this layer ever "
                "been built?"
            )

        self._initial_state = tf.compat.v1.placeholder(
            keras.backend.floatx(), shape=[1] + self.state_shape[1:], name="state_input"
        )

    @property
    def data_format(self):
        """
        Returns the data format for this layer.

        This is preferred over `self._data_format` because that value may occasionally
        be a string. This method always ensures that the returned value is a `DataFormat`
        object.
        """
        if not isinstance(self._data_format, DataFormat):
            self._data_format = DataFormat(self._data_format)
        return self._data_format

    @property
    def bias_regularizer(self):
        """Returns the bias regularizer for this layer."""
        if isinstance(self._bias_regularizer, dict):
            return None
        return self._bias_regularizer

    @property
    def kernel_regularizer(self):
        """Returns the kernel regularizer for this layer."""
        if isinstance(self._kernel_regularizer, dict):
            return None
        return self._kernel_regularizer

    def build(self, input_shapes):
        """Initializes internal parameters given the shape of the inputs."""
        input_shape = input_shapes[0]
        n_input_shape = self._get_normalized_size(input_shape)
        self.n_input_shape = n_input_shape

        self.state_shape = self._cvt_to_df(
            [1, self.filters, n_input_shape[2], n_input_shape[3]]
        )

        self._inital_state = tf.zeros(self.state_shape, dtype=keras.backend.floatx())

        if self.is_stateful and not self.is_export_mode:
            self._inital_state = tf.Variable(self._initial_state, trainable=False)

        if self.is_export_mode:
            self.prepare_for_export()
        else:
            self.rnn = keras.layers.Lambda(self._do_work, self._rnn_output_shape)

        super(RNNConv2dBase, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        """Computes the output shape for this layer."""
        logger.info("Input Shapes: {}".format(input_shapes))

        output_shape = list(input_shapes[0])
        if not self.is_export_mode:
            output_shape[0] = None

        logger.info("Output Shape: {}".format(output_shape))

        return type(input_shapes[0])(output_shape)

    def call(self, args):
        """
        Construct the graph.

        Args:
            args (tuple<tf.Tensor,tf.Tensor>): The input tensor and temporal length tensor.

        Returns:
            tf.Tensor: The output tensor.
        """
        x, temporal_length = args

        if self.is_export_mode:
            return self.iteration(x, self._initial_state)

        if self.is_stateful:
            state = tf.identity(self._initial_state)
        else:
            state = tf.zeros(self.state_shape, dtype=keras.backend.floatx())

        ex_counter = tf.compat.v1.train.get_or_create_global_step()

        state = self.rnn([state, x, temporal_length, ex_counter])

        if self.is_stateful and not self.is_export_mode:
            state = tf.compat.v1.assign(
                ref=self._initial_state, value=state, validate_shape=False
            )

        return state

    def _do_work(self, args):
        initial_state = args[0]
        x = args[1]
        temporal_length = args[2]
        ex_counter = args[3]

        input_shape = tf.shape(input=x)
        temporal_shape = [
            input_shape[0] // temporal_length,
            temporal_length,
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ]
        temporal_x = tf.reshape(x, temporal_shape)

        time_first_x = tf.transpose(a=temporal_x, perm=[1, 0, 2, 3, 4])

        temporal_length_int64 = tf.cast(temporal_length, tf.int64)
        start_offset = self._get_step_offset(ex_counter, temporal_length_int64)

        def should_continue(t, *args):
            return start_offset + t < temporal_length_int64

        def _iteration(t, inputs, state):
            step = start_offset + t
            instant_input = inputs[step]

            state = self.iteration(instant_input, state)

            if self.num_backprop_timesteps:
                cond = tf.math.equal(
                    step, temporal_length_int64 - self.num_backprop_timesteps - 1
                )
                state = tf.cond(
                    pred=cond,
                    true_fn=lambda: tf.stop_gradient(state),
                    false_fn=lambda: state,
                )

            return t + 1, inputs, state

        initial_t = tf.Variable(0, dtype=tf.int64, trainable=False)

        invariant_state_shape = list(initial_state.get_shape())
        invariant_state_shape[0] = None
        invariant_state_shape = tf.TensorShape(invariant_state_shape)

        _, _, state = tf.while_loop(
            cond=should_continue,
            body=_iteration,
            loop_vars=(initial_t, time_first_x, initial_state),
            back_prop=True,
            parallel_iterations=1,
            swap_memory=False,
            shape_invariants=(
                initial_t.get_shape(),
                time_first_x.get_shape(),
                invariant_state_shape,
            ),
        )

        return state

    def iteration(self, x, state):
        """
        Implements the recurrent activation on a single timestep.

        Args:
            x (tf.Tensor): The input tensor for the current timestep.
            state (tf.Tensor): The state of the recurrent module, up to the current timestep.

        Returns:
            state (tf.Tensor): The state of the recurrent module after processing this timestep.
        """
        raise NotImplementedError("This method must be implemented by derived classes.")

    def _get_hidden_shape(self):
        st_shape = self.kernel_size if self.is_space_time_kernel else (1, 1)
        return [st_shape[0], st_shape[1], self.filters, self.filters]

    def _get_normalized_size(self, input_shape):
        return [
            v.value if isinstance(v, tf.compat.v1.Dimension) else v
            for v in self.data_format.convert_shape(input_shape, CHANNELS_FIRST)
        ]

    def _cvt_to_df(self, n_input_shape):
        return CHANNELS_FIRST.convert_shape(n_input_shape, self.data_format)

    def _get_step_offset(self, ex_counter, temporal_length):
        if self.seq_length_schedule is None:
            return tf.constant(0, dtype=tf.int64)
        if isinstance(self.seq_length_schedule, int):
            return tf.maximum(
                temporal_length - ex_counter // self.seq_length_schedule - 1, 0
            )
        if isinstance(self.seq_length_schedule, (list, tuple)):
            tf_schedule = tf.constant(self.seq_length_schedule, tf.int64)
            gt_schedule = tf.cast(tf.greater_equal(ex_counter, tf_schedule), tf.int64)
            gt_count = tf.reduce_sum(input_tensor=gt_schedule) + 1
            return tf.maximum(temporal_length - gt_count, 0)
        raise ValueError(
            "Unsupported sequence length schedule parameter. "
            "Expected (None, int, list). Got: {}".format(type(self.seq_length_schedule))
        )

    @staticmethod
    def _rnn_output_shape(input_shapes):
        return input_shapes[1]

    def _conv2d(self, x, W, strides=1, padding="same"):
        """Convenience wrapper around 2d convolution."""
        return keras.backend.conv2d(
            x,
            W,
            strides=duple(strides),
            padding=padding,
            data_format=str(self.data_format),
        )

    def _bias_add(self, x, b):
        """Convenience wrapper around B.bias_add."""
        return x + b
