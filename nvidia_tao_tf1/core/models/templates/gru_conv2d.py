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
"""Convolutional GRU Layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.models.templates.rnn_conv2d_base import RNNConv2dBase

import tensorflow as tf


class GRUConv2d(RNNConv2dBase):
    """Convolutional GRU Module."""

    TYPE_NAME = "GRU"

    # Variable names of this layer, grouped according to their functions.
    INPUT_PROJECTION_VARIABLES_NAMES = ["W_z", "W_r", "W_h"]
    STATE_VARIABLES_NAMES = ["U_z", "U_r", "U_h"]
    BIAS_VARIABLES_NAMES = ["b_z", "b_r", "b_h"]

    def build(self, input_shapes):
        """Initializes internal parameters given the shape of the inputs."""
        input_shape = input_shapes[0]
        n_input_shape = self._get_normalized_size(input_shape)
        self.n_input_shape = n_input_shape
        kernel_height, kernel_width = self.kernel_size

        # Create variables here.
        for var_name in self.INPUT_PROJECTION_VARIABLES_NAMES:
            tmp_var = self.add_weight(
                name=var_name,
                shape=[kernel_height, kernel_width, n_input_shape[1], self.filters],
                initializer="glorot_uniform",
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
            setattr(self, var_name, tmp_var)
        for var_name in self.STATE_VARIABLES_NAMES:
            tmp_var = self.add_weight(
                name=var_name,
                shape=self._get_hidden_shape(),
                initializer="glorot_uniform",
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
            setattr(self, var_name, tmp_var)
        for var_name in self.BIAS_VARIABLES_NAMES:
            tmp_var = self.add_weight(
                name=var_name,
                shape=self._cvt_to_df([1, self.filters, 1, 1]),
                initializer="zeros",
                trainable=True,
                regularizer=self.bias_regularizer,
            )
            setattr(self, var_name, tmp_var)

        super(GRUConv2d, self).build(input_shapes)

    def iteration(self, x, state):
        """
        Implements the recurrent activation on a single timestep.

        Args:
            x (tf.Tensor): The input tensor for the current timestep.
            state (tf.Tensor): The state of the recurrent module, up to the current timestep.

        Returns:
            state (tf.Tensor): The state of the recurrent module after processing this timestep.
        """
        # Scale the state down to simulate the necessary leak.
        state = state * self.state_scaling
        # Convolutional GRU operations
        z = self._conv2d(x, self.W_z) + self._conv2d(state, self.U_z)
        z = self._bias_add(z, self.b_z)
        z = tf.sigmoid(z)

        r = self._conv2d(x, self.W_r) + self._conv2d(state, self.U_r)
        r = self._bias_add(r, self.b_r)
        r = tf.sigmoid(r)

        h = self._conv2d(x, self.W_h) + self._conv2d(tf.multiply(state, r), self.U_h)
        h = self._bias_add(h, self.b_h)
        h = tf.tanh(h)

        out_name = "state_output" if self.is_export_mode else None

        state = tf.subtract(
            tf.multiply(z, h), tf.multiply((z - 1.0), state), name=out_name
        )

        return state
