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

import keras

import numpy as np
from nvidia_tao_tf1.core.models.templates.rnn_conv2d_base import RNNConv2dBase
import tensorflow as tf


class RNNConv2d(RNNConv2dBase):
    """Convolutional RNN Module."""

    TYPE_NAME = "RNN"

    def _get_id_init(self):
        return "glorot_uniform"

    def build(self, input_shapes):
        """Builds the RNN module.

        NOTE: Subclasses can modify the initial recurrent matrix by overriding `_get_id_init`.
        """
        input_shape = input_shapes[0]
        n_input_shape = self._get_normalized_size(input_shape)

        self.W_x = self.add_weight(
            name="W_x",
            shape=[
                self.kernel_size[0],
                self.kernel_size[1],
                n_input_shape[1],
                self.filters,
            ],
            initializer="glorot_uniform",
            trainable=True,
            regularizer=self.kernel_regularizer,
        )

        self.W_h = self.add_weight(
            name="W_h",
            shape=self._get_hidden_shape(),
            initializer=self._get_id_init(),
            trainable=True,
            regularizer=self.kernel_regularizer,
        )

        self.bias = self.add_weight(
            name="bias",
            shape=self._cvt_to_df([1, self.filters, 1, 1]),
            initializer="zeros",
            trainable=True,
            regularizer=self.bias_regularizer,
        )

        super(RNNConv2d, self).build(input_shapes)

    def iteration(self, x, state):
        """
        Implements the recurrent activation on a single timestep.

        Args:
            x (tf.Tensor): The input tensor for the current timestep.
            state (tf.Tensor): The state of the recurrent module, up to the current timestep.

        Returns:
            state (tf.Tensor): The state of the recurrent module after processing this timestep.
        """
        state = state * self.state_scaling

        z = self._conv2d(x, self.W_x) + self._conv2d(state, self.W_h)
        z = self._bias_add(z, self.bias)
        z = self._activation(z, name="state_output" if self.is_export_mode else None)

        state = z

        return state

    def _activation(self, inputs, name=None):
        return keras.layers.Activation(self.activation_type, name=name)(inputs)


class IRNNConv2d(RNNConv2d):
    """Convolutional RNN module with identity initialization."""

    TYPE_NAME = "IRNN"

    def _get_id_init(self):
        shape = self._get_hidden_shape()
        np_init = 0.01 * np.random.randn(*shape)
        c_y = shape[0] // 2
        c_x = shape[1] // 2

        np_init[c_y, c_x, :, :] += np.identity(self.filters)

        return tf.compat.v1.initializers.constant(value=np_init)

    def _activation(self, inputs, name=None):
        return tf.nn.relu(inputs, name=name)
