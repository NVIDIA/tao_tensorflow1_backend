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

"""Quantized Dense Layer for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import keras.backend as K

from keras.layers import Dense

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

logger = logging.getLogger(__name__)


class QuantizedDense(Dense):
    """Quantized Dense layer in Keras for QAT.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        quantize: Quantize the input in addition to weights.
        bitwidth: Quantization precision.
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 quantize=True,
                 bitwidth=8,
                 **kwargs):
        """Initialize QuantizedDense layer."""
        super(QuantizedDense, self).__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.quantize_input = quantize
        self.bitwidth = bitwidth

    def build(self, input_shape):
        # The parent class build function should be called first so quantize input is weights[-1]
        """Allocate weights, etc to build the layer."""
        super(QuantizedDense, self).build(input_shape)

        if self.quantize_input:
            self.scaling_factor = self.add_weight(
                shape=[],
                initializer=init_ops.constant_initializer(6.0),
                name="scaling_factor",
                trainable=False,
            )
        else:
            self.scaling_factor = None

    def call(self, inputs):
        """quantization and inner product."""
        if self.quantize_input:
            assert (
                self.scaling_factor is not None
            ), "Quantization enabled but scaling factor parameter not defined."
            # Quantize the input.
            keras_learning_phase = K.learning_phase()
            if tf.is_tensor(keras_learning_phase):
                keras_learning_phase = 0
                logger.warning(
                    "QuantizedDense: Keras learning_phase not set. Assuming evaluation."
                )

            if keras_learning_phase:
                batch_min = math_ops.reduce_min(inputs, name="BatchMin")
                batch_min = math_ops.minimum(batch_min, 0.0)
                batch_max = math_ops.reduce_max(inputs, name="BatchMax")
                batch_max = math_ops.maximum(batch_max, 0.0)

                abs_max = math_ops.maximum(
                    math_ops.abs(batch_min), math_ops.abs(batch_max), name="tensor_scale"
                )

                assign_max = moving_averages.assign_moving_average(
                    self.scaling_factor, abs_max, 0.999, name="AssignMaxEma"
                )
            else:
                assign_max = self.scaling_factor

            assign_min = math_ops.negative(assign_max)

            assert assign_min.get_shape() == [], "Unexpected shape for tensor minimum."
            assert assign_max.get_shape() == [], "Unexpected shape for tensor maximum."
            inputs = tf.quantization.quantize_and_dequantize(
                input=inputs,
                input_min=assign_min,
                input_max=assign_max,
                range_given=True,
                signed_input=True,
                num_bits=self.bitwidth,
            )

        # Quantizing the weights.
        kernel = tf.quantization.quantize_and_dequantize(
            input=self.kernel,
            input_min=0.0,
            input_max=0.0,
            range_given=False,
            signed_input=True,
            num_bits=self.bitwidth,
        )
        output = K.dot(inputs, kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        """Get the config dict."""
        config = super(QuantizedDense, self).get_config()
        config["quantize"] = self.quantize_input
        config["bitwidth"] = self.bitwidth
        return config
