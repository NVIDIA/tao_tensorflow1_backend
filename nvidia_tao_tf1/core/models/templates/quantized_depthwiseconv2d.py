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

"""Quantized DepthwiseConv2D for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import keras.backend as K

from keras.layers import DepthwiseConv2D

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

logger = logging.getLogger(__name__)


# @zeyuz: note Keras 2.2.4 DepthwiseConv2D has no dilation support. Dilation rate is here to
#    support future keras version. This value should NOT be set to anything else than (1, 1)
class QuantizedDepthwiseConv2D(DepthwiseConv2D):
    """Quantized Depthwise 2D convolution.

    Depthwise convolution performs
    just the first step of a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.

    # Arguments
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be 'channels_last'.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. 'linear' activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix.
        bias_initializer: Initializer for the bias vector.
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its 'activation').
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix.
        bias_constraint: Constraint function applied to the bias vector
        quantize: Quantize the input in addition to weights.
        bitwidth: Quantization precision.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(batch, rows, cols, channels)`
        if `data_format` is `"channels_last"`.
    # Output shape
        4D tensor with shape:
        `(batch, channels * depth_multiplier, new_rows, new_cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(batch, new_rows, new_cols,  channels * depth_multiplier)`
        if `data_format` is `"channels_last"`.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 quantize=True,
                 bitwidth=8,
                 **kwargs):
        """Init function."""
        super(QuantizedDepthwiseConv2D, self).__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.quantize_input = quantize
        self.bitwidth = bitwidth

    def build(self, input_shape):
        """Keras layer build."""

        # The parent class build function should be called first so quantize input is weights[-1]
        super(QuantizedDepthwiseConv2D, self).build(input_shape)

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
        """Call function to apply QAT."""
        if self.quantize_input:
            assert (
                self.scaling_factor is not None
            ), "Quantization enabled but scaling factor parameter not defined."
            # Quantize the input.
            keras_learning_phase = K.learning_phase()
            if tf.is_tensor(keras_learning_phase):
                keras_learning_phase = 0
                logger.warning(
                    "QuantizedDepthwiseConv2D: Keras learning_phase not set. Assuming evaluation."
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
            input=self.depthwise_kernel,
            input_min=0.0,
            input_max=0.0,
            range_given=False,
            signed_input=True,
            num_bits=self.bitwidth,
        )

        outputs = K.depthwise_conv2d(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        """get config function."""
        config = super(QuantizedDepthwiseConv2D, self).get_config()
        config["quantize"] = self.quantize_input
        config["bitwidth"] = self.bitwidth
        return config
