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

"""Quantized Conv2D for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import keras.backend as K
from keras.backend import image_data_format

from keras.backend.tensorflow_backend import _preprocess_padding

from keras.layers import Conv2D
from keras.layers import InputSpec

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

logger = logging.getLogger(__name__)

DATA_FORMAT_MAP = {"channels_first": "NCHW", "channels_last": "NHWC"}


def _conv2d(
    x,
    kernel,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    quantize_input=False,
    bitwidth=8,
    scaling_factor=None,
    training=None,
):
    """2D convolution.

    Arguments:
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: tuple of 2 integers.
        quantize_input: boolean, quantize both the weights and the inputs.
        bitwidth: number of bits to use for quantization.
        scaling_factor: variable holding the moving average of absolute max of input tensor.
        training: boolean or interger determining training or alternative phase.

    Returns:
        A tensor, result of 2D convolution.

    Raises:
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in DATA_FORMAT_MAP:
        raise ValueError("Unknown data_format " + str(data_format))

    tf_data_format = DATA_FORMAT_MAP[data_format]
    # Avoid Tensorflow's implicit assymetric padding by explicit symmetric padding.
    # See https://stackoverflow.com/questions/42924324/tensorflows-asymmetric-padding-assumptions
    if padding == "same":
        filter_shape = kernel.get_shape()
        width_padding = ((filter_shape[0].value - 1) * dilation_rate[0] + 1) // 2
        height_padding = ((filter_shape[1].value - 1) * dilation_rate[1] + 1) // 2
        if tf_data_format == "NCHW":
            padding_pattern = [
                [0, 0],
                [0, 0],
                [width_padding, width_padding],
                [height_padding, height_padding],
            ]
        else:  # 'NHWC'
            padding_pattern = [
                [0, 0],
                [width_padding, width_padding],
                [height_padding, height_padding],
                [0, 0],
            ]
        x = tf.pad(x, padding_pattern, mode="CONSTANT")
        padding = "valid"

    nhwc_roundtrip = not K._has_nchw_support() and tf_data_format == "NCHW"

    if nhwc_roundtrip:
        tf_data_format = "NHWC"
        x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

    padding = _preprocess_padding(padding)

    if quantize_input:
        assert (
            scaling_factor is not None
        ), "Quantization enabled but scaling factor parameter not defined."
        # Quantize the input.
        keras_learning_phase = K.learning_phase()
        if tf.is_tensor(keras_learning_phase):
            keras_learning_phase = 0
            logger.warning(
                "QuantizedConv2D: Keras learning_phase was not set. Assuming evaluation phase."
            )

        if keras_learning_phase:
            batch_min = math_ops.reduce_min(x, name="BatchMin")
            batch_min = math_ops.minimum(batch_min, 0.0)
            batch_max = math_ops.reduce_max(x, name="BatchMax")
            batch_max = math_ops.maximum(batch_max, 0.0)

            abs_max = math_ops.maximum(
                math_ops.abs(batch_min), math_ops.abs(batch_max), name="tensor_scale"
            )

            assign_max = moving_averages.assign_moving_average(
                scaling_factor, abs_max, 0.999, name="AssignMaxEma"
            )
        else:
            assign_max = scaling_factor

        assign_min = math_ops.negative(assign_max)

        assert assign_min.get_shape() == [], "Unexpected shape for tensor minimum."
        assert assign_max.get_shape() == [], "Unexpected shape for tensor maximum."
        x = tf.quantization.quantize_and_dequantize(
            input=x,
            input_min=assign_min,
            input_max=assign_max,
            range_given=True,
            signed_input=True,
            num_bits=bitwidth,
        )

    # Quantizing the weights.
    kernel = tf.quantization.quantize_and_dequantize(
        input=kernel,
        input_min=0.0,
        input_max=0.0,
        range_given=False,
        signed_input=True,
        num_bits=bitwidth,
    )

    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
    )

    if nhwc_roundtrip:
        x = tf.transpose(x, (0, 3, 1, 2))  # NCHW -> NHWC

    return x


class QuantizedConv2D(Conv2D):
    """Quantized 2D convolution layer (e.g. spatial convolution over images).

    This layer creates a quantized convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
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
            Note that `"same"` is slightly inconsistent across backends with
            `strides` != 1, as described
            [here](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix
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
        `(batch, filters, new_rows, new_cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if `data_format` is `"channels_last"`.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        quantize=True,
        bitwidth=8,
        **kwargs
    ):
        """Construct the QuantizedConv2D layer."""
        super(QuantizedConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
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

    def call(self, inputs, training=None):
        """Keras layer call."""

        outputs = _conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            quantize_input=self.quantize_input,
            bitwidth=self.bitwidth,
            scaling_factor=self.scaling_factor,
            training=training,
        )

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def build(self, input_shape):
        """Keras layer build."""

        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        if self.quantize_input:
            self.scaling_factor = self.add_weight(
                shape=[],
                initializer=init_ops.constant_initializer(6.0),
                name="scaling_factor",
                trainable=False,
            )
        else:
            self.scaling_factor = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def get_config(self):
        """Get the layer configuration for QuantizedConv2D layer."""
        config = super(QuantizedConv2D, self).get_config()
        config["quantize"] = self.quantize_input
        config["bitwidth"] = self.bitwidth
        return config
