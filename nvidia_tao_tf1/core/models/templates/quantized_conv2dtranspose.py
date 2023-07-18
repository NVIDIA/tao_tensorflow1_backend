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

"""Quantized Conv2DTranspose for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import keras.backend as K

from keras.layers import Conv2DTranspose

import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

logger = logging.getLogger(__name__)


class QuantizedConv2DTranspose(Conv2DTranspose):
    """Quantized transposed convolution layer.

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
        output_padding: An integer or tuple/list of 2 integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
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
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
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
        If `output_padding` is specified:
        ```
        new_rows = ((rows - 1) * strides[0] + kernel_size[0]
                    - 2 * padding[0] + output_padding[0])
        new_cols = ((cols - 1) * strides[1] + kernel_size[1]
                    - 2 * padding[1] + output_padding[1])
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        output_padding=None,
        data_format=None,
        dilation_rate=(1, 1),
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
        **kwargs
    ):
        """init function."""

        super(QuantizedConv2DTranspose, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
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
            **kwargs)

        self.quantize_input = quantize
        self.bitwidth = bitwidth

    def build(self, input_shape):
        """Keras layer build."""

        # The parent class build function should be called first so quantize input is weights[-1]
        super(QuantizedConv2DTranspose, self).build(input_shape)

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
        """call function to apply QAT."""

        inputs_shape = K.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        if self.quantize_input:
            assert (
                self.scaling_factor is not None
            ), "Quantization enabled but scaling factor parameter not defined."
            # Quantize the input.
            keras_learning_phase = K.learning_phase()
            if tf.is_tensor(keras_learning_phase):
                keras_learning_phase = 0
                logger.warning(
                    "QuantizedConv2DTranspose: Keras learning_phase not set. Assuming evaluation."
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

        outputs = K.conv2d_transpose(
            inputs,
            kernel,
            output_shape,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

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
        config = super(QuantizedConv2DTranspose, self).get_config()
        config["quantize"] = self.quantize_input
        config["bitwidth"] = self.bitwidth
        return config
