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

"""Modulus Keras-extensions for mixed-precision training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import keras
from keras import backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras.regularizers import Regularizer
from keras.utils import conv_utils


"""Logger for Keras tensorflow backend."""
logger = logging.getLogger(__name__)


@interfaces.legacy_add_weight_support
def _layer_add_weight(
    self,
    name,
    shape,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    constraint=None,
):
    """Add a weight variable to the layer.

    # Arguments
        name: String, the name for the weight variable.
        shape: The shape tuple of the weight.
        dtype: The dtype of the weight.
        initializer: An Initializer instance (callable).
        regularizer: An optional Regularizer instance.
        trainable: A boolean, whether the weight should
            be trained via backprop or not (assuming
            that the layer itself is also trainable).
        constraint: An optional Constraint instance.
    # Returns
        The created weight variable.
    """
    initializer = initializers.get(initializer)
    # If dtype is given, use it directly.
    if dtype:
        variable_dtype = output_dtype = dtype
    # In mixed precision training, by default, variables are created fp32 and cast to fp16.
    elif not dtype and K.floatx() == "float16":
        variable_dtype = "float32"
        output_dtype = "float16"
    # If dtype is not given, use the global default.
    else:
        variable_dtype = output_dtype = K.floatx()

    weight = K.variable(
        initializer(shape, dtype=variable_dtype),
        dtype=variable_dtype,
        name=name,
        constraint=constraint,
    )
    if regularizer is not None:
        self.add_loss(regularizer(weight))
    if trainable:
        self._trainable_weights.append(weight)
    else:
        self._non_trainable_weights.append(weight)

    # For mixed-precision training, return a cast version of the variable.
    if output_dtype != variable_dtype:
        return K.cast(weight, output_dtype)
    return weight


def _batch_normalization_build(self, input_shape):
    dim = input_shape[self.axis]
    if dim is None:
        raise ValueError(
            "Axis " + str(self.axis) + " of "
            "input tensor should have a defined dimension "
            "but the layer received an input with shape " + str(input_shape) + "."
        )
    self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})
    shape = (dim,)

    # For mixed-precision training, BN variables have to be created as float32.
    if K.floatx() == "float16":
        dtype_for_bn_variables = "float32"
    else:
        dtype_for_bn_variables = None

    if self.scale:
        self.gamma = self.add_weight(
            shape=shape,
            name="gamma",
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
            dtype=dtype_for_bn_variables,
        )
    else:
        self.gamma = None
    if self.center:
        self.beta = self.add_weight(
            shape=shape,
            name="beta",
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
            dtype=dtype_for_bn_variables,
        )
    else:
        self.beta = None
    self.moving_mean = self.add_weight(
        shape=shape,
        name="moving_mean",
        initializer=self.moving_mean_initializer,
        trainable=False,
        dtype=dtype_for_bn_variables,
    )
    self.moving_variance = self.add_weight(
        shape=shape,
        name="moving_variance",
        initializer=self.moving_variance_initializer,
        trainable=False,
        dtype=dtype_for_bn_variables,
    )
    self.built = True


def _batch_normalization_call(self, inputs, training=None):
    input_shape = K.int_shape(inputs)
    # Prepare broadcasting shape.
    ndim = len(input_shape)
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[self.axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[self.axis] = input_shape[self.axis]

    # Determines whether broadcasting is needed.
    needs_broadcasting = sorted(reduction_axes) != list(range(ndim))[:-1]

    def normalize_inference():
        if needs_broadcasting:
            # In this case we must explicitly broadcast all parameters.
            broadcast_moving_mean = K.reshape(self.moving_mean, broadcast_shape)
            broadcast_moving_variance = K.reshape(self.moving_variance, broadcast_shape)
            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
            else:
                broadcast_beta = None
            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            else:
                broadcast_gamma = None
            return K.batch_normalization(
                inputs,
                K.cast(broadcast_moving_mean, inputs.dtype),
                K.cast(broadcast_moving_variance, inputs.dtype),
                K.cast(broadcast_beta, inputs.dtype),
                K.cast(broadcast_gamma, inputs.dtype),
                axis=self.axis,
                epsilon=self.epsilon,
            )
        else:
            return K.batch_normalization(
                inputs,
                K.cast(self.moving_mean, inputs.dtype),
                K.cast(self.moving_variance, inputs.dtype),
                K.cast(self.beta, inputs.dtype),
                K.cast(self.gamma, inputs.dtype),
                axis=self.axis,
                epsilon=self.epsilon,
            )

    # If the learning phase is *static* and set to inference:
    if training in {0, False}:
        return normalize_inference()

    # If the learning is either dynamic, or set to training:
    normed_training, mean, variance = K.normalize_batch_in_training(
        inputs, self.gamma, self.beta, reduction_axes, epsilon=self.epsilon
    )

    # if K.backend() != 'cntk':
    #     sample_size = K.prod([K.shape(inputs)[axis]
    #                           for axis in reduction_axes])
    #     sample_size = K.cast(sample_size, dtype=K.dtype(variance))

    #     # sample variance - unbiased estimator of population variance
    #     variance *= sample_size / (sample_size - (1.0 + self.epsilon))

    self.add_update(
        [
            K.moving_average_update(self.moving_mean, mean, self.momentum),
            K.moving_average_update(self.moving_variance, variance, self.momentum),
        ],
        inputs,
    )

    # Pick the normalized form corresponding to the training phase.
    return K.in_train_phase(normed_training, normalize_inference, training=training)


class _RegularizerL1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x):
        regularization = 0.0
        if self.l1:
            regularization += K.sum(K.cast(self.l1, x.dtype) * K.abs(x))
        if self.l2:
            regularization += K.sum(K.cast(self.l2, x.dtype) * K.square(x))
        return regularization

    def get_config(self):
        return {"l1": float(self.l1), "l2": float(self.l2)}


def _conv2dtranspose_call(self, inputs):
    input_shape = K.shape(inputs)
    batch_size = input_shape[0]
    if self.data_format == "channels_first":
        h_axis, w_axis = 2, 3
    else:
        h_axis, w_axis = 1, 2

    height, width = input_shape[h_axis], input_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides
    if self.output_padding is None:
        out_pad_h = out_pad_w = None
    else:
        out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_length(
        height, stride_h, kernel_h, self.padding, out_pad_h, self.dilation_rate[0]
    )
    out_width = conv_utils.deconv_length(
        width, stride_w, kernel_w, self.padding, out_pad_w, self.dilation_rate[1]
    )
    if self.data_format == "channels_first":
        output_shape = (batch_size, self.filters, out_height, out_width)
    else:
        output_shape = (batch_size, out_height, out_width, self.filters)

    outputs = K.conv2d_transpose(
        inputs,
        self.kernel,
        output_shape,
        self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate,
    )

    if self.use_bias:
        outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

    if self.activation is not None:
        return self.activation(outputs)
    return outputs


def patch():
    """Apply the patches to the module."""
    _layer_add_weight.__name__ = "add_weight"
    keras.engine.Layer.add_weight = _layer_add_weight
    _batch_normalization_build.__name__ = "build"
    keras.layers.BatchNormalization.build = _batch_normalization_build
    _batch_normalization_call.__name__ = "call"
    keras.layers.BatchNormalization.call = _batch_normalization_call
    _RegularizerL1L2.__name__ = "L1L2"
    keras.regularizers.L1L2 = _RegularizerL1L2
    _conv2dtranspose_call.__name__ = "call"
    keras.layers.Conv2DTranspose.call = _conv2dtranspose_call
