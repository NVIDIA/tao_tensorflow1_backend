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

"""Quantize and De-Quantize layer for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import keras.backend as K

from keras.layers import Layer
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

logger = logging.getLogger(__name__)


class QDQ(Layer):
    """Quantize and de-Quantize layer.

    This layer simulates the quantization of the output tensor of
        its input layer. This layer calculates the Exponential Moving
        Average (EMA) of the input tensor and use it as dynamic range
        for simulation of the quantization.

    # Arguments
        bitwidth: Quantization precision that this layer simulates.
            Default value is 8-bits.

    # Input shape
        Tensor with arbitrary shape:
        This layer reduces the entire tensor to min/max.
        So the input shape does not matter.

    # Output shape
        The same as input shape
    """

    def __init__(self, bitwidth=8, **kwargs):
        """Construct the QDQ layer."""
        super(QDQ, self).__init__(**kwargs)
        self.bitwidth = bitwidth

    def call(self, inputs):
        """Keras layer call."""

        # Quantize the input.
        assert tf.is_tensor(inputs), "The input to QDQ layer should be a tensor."
        x = inputs
        keras_learning_phase = K.learning_phase()
        if tf.is_tensor(keras_learning_phase):
            keras_learning_phase = 0
            logger.warning(
                "QDQ: Keras learning_phase was not set. Assuming evaluation phase."
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
                self.scaling_factor, abs_max, 0.999, name="AssignMaxEma"
            )
        else:
            assign_max = self.scaling_factor

        assign_min = math_ops.negative(assign_max)

        assert assign_min.get_shape() == [], "Unexpected shape for tensor minimum."
        assert assign_max.get_shape() == [], "Unexpected shape for tensor maximum."
        x = tf.quantization.quantize_and_dequantize(
            input=x,
            input_min=assign_min,
            input_max=assign_max,
            range_given=True,
            signed_input=True,
            num_bits=self.bitwidth,
        )
        return x

    def build(self, input_shape):
        """Keras layer build."""

        self.scaling_factor = self.add_weight(
            shape=[],
            initializer=init_ops.constant_initializer(6.0),
            name="scaling_factor",
            trainable=False,
        )

        self.built = True

    def get_config(self):
        """Get the layer configuration for QDQ layer."""
        config = {"bitwidth": self.bitwidth}
        base_config = super(QDQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """Compute the output shape of QDQ layer."""
        return input_shape
