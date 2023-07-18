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
"""Input layer for temporal models that receive a temporal input."""

import keras


class TemporalInput(keras.layers.Layer):
    """
    Temporal input module.

    This model is compatible with the modulus export facility that
    deals with the difference in model architecture between training
    and inference.
    """

    def __init__(self, is_export_mode=False, **kwargs):
        """
        Initialization.

        Args:
            is_export_mode (bool) Whether or not this layer should behave in single-frame
                                  inference mode (e.g. driveworks).
        """
        super(TemporalInput, self).__init__(**kwargs)
        self.is_export_mode = is_export_mode

    def call(self, x):
        """
        Call function.

        Composes this part of the graph. If this layer is in export mode, then it will
        simply forward the input. If it's in training mode, then all but the final 3
        tensor dimensions will be flattened.

        Args:
            x (tf.Tensor): The input tensor.
        """
        if self.is_export_mode:
            return x

        shape = x.shape
        return keras.backend.reshape(x, (-1, shape[2], shape[3], shape[4]))

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape given the specified input shape.

        The behavior changes between training and export mode.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        shape_type = type(input_shape)
        input_shape = list(input_shape)
        if self.is_export_mode:
            output_shape = shape_type([input_shape[0]] + input_shape[-3:])
        else:
            batch_dim = input_shape[0]
            temporal_dim = input_shape[1]
            if batch_dim is None or temporal_dim is None:
                batch_output = None
            else:
                batch_output = batch_dim * temporal_dim
            output_shape = shape_type([batch_output] + input_shape[2:])
        return output_shape

    def prepare_for_export(self):
        """Configures this layer for export mode if this didn't already happen in the init."""
        self.is_export_mode = True
