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
"""Input layer for non-temporal models that receive a temporal input."""

import keras


class NonTemporalInput(keras.layers.Layer):
    """
    Non-temporal input module.

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
        super(NonTemporalInput, self).__init__(**kwargs)
        self.is_export_mode = is_export_mode

    def call(self, x):
        """
        Call function.

        Composes this part of the graph. If this layer is in export mode, then it will
        simply forward the input. If it's in training mode, then the final value in
        the 2nd dimension of the tensor will be selected.

        Args:
            x (tf.Tensor): The input tensor.
        """
        if self.is_export_mode:
            return x

        return x[:, -1]

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape given the specified input shape.

        The behavior changes between training and export mode.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        if self.is_export_mode:
            return input_shape

        return input_shape[:1] + input_shape[2:]

    def prepare_for_export(self):
        """Configures this layer for export mode if this didn't already happen in the init."""
        self.is_export_mode = True
