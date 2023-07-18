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
"""Softargmax operator implementation in Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.initializers import Constant
from keras.layers import Layer
import numpy as np


class Softargmax(Layer):
    """Class for a custom Softargmax operator in Keras."""

    def __init__(self,
                 input_shape,
                 beta,
                 data_format='channels_first',
                 **kwargs):
        """Initialize the Softargmax operator.

        Args:
            input_shape (4-element list or tuple): Input shape to the Softargmax operator.
            beta (float): Coefficient used for multiplying the key-point maps after
                subtracting the channel-wise maximum.
        Optional args:
            data_format (str): Expected tensor format, either 'channels_first' or 'channels_last'.
                Default value is 'channels_first'.

        """
        assert isinstance(input_shape, (list, tuple))
        self._input_shape = tuple(input_shape)
        assert len(self._input_shape) == 4

        if data_format == 'channels_first':
            _, self._nkeypoints, self._height, self._width = self._input_shape
        elif data_format == 'channels_last':
            _, self._height, self._width, self._nkeypoints = self._input_shape
        else:
            raise ValueError(
                "Provide either 'channels_first' or 'channels_last' for `data_format`."
            )
        self._beta = beta
        self._data_format = data_format

        kwargs['trainable'] = False

        super(Softargmax, self).__init__(
            input_shape=self._input_shape, **kwargs)

        row_initializer, column_initializer = Softargmax._index_initializers(
            self._height, self._width, K.floatx())

        self._row_indexes = self.add_weight(
            name='row_indexes',
            shape=(1, 1, self._height, self._width),
            initializer=row_initializer,
            trainable=False)

        self._column_indexes = self.add_weight(
            name='column_indexes',
            shape=(1, 1, self._height, self._width),
            initializer=column_initializer,
            trainable=False)

    def build(self, input_shape):
        """Create the Softargmax operator at build time.

        Args:
            input_shape (4-element list or tuple): Input shape to the Softargmax operator.

        """
        assert isinstance(input_shape, (list, tuple))
        input_shape = tuple(input_shape)
        assert len(input_shape) == 4
        assert tuple(self._input_shape[1:]) == tuple(input_shape[1:])

        super(Softargmax, self).build(input_shape)

    def call(self, inputs):
        """Call the Softargmax operator at run time.

        Args:
            inputs (Tensor, 4D): Input tensor with the data to process.

        """
        if self._data_format == 'channels_last':
            inputs = K.permute_dimensions(inputs, (0, 3, 1, 2))
        elif self._data_format != 'channels_first':
            raise ValueError(
                'Provide either `channels_first` or `channels_last` for `data_format`.'
            )

        # Shape: (N, C, 1, 1) - Find the maximum pixel value in each channel (corresponding to a
        # key point each).
        max_per_channel = self._reduce_channel(inputs, K.max)

        # Shape: (N, C, H, W) - Shift the original input values down by the maximum value per
        # channel. Results in a 4D tensor with non-positive values.
        normalized = inputs - max_per_channel
        # normalized = inputs

        # Shape: (N, C, H, W) - Multiply all values by the pre-defined beta value and exponentiate
        # them.
        prod_beta = self._beta * normalized
        exp_maps = K.exp(prod_beta)

        # Shape: (N, C, 1, 1) - Sum-reduce all channels to a single value.
        sum_per_channel = self._reduce_channel(exp_maps)

        # Shape: (N, C, 1, 1) - Find the average value per channel through division
        # by the number of pixels per channel.
        # Output value, representing the confidence of each key point.
        # confidence_output = sum_per_channel / (self._height * self._width)

        # Shape: (N, C, H, W) - Softmax operation per channel: Divide all exp_maps by their channel
        # sum. Results in probability values of the key-point location in every pixel (values in
        # [0,1] interval). Each channel sums up to 1.
        prob_maps = exp_maps / (sum_per_channel)

        # confidence_output = tf.math.reduce_max(prob_maps, axis=[2, 3], keepdims=True)
        confidence_output = K.max(prob_maps, axis=[2, 3], keepdims=True)

        # Shape: (N, C, 1, 1) - Multiply the column and row indexes with prob_maps (in batched_dot
        # fashion), respectively. Then sum-reduce them to a single coordinate, corresponding to the
        # weighted location of a key point. Both are output values.
        x_coord_output = Softargmax._reduce_channel(
            self._column_indexes * prob_maps, K.sum)
        y_coord_output = Softargmax._reduce_channel(
            self._row_indexes * prob_maps, K.sum)

        # Shape: (N, C, 3, 1) - Concatenate all output values.
        outputs = K.concatenate(
            [x_coord_output, y_coord_output, confidence_output], axis=2)

        # Shape: (N, C, 3) - Eliminate the redundant dimension.
        outputs = K.squeeze(outputs, axis=3)

        # case when we would like to reshape the outputs at the layer stage
        # outputs_1 = K.reshape(outputs[:,:,:2], (outputs.shape[0].value,-1, 1, 1))
        # outputs_2 = K.reshape(outputs[:,:,2], (outputs.shape[0].value,-1, 1, 1))

        # separating the keypoints and confidence predictions
        outputs_1 = outputs[:, :, :2]
        outputs_2 = outputs[:, :, 2]

        return[outputs_1, outputs_2]

    def compute_output_shape(self, input_shape):
        """Compute the shape of the output tensor produced by the Softargmax operator.

        Args:
            input_shape (4-element list or tuple): Input shape to the Softargmax operator.
        """
        assert isinstance(input_shape, (list, tuple))
        assert len(input_shape) == 4
        if self._data_format == 'channels_first':
            batch_size, nkeypoints, _, _ = input_shape
        elif self._data_format == 'channels_last':
            batch_size, _, _, nkeypoints = input_shape
        else:
            raise ValueError(
                'Provide either `channels_first` or `channels_last` for `data_format`.'
            )

        output_shape_1 = (batch_size, nkeypoints*2, 1, 1)
        output_shape_2 = (batch_size, nkeypoints, 1, 1)

        return [output_shape_1, output_shape_2]

    def get_config(self):
        """Create the config, enabling (de)serialization."""
        config = super(Softargmax, self).get_config()
        config['input_shape'] = self._input_shape
        config['beta'] = self._beta
        config['data_format'] = self._data_format
        return config

    @classmethod
    def _index_initializers(cls, height, width, dtype):
        """Create constant initializers for the x and y locations, respectively.

        Args:
            height (int): Input height to the Softargmax operator.
            width (int): Input width to the Softargmax operator.
            dtype (type): Data type for the initializers.

        """
        col_indexes_per_row = np.arange(0, height, dtype=dtype)
        row_indexes_per_col = np.arange(0, width, dtype=dtype)

        # col_grid gives a column measurement matrix to be used for getting
        # 'x'. It is a matrix where each row has the sequential values starting
        # from 0 up to n_col-1:
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1

        # row_grid gives a row measurement matrix to be used for getting 'y'.
        # It is a matrix where each column has the sequential values starting
        # from 0 up to n_row-1:
        # 0,0,0, ..., 0
        # 1,1,1, ..., 1
        # 2,2,2, ..., 2
        # ...
        # n_row-1, ..., n_row-1

        col_grid, row_grid = np.meshgrid(row_indexes_per_col,
                                         col_indexes_per_row)
        row_index_init = Constant(value=row_grid)
        col_index_init = Constant(value=col_grid)
        return row_index_init, col_index_init

    @classmethod
    def _reduce_channel(cls, inputs, operation=K.sum, keepdims=True):
        """Reduce all channels with the specified operation to a single value.

        Args:
            inputs (Tensor, 4D): Input tensor with the data to reduced.
        Optional args:
            operation (function): Reduce operation to be performed (default: K.sum).
            keepdims (bool): Toggles if 1-dimensions should be kept (default: True).

        """
        reduced_per_row = operation(inputs, axis=2, keepdims=True)
        reduced_per_channel = operation(
            reduced_per_row, axis=3, keepdims=keepdims)
        return reduced_per_channel
