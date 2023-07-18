# Copyright (channels) 2020, NVIDIA CORPORATION.  All rights reserved.
""" Test for the Softargmax operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.models import Model

import numpy as np
import pytest

from nvidia_tao_tf1.cv.fpenet.models.custom.softargmax import Softargmax


def softargmax_numpy(input_vals,
                     beta,
                     epsilon=1e-6,
                     data_format='channels_first'):
    """A NumPy implementation of Softargmax.

    Args:
        input_vals (numpy.ndarray, 4d): Input values to be processed.
        beta (float): Coefficient used for multiplying the key-point maps after
                subtracting the channel-wise maximum.
    Optional args:
        epsilon (float): Epsilon added to the denominator of the Softmax-like operation
            per channel (default value taken from original implementation in Theano).
        data_format (str): Expected tensor format, either 'channels_first' or 'channels_last'.
            Default value is 'channels_first' because 'channels_last' is not implemented.
    """

    if data_format == 'channels_last':
        input_vals = input_vals.transpose(0, 3, 1, 2)
    elif data_format != 'channels_first':
        raise ValueError(
            'Provide either `channels_first` or `channels_last` for `data_format`.'
        )

    # Shape: (batch_size, channels, height, width)
    batch_size, channels, height, width = input_vals.shape

    # Cast inputs to float32 precision:
    # input_vals = input_vals.astype('float32')
    n_row, n_col = height, width
    n_kpts = channels

    # Shape: (batch_size, channels, height*width)
    input_3d = input_vals.reshape(batch_size, channels, height * width)

    # Shape: (batch_size, channels) - Find the maximum pixel value in each channel (corresponding to
    # a key-point
    # each).
    map_max = input_3d.max(axis=2)

    # Shape: (batch_size, channels, 1) - Achieve the same number of dimensions as input_3d so that
    # we can calculate the difference.
    map_max_3d = np.expand_dims(map_max, axis=-1)

    # Shape: (batch_size, channels, height*width) - Shift the original input values down by the
    # maximum value, achieving a 3d tensor with non-positive values.
    input_3d = input_3d - map_max_3d

    # Everything in this section can be implemented with a standard Softmax call from cuDNN or
    # TensorRT:
    ###############################################################################################
    # Shape: (batch_size, channels, height*width) - Multiply each (non-positive) value from input_3d
    # with beta. For hand-pose network, beta = 0.1.
    product_beta = np.multiply(input_3d, beta)

    # Shape: (batch_size, channels, height*width) - See activation value range here:
    # https://www.wolframalpha.com/input/?i=exp(0.1x).
    # Maximum value of exp_maps is 1 (at location of maximum from input_3d)because inputs are
    # non-positive.
    # Runtime on Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz: 1.25 ms.
    exp_maps = np.exp(product_beta)
    # print(exp_maps.reshape(batch_size,channels,height,width)[0,0,:,:2])
    probs = exp_maps.mean(axis=2)

    # Shape: (batch_size, channels) - Sum of all values along positional dimension.
    exp_maps_sum = np.sum(exp_maps, axis=2)
    # print(exp_maps.reshape(batch_size,channels,height,width).[0,0,:,:2])

    # Shape: (batch_size, channels), output matrix, third output index of the layer.
    # z_vals = exp_maps_sum

    # Shape after loop: (batch_size, channels, 1, 1) - Achieve 4d representation for element-wise
    # division.
    input_3d_sum_4d = exp_maps_sum
    for _ in range(2):
        input_3d_sum_4d = np.expand_dims(input_3d_sum_4d, axis=-1)

    # Shape: (batch_size, channels, height, width) - Achieve 4d representation for element-wise
    # division.
    exp_maps_reshaped = exp_maps.reshape([-1, n_kpts, n_row, n_col])

    # Shape: (batch_size, channels, 1, 1) - Add epsilon to prevent division by zero.
    input_3d_sum_4d_epsilon = np.add(input_3d_sum_4d, epsilon)

    # Shape: (batch_size, channels, height, width) - Divide each element by the sum.
    # Similar to classical Softmax. Resulting in float values between 0 and 1 that can be
    # interpreted as probabilities.
    # Runtime on Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz: 0.4 ms
    normalized_maps_4d = np.divide(exp_maps_reshaped, input_3d_sum_4d_epsilon)
    ###############################################################################################
    # Shape: (batch_size, channels, height, width), output tensor, fourth and last output index of
    # the layer
    # z_maps = normalized_maps_4d

    col_vals = np.arange(n_col, dtype=input_vals.dtype)
    col_repeat = np.tile(col_vals, n_row)
    # Shape: (1, 1, height, width)
    col_idx = col_repeat.reshape(1, 1, n_row, n_col)
    # col_mat gives a column measurement matrix to be used for getting
    # 'x'. It is a matrix where each row has the sequential values starting
    # from 0 up to n_col-1:
    # 0,1,2, ..., n_col-1
    # 0,1,2, ..., n_col-1
    # 0,1,2, ..., n_col-1

    row_vals = np.arange(n_row, dtype=input_vals.dtype)
    row_repeat = np.repeat(row_vals, n_col)
    # Shape: (1, 1, height, width)
    row_idx = row_repeat.reshape(1, 1, n_row, n_col)
    # row_mat gives a row measurement matrix to be used for getting 'y'.
    # It is a matrix where each column has the sequential values starting
    # from 0 up to n_row-1:
    # 0,0,0, ..., 0
    # 1,1,1, ..., 1
    # 2,2,2, ..., 2
    # ...
    # n_row-1, ..., n_row-1

    # Shape: (batch_size, channels, height, width)
    # Get a probability-weighted column index.
    # Runtime on Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz: 0.5 ms
    weighted_x = np.multiply(normalized_maps_4d, col_idx)
    # Shape: (batch_size, channels, height*width)
    # Reshape for sum operation
    weighted_x_3d = weighted_x.reshape(batch_size, channels, height * width)

    # Shape: (batch_size, channels)
    # Calculate weighted sum of X coordinates for each key-point.
    # Output matrix, first output index of the layer
    x_vals = np.sum(weighted_x_3d, axis=2)
    # Shape: (batch_size, channels, height, width)
    # Get a probability-weighted row index.
    # Runtime on Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz: 0.5 ms
    weighted_y = np.multiply(normalized_maps_4d, row_idx)
    # Shape: (batch_size, channels, height*width)
    # Reshape for sum operation
    weighted_y_3d = weighted_y.reshape(batch_size, channels, height * width)

    # Shape: (batch_size, channels), output matrix, second output index of the layer
    # Calculate weighted sum of Y coordinates for each key-point.
    y_vals = np.sum(weighted_y_3d, axis=2)

    outputs = np.stack((x_vals, y_vals, probs), axis=1)
    outputs = np.transpose(outputs, (0, 2, 1))

    # keypoints values
    outputs1 = outputs[:, :, :2]
    # confidence values
    outputs2 = outputs[:, :, 2]

    return outputs1, outputs2


def calculate_absdiff(tensor_a, tensor_b):
    """Calculate the absolute difference between two tensors.

    Args:
        tensor_a (numpy.ndarray): The first tensor.
        tensor_b (numpy.ndarray): The second tensor.
    """
    assert hasattr(tensor_a, 'shape') and hasattr(tensor_b, 'shape')
    assert tensor_a.shape == tensor_b.shape
    diff = tensor_a - tensor_b
    absdiff = np.abs(diff)
    return absdiff


def create_softargmax_model(input_shape, beta, data_format):
    """Create a Keras model consisting of a single Softargmax layer.

    Args:
        input_shape (4-element list or tuple): Input shape in the specified data format.
        beta (float): Coefficient used for multiplying the key-point maps after
            subtracting the channel-wise maximum.
        data_format (str): Expected tensor format, either 'channels_first' or 'channels_last'.

    """
    input_shape_without_batch = input_shape[1:]
    inputs = Input(name='input', shape=input_shape_without_batch)
    softargmax = Softargmax(
        input_shape, beta=beta, data_format=data_format)(inputs)
    model = Model(inputs=inputs, outputs=softargmax)
    return model


@pytest.mark.parametrize(
    'batch_size, nkeypoints, height, width, beta, data_format',
    [(1, 68, 80, 80, 0.1, 'channels_first'),
     (128, 68, 80, 80, 0.1, 'channels_first'),
     (3, 21, 99, 40, 0.5, 'channels_last')])
def test_softargmax(batch_size,
                    nkeypoints,
                    height,
                    width,
                    beta,
                    data_format,
                    acceptable_diff=1e-4):
    """ Test the Softargmax implementation in Keras against a NumPy implementation.

    Args:
        batch_size, nkeypoints, height, width: Input dimensions to be processed.
        beta (float): Coefficient used for multiplying the key-point maps after
                subtracting the channel-wise maximum.
    Optional args:
        data_format (str): Expected tensor format, either 'channels_first' or 'channels_last'.
        acceptable_diff (float): Indicates the maximum acceptable difference value between the
            Keras prediction and the NumPy prediction.
    """

    if data_format == 'channels_first':
        input_shape = (batch_size, nkeypoints, height, width)
    elif data_format == 'channels_last':
        input_shape = (batch_size, height, width, nkeypoints)
    else:
        raise ValueError(
            'Provide either `channels_first` or `channels_last` for `data_format`.'
        )

    model = create_softargmax_model(input_shape, beta, data_format)

    model.compile(optimizer='rmsprop', loss='mse')

    input_vals = np.random.rand(batch_size, nkeypoints, height, width)
    input_vals = input_vals.astype('float32')

    if data_format == 'channels_last':
        input_vals = input_vals.transpose(0, 2, 3, 1)
    elif data_format != 'channels_first':
        raise ValueError(
            'Provide either `channels_first` or `channels_last` for `data_format`.'
        )

    prediction_keypoints, prediction_confidence = model.predict(input_vals)
    assert hasattr(prediction_confidence, 'shape')
    epsilon_numpy = 1e-6
    prediction_numpy_keypoints, prediction_numpy_confidence = softargmax_numpy(
        input_vals, beta=beta, epsilon=epsilon_numpy, data_format=data_format)
    assert hasattr(prediction_numpy_confidence, 'shape')
    absdiff = calculate_absdiff(prediction_keypoints, prediction_numpy_keypoints)
    max_absdiff = np.max(absdiff)
    assert max_absdiff < acceptable_diff, 'The acceptable maximum absolute difference between \
    Numpy and Keras prediction exceeds the specified threshold.'
