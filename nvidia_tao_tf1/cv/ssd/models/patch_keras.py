# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Patch keras's conv2d and pool2d for handling floor mode in downsampling with kernel_size=2."""

import keras


def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        _padding = (filter_size - 1) // 2
        return (input_length + 2 * _padding - dilated_filter_size) // stride + 1

    if padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def conv_compute_output_shape(self, input_shape):
    """Compute the output dimension of a convolution."""
    if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)
    if self.data_format == 'channels_first':
        space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0], self.filters) + tuple(new_space)
    return None


def pool_compute_output_shape(self, input_shape):
    """Compute the output dimension of a pooling."""
    if self.data_format == 'channels_first':
        rows = input_shape[2]
        cols = input_shape[3]
    elif self.data_format == 'channels_last':
        rows = input_shape[1]
        cols = input_shape[2]
    rows = conv_output_length(rows, self.pool_size[0],
                              self.padding, self.strides[0])
    cols = conv_output_length(cols, self.pool_size[1],
                              self.padding, self.strides[1])
    if self.data_format == 'channels_first':
        return (input_shape[0], input_shape[1], rows, cols)
    if self.data_format == 'channels_last':
        return (input_shape[0], rows, cols, input_shape[3])
    return None


def patch():
    """Apply the patches to the module."""
    keras.layers.MaxPooling2D.compute_output_shape = pool_compute_output_shape
    keras.layers.Conv2D.compute_output_shape = conv_compute_output_shape
