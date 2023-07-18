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

# -*- coding: utf-8 -*-
"""Contains a set of utilities that allow building the UNet model."""

import keras
from keras import backend as K
from keras.layers import BatchNormalization
import numpy as np
import tensorflow as tf

keras.backend.set_image_data_format('channels_first')


def downsample_block(inputs, filters):
    """UNet downsample block.

    Perform 2 unpadded convolutions with a specified number of filters and downsample
    through max-pooling

    Args:
    inputs (tf.Tensor): Tensor with inputs
    filters (int): Number of filters in convolution

    Return:
    Tuple of convolved ``inputs`` after and before downsampling

    """

    out = inputs
    out = keras.layers.Conv2D(
                           filters=filters,
                           kernel_size=(3, 3),
                           activation=tf.nn.relu)(out)
    out = keras.layers.Conv2D(
                           filters=filters,
                           kernel_size=(3, 3),
                           activation=tf.nn.relu)(out)
    out_pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(out)
    return out_pool, out


def conv_kernel_initializer(shape, dtype=K.floatx()):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
    Args:
        shape: shape of variable
        dtype: dtype of variable
    Returns:
        an initialization for the variable
    """
    kernel_height, kernel_width, _ , out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def upsample_block(inputs, residual_input, filters):
    """UNet upsample block.

    Perform 2 unpadded convolutions with a specified number of filters and upsample

    Args:
        inputs (tf.Tensor): Tensor with inputs
        residual_input (tf.Tensor): Residual input
        filters (int): Number of filters in convolution

    Return:
       Convolved ``inputs`` after upsampling
    """

    cropped = keras.layers.Cropping2D(((filters[1], filters[1]), (filters[1],
                                      filters[1])), data_format="channels_first")(residual_input)
    concat_axis = 3 if K.image_data_format() == 'channels_last' else 1
    out = keras.layers.Concatenate(axis=concat_axis)([inputs, cropped])
    out = keras.layers.Conv2D(
                           filters=filters[0],
                           kernel_size=(3, 3),
                           activation=tf.nn.relu)(out)
    out = keras.layers.Conv2D(
                           filters=int(filters[0]),
                           kernel_size=(3, 3),
                           activation=tf.nn.relu)(out)

    out = keras.layers.Conv2DTranspose(
                                      filters=int(filters[2]),
                                      kernel_size=(3, 3),
                                      strides=(2, 2),
                                      padding='same',
                                      activation=tf.nn.relu)(out)
    return out


def conv2D_bn(x, filters, use_batchnorm, freeze_bn, kernel_size=(3, 3),
              kernel_initializer='glorot_uniform', activation=None):
    """UNet conv2D_bn.

    Perform convolution followed by BatchNormalization

    Args:
        x (tf.Tensor): Tensor with inputs
        use_batchnorm (bool): Flag to set the batch normalization.
        filters (int): Number of filters in convolution
        kernel_initializer (str): Initialization of layer.
        kernel_size (tuple): Size of filter

    Return:
       Convolved ``inputs`` after convolution.
    """

    x = keras.layers.Conv2D(
                           filters=filters,
                           kernel_size=kernel_size,
                           activation=None, padding='same',
                           kernel_initializer=kernel_initializer)(x)
    if use_batchnorm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(axis=1)(x, training=False)
        else:
            x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.Activation('relu')(x)

    return x


def upsample_block_other(inputs, residual_input, filters, use_batchnorm=False,
                         freeze_bn=False, initializer='glorot_uniform'):
    """UNet upsample block.

    Perform 2 unpadded convolutions with a specified number of filters and upsample

    Args:
        inputs (tf.Tensor): Tensor with inputs
        residual_input (tf.Tensor): Residual input
        filters (int): Number of filters in convolution
        freeze_bn (bool): Flag to freeze batch norm.
    Return:
       Convolved ``inputs`` after upsampling.

    """
    x = keras.layers.Conv2DTranspose(
                                      filters=int(filters),
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding='same',
                                      activation=None,
                                      data_format="channels_first",
                                      kernel_initializer=initializer)(inputs)
    concat_axis = 3 if K.image_data_format() == 'channels_last' else 1
    if residual_input is not None:
        x = keras.layers.Concatenate(axis=concat_axis)([x, residual_input])
    if use_batchnorm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(axis=1)(x, training=False)
        else:
            x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(
                           filters=filters,
                           kernel_size=(3, 3),
                           activation=None, padding='same',
                           kernel_initializer=initializer)(x)
    if use_batchnorm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(axis=1)(x, training=False)
        else:
            x = keras.layers.BatchNormalization(axis=1)(x)

    x = keras.layers.Activation('relu')(x)

    return x


def bottleneck(inputs, filters_up, mode):
    """UNet central block.

    Perform 2 unpadded convolutions with a specified number of filters and upsample
    including dropout before upsampling for training

    Args:
        inputs (tf.Tensor): Tensor with inputs
        filters_up (int): Number of filters in convolution

    Return:
        Convolved ``inputs`` after bottleneck.

    """
    out = inputs
    out = keras.layers.Conv2DTranspose(
                                      filters=filters_up,
                                      kernel_size=(3, 3),
                                      strides=(2, 2),
                                      padding='same',
                                      activation=tf.nn.relu)(out)
    return out


def output_block(inputs, residual_input, filters, n_classes):
    """UNet output.

    Perform 3 unpadded convolutions, the last one with the same number
    of channels as classes we want to classify

    Args:
        inputs (tf.Tensor): Tensor with inputs
        residual_input (tf.Tensor): Residual input
        filters (int): Number of filters in convolution
        n_classes (int): Number of output classes

    Return:
        Convolved ``inputs`` with as many channels as classes

    """
    crop_pad = (K.int_shape(residual_input)[2] - K.int_shape(inputs)[2])//2
    cropped = keras.layers.Cropping2D(((crop_pad, crop_pad), (crop_pad,
                                      crop_pad)), data_format="channels_first")(residual_input)
    concat_axis = 3 if K.image_data_format() == 'channels_last' else 1
    out = keras.layers.Concatenate(axis=concat_axis)([inputs, cropped])

    out = keras.layers.Conv2D(
                           filters=filters,
                           kernel_size=(3, 3),
                           activation=tf.nn.relu)(out)
    out = keras.layers.Conv2D(
                           filters=filters,
                           kernel_size=(3, 3),
                           activation=tf.nn.relu)(out)

    return keras.layers.Conv2D(
                            filters=n_classes,
                            kernel_size=(1, 1),
                            activation=None)(out)


def output_block_other(inputs, n_classes):
    """UNet output.

    Perform 3 unpadded convolutions, the last one with the same number
    of channels as classes we want to classify

    Args:
        inputs (tf.Tensor): Tensor with inputs
        residual_input (tf.Tensor): Residual input
        filters (int): Number of filters in convolution
        n_classes (int): Number of output classes

    Return:
        Convolved ``inputs`` with as many channels as classes

    """

    return keras.layers.Conv2D(filters=n_classes,
                               kernel_size=(3, 3),
                               padding='same',
                               activation=None)(inputs)


def input_block(inputs, filters):
    """UNet input block.

    Perform 2 unpadded convolutions with a specified number of filters and downsample
    through max-pooling. First convolution

    Args:
        inputs (tf.Tensor): Tensor with inputs
        filters (int): Number of filters in convolution

    Return:
        Tuple of convolved ``inputs`` after and before downsampling

    """

    out = inputs
    out = keras.layers.Conv2D(
                           filters=filters,
                           kernel_size=(3, 3),
                           activation=tf.nn.relu)(out)
    out = keras.layers.Conv2D(
                           filters=filters,
                           kernel_size=(3, 3),
                           activation=tf.nn.relu)(out)
    out_pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(out)
    return out_pool, out


def conv_block(input_tensor, num_filters, use_batch_norm, freeze_bn,
               initializer='glorot_uniform'):
    """UNet conv block.

    Perform convolution followed by BatchNormalization.

    Args:
        inputs (tf.Tensor): Tensor with inputs
        filters (int): Number of filters in convolution

    Return:
        Tuple of convolved ``inputs`` after and before downsampling

    """
    encoder = keras.layers.Conv2D(num_filters, (3, 3), padding="same",
                                  kernel_initializer=initializer)(input_tensor)
    if use_batch_norm:
        if freeze_bn:
            encoder = BatchNormalization(axis=1)(encoder, training=False)
        else:
            encoder = BatchNormalization(axis=1)(encoder)
    encoder = keras.layers.Activation("relu")(encoder)
    encoder = keras.layers.Conv2D(num_filters, (3, 3), padding="same",
                                  kernel_initializer=initializer)(encoder)
    if use_batch_norm:
        if freeze_bn:
            encoder = BatchNormalization(axis=1)(encoder, training=False)
        else:
            encoder = BatchNormalization(axis=1)(encoder)
    encoder = keras.layers.Activation("relu")(encoder)
    return encoder


def encoder_block(input_tensor, num_filters, block_idx, use_batch_norm,
                  freeze_bn, initializer):
    """UNet encoder block.

    Perform convolution followed by BatchNormalization.

    Args:
        input_tensor (tf.Tensor): Tensor with inputs
        num_filters (int): Number of filters in convolution

    Return:
        Tuple of convolved ``inputs`` after and before downsampling

    """
    encoder = conv_block(input_tensor, num_filters, use_batch_norm, freeze_bn,
                         initializer)
    encoder_pool = keras.layers.MaxPool2D((2, 2), strides=(2, 2),
                                          data_format='channels_first')(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters, block_idx, use_batch_norm,
                  freeze_bn, initializer):
    """UNet decoder block.

    Perform convolution followed by BatchNormalization.

    Args:
        input_tensor (tf.Tensor): Tensor with inputs
        num_filters (int): Number of filters in convolution
        concat_tensor (tensor): Tensor to concatenate.

    Return:
        Tuple of convolved ``inputs`` after and before downsampling

    """

    concat_axis = 3 if K.image_data_format() == 'channels_last' else 1
    decoder = keras.layers.Conv2DTranspose(num_filters, (2, 2),
                                           strides=(2, 2), padding='same')(input_tensor)
    decoder = keras.layers.Concatenate(axis=concat_axis)([concat_tensor, decoder])
    if use_batch_norm:
        if freeze_bn:
            decoder = BatchNormalization(axis=1)(decoder, training=False)
        else:
            decoder = BatchNormalization(axis=1)(decoder)
    decoder = keras.layers.Activation("relu")(decoder)
    decoder = keras.layers.Conv2D(num_filters, (3, 3), padding="same",
                                  kernel_initializer=initializer)(decoder)
    if use_batch_norm:
        if freeze_bn:
            decoder = BatchNormalization(axis=1)(decoder, training=False)
        else:
            decoder = BatchNormalization(axis=1)(decoder)
    decoder = keras.layers.Activation("relu")(decoder)
    decoder = keras.layers.Conv2D(num_filters, (3, 3), padding="same",
                                  kernel_initializer=initializer)(decoder)
    if use_batch_norm:
        if freeze_bn:
            decoder = BatchNormalization(axis=1)(decoder, training=False)
        else:
            decoder = BatchNormalization(axis=1)(decoder)
    decoder = keras.layers.Activation("relu")(decoder)

    return decoder


def Conv2DTranspose_block(input_tensor, filters, kernel_size=(3, 3),
                          transpose_kernel_size=(2, 2), upsample_rate=(2, 2),
                          initializer='glorot_uniform', skip=None,
                          use_batchnorm=False, freeze_bn=False):
    """UNet Conv2DTranspose_block.

    Perform convolution followed by BatchNormalization.

    Args:
        input_tensor (tf.Tensor): Tensor with inputs
        filters (int): The filters to be used for convolution.
        transpose_kernel_size (int): Kernel size.
        skip (Tensor): Skip tensor to be concatenated.
        use_batchnorm (Bool): Flag to use batch norm or not.
        freeze_bn (Bool): Flaf to freeze the Batchnorm/ not.

    Return:
        Tuple of convolved ``inputs`` after and before downsampling

    """
    x = keras.layers.Conv2DTranspose(filters, transpose_kernel_size,
                                     strides=upsample_rate, padding='same',
                                     data_format="channels_first")(input_tensor)
    concat_axis = 3 if K.image_data_format() == 'channels_last' else 1
    if skip is not None:
        x = keras.layers.Concatenate(axis=concat_axis)([x, skip])
    x = DoubleConv(x, filters, kernel_size, initializer=initializer,
                   use_batchnorm=use_batchnorm, freeze_bn=freeze_bn)

    return x


def DoubleConv(x, filters, kernel_size, initializer='glorot_uniform',
               use_batchnorm=False, freeze_bn=False):
    """UNet DoubleConv.

    Perform convolution followed by BatchNormalization.

    Args:
        x (tf.Tensor): Tensor with inputs
        kernel_size (int): Size of filter.
        initializer (str): The intization of layer.
        use_batchnorm (Bool): Flag to use batch norm or not.
        freeze_bn (Bool): Flaf to freeze the Batchnorm/ not.

    Return:
        Tuple of convolved ``inputs`` after and before downsampling

    """
    x = keras.layers.Conv2D(filters, kernel_size, padding='same',
                            use_bias=False, kernel_initializer=initializer)(x)
    if freeze_bn:
        x = BatchNormalization(axis=1)(x, training=False)
    else:
        x = BatchNormalization(axis=1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False,
                            kernel_initializer=initializer)(x)
    if use_batchnorm:
        if freeze_bn:
            x = BatchNormalization(axis=1)(x, training=False)
        else:
            x = BatchNormalization(axis=1)(x)

    x = keras.layers.Activation('relu')(x)

    return x


def convTranspose2D_bn(x, filters, use_batchnorm, freeze_bn,
                       kernel_initializer='glorot_uniform', activation=None, padding='None'):
    """UNet convTranspose2D_bn.

    Perform transposed convolution followed by BatchNormalization

    Args:
        x (tf.Tensor): Tensor with inputs
        use_batchnorm (bool): Flag to set the batch normalization.
        filters (int): Number of filters in convolution
        kernel_initializer (str): Initialization of layer.
        activation:

    Return:
       Transposed convolution ``inputs``.
    """

    x = keras.layers.Conv2DTranspose(
                                      filters=filters,
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding='same',
                                      activation=None,
                                      data_format="channels_first")(x)
    if use_batchnorm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(axis=1, epsilon=1e-2)(x, training=False)
        else:
            x = keras.layers.BatchNormalization(axis=1, epsilon=1e-2)(x)

    return x
