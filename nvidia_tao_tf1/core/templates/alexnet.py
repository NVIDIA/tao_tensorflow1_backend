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
"""Modulus model templates for alexnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from nvidia_tao_tf1.core.models.import_keras import keras as keras_fn

keras = keras_fn()
K = keras.backend


def AlexNet(input_shape,
            nclasses=1000,
            data_format=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            add_head=True,
            weights=None,
            hidden_fc_neurons=4096,
            freeze_blocks=None):
    """
    Construct AlexNet with/without dense head layers.

    Args:
        input_shape (tuple): shape of the input image. The shape must be
            provided as per the data_format input for the model.
            (C, W, H) for channels_first or (W, H, C for
            channels last), where
            C = number of channels,
            W = width of image,
            H = height of image.
        nclasses (int): number of output classes (defaulted to 1000 outputs)
        data_format (str): either 'channels last' or 'channels_first'
        kernel_regularizer (keras.regularizer attribute): Regularization type
            for kernels.
            keras.regularizer.l1(wd) or,
            keras.regularizer.l2(wd) where,
            wd = weight_decay.
        bias_regularizer (keras.regularizer attribute): Regularization type
            for biases.
            keras.regularizer.l1(wd) or,
            keras.regularizer.l2(wd) where,
            wd = weight_decay.
        add_head (bool) : whether to add FC layer heads to the model or not.
            If 'False', the network will not have to FC-6 to FC-8 dense layers.
            If 'True' , the network will have the FC layers appended to it
        weights (str) = path to the pretrained weights .h5 file
        hidden_fc_neurons (int): number of neurons in hidden fully-connected
            layers. The original AlexNet has 4096 of those but a smaller number
            can be used to build a more parsimonious model.
        freeze_blocks(list): the list of blocks to be frozen in the model.

    Returns:
        Model: The output model after applying Alexnet on input 'x'
    """
    if data_format is None:
        data_format = K.image_data_format()

    if data_format not in ["channels_first", "channels_last"]:
        raise ValueError("Unsupported data_format (%s)" % data_format)

    if freeze_blocks is None:
        freeze_blocks = []
    # Input layer
    input_image = keras.layers.Input(shape=(input_shape))

    # Conv block 1
    conv1 = keras.layers.Conv2D(
        96,
        kernel_size=(11, 11),
        strides=(4, 4),
        data_format=data_format,
        name='conv1',
        kernel_regularizer=kernel_regularizer,
        padding='same',
        bias_regularizer=bias_regularizer,
        activation='relu',
        trainable=not(1 in freeze_blocks))(input_image)
    conv1 = keras.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same', data_format=data_format,
        name='pool1')(conv1)
    # Conv block 2
    conv2 = keras.layers.Conv2D(
        256, (5, 5),
        strides=(1, 1),
        data_format=data_format,
        name='conv2',
        kernel_regularizer=kernel_regularizer,
        padding='same',
        bias_regularizer=bias_regularizer,
        activation='relu',
        trainable=not(2 in freeze_blocks))(conv1)
    conv2 = keras.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same', data_format=data_format,
        name='pool2')(conv2)
    # 'Conv block 3'
    conv3 = keras.layers.Conv2D(
        384, (3, 3),
        strides=(1, 1),
        data_format=data_format,
        name='conv3',
        kernel_regularizer=kernel_regularizer,
        padding='same',
        bias_regularizer=bias_regularizer,
        activation='relu',
        trainable=not(3 in freeze_blocks))(conv2)
    # 'Conv block 4'
    conv4 = keras.layers.Conv2D(
        384, (3, 3),
        strides=(1, 1),
        data_format=data_format,
        name='conv4',
        kernel_regularizer=kernel_regularizer,
        padding='same',
        bias_regularizer=bias_regularizer,
        activation='relu',
        trainable=not(4 in freeze_blocks))(conv3)
    # 'Conv block 5'
    x = keras.layers.Conv2D(
        256, (3, 3),
        strides=(1, 1),
        data_format=data_format,
        name='conv5',
        kernel_regularizer=kernel_regularizer,
        padding='same',
        bias_regularizer=bias_regularizer,
        activation='relu',
        trainable=not(5 in freeze_blocks))(conv4)
    # 'FC Layers'
    if add_head:
        conv5 = keras.layers.Flatten(name='flatten')(x)
        # FC - 6
        fc6 = keras.layers.Dense(
            hidden_fc_neurons,
            name='fc6',
            activation='relu',
            kernel_initializer='glorot_uniform',
            use_bias=True,
            bias_initializer='zeros',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=not(6 in freeze_blocks))(conv5)
        # FC - 7
        fc7 = keras.layers.Dense(
            hidden_fc_neurons,
            name='fc7',
            activation='relu',
            kernel_initializer='glorot_uniform',
            use_bias=True,
            bias_initializer='zeros',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=not(7 in freeze_blocks))(fc6)
        # FC - 8
        x = keras.layers.Dense(
            nclasses,
            activation='softmax',
            name='head_fc8',
            kernel_initializer='glorot_uniform',
            use_bias=True,
            bias_initializer='zeros',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=not(8 in freeze_blocks))(fc7)

    # Setting up graph
    model = keras.models.Model(inputs=input_image, outputs=x, name='AlexNet')

    # Loading pretrained weights if mentioned
    if weights is not None:
        if os.path.exists(weights):
            model.load_weights(weights, by_name=True)

    return model
