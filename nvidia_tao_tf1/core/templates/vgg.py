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

"""Maglev model templates for VGG16/19."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras.models import Model

from nvidia_tao_tf1.core.templates.utils import arg_scope
from nvidia_tao_tf1.core.templates.utils import CNNBlock


def VggNet(nlayers, inputs, use_batch_norm=False, data_format=None, add_head=False,
           nclasses=None, kernel_regularizer=None, bias_regularizer=None, activation_type='relu',
           use_pooling=True, freeze_bn=False, freeze_blocks=None, use_bias=True,
           dropout=0.5):
    """
    Construct a fixed-depth VggNet, based on the architectures from the original paper [1].

    Args:
        nlayers (int): the number of layers in the desired VGG (e.g. 16, 19).
        inputs (tensor): the input tensor.
        use_batch_norm (bool): whether batchnorm should be added after each convolution.
        data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
        add_head (bool): whether to add the original [1] classification head. Note that if you
            don't include the head, the actual number of layers in the model produced by this
            function is 'nlayers-3`, as they don't include the last 3 FC layers.
        nclasses (int): the number of classes to be added to the classification head. Can be `None`
            if unused.
        kernel_regularizer: regularizer to apply to kernels.
        bias_regularizer: regularizer to apply to biases.
        use_pooling (bool): whether to use MaxPooling2D layer after first conv layer or use a
        stride of 2 for first convolutional layer in subblock
        freeze_bn(bool): Whether or not to freeze the BN layers.
        freeze_blocks(list): the list of blocks in the model to be frozen.
        use_bias(bool): whether or not to use bias for the conv layers.
        dropout(float): The drop rate for dropout.
    Returns:
        Model: the output model after applying the VggNet on top of input `x`.

    [1] Very Deep Convolutional Networks for Large-Scale Image Recognition
        (https://arxiv.org/abs/1409.1556)
    """
    if data_format is None:
        data_format = K.image_data_format()

    if freeze_blocks is None:
        freeze_blocks = []
    # Perform strided convolutions if pooling disabled.
    first_stride = 1
    stride = 2
    if use_pooling:
        # Disable strided convolutions with pooling enabled.
        stride = 1

    # Define a block functor which can create blocks.
    with arg_scope([CNNBlock],
                   use_batch_norm=use_batch_norm,
                   use_shortcuts=False,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   activation_type=activation_type,
                   freeze_bn=freeze_bn,
                   use_bias=use_bias):
        # Implementing VGG 16 architecture.
        if nlayers == 16:
            # Block - 1.
            x = CNNBlock(repeat=2, stride=first_stride, subblocks=[(3, 64)], index=1,
                         freeze_block=(1 in freeze_blocks))(inputs)
            if use_pooling:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 data_format=data_format, name='block1_pool')(x)
            # Block - 2.
            x = CNNBlock(repeat=2, stride=stride, subblocks=[(3, 128)], index=2,
                         freeze_block=(2 in freeze_blocks))(x)
            if use_pooling:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 data_format=data_format, name='block2_pool')(x)
            # Block - 3.
            x = CNNBlock(repeat=3, stride=stride, subblocks=[(3, 256)], index=3,
                         freeze_block=(3 in freeze_blocks))(x)
            if use_pooling:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 data_format=data_format, name='block3_pool')(x)
            # Block - 4.
            x = CNNBlock(repeat=3, stride=stride, subblocks=[(3, 512)], index=4,
                         freeze_block=(4 in freeze_blocks))(x)
            if use_pooling:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 data_format=data_format, name='block4_pool')(x)
            # Block - 5.
            x = CNNBlock(repeat=3, stride=stride, subblocks=[(3, 512)], index=5,
                         freeze_block=(5 in freeze_blocks))(x)
        # Implementing VGG 19 architecture.
        elif nlayers == 19:
            # Block - 1.
            x = CNNBlock(repeat=2, stride=first_stride, subblocks=[(3, 64)], index=1,
                         freeze_block=(1 in freeze_blocks))(inputs)
            if use_pooling:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 data_format=data_format, name='block1_pool')(x)
            # Block - 2.
            x = CNNBlock(repeat=2, stride=stride, subblocks=[(3, 128)], index=2,
                         freeze_block=(2 in freeze_blocks))(x)
            if use_pooling:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 data_format=data_format, name='block2_pool')(x)
            # Block - 3.
            x = CNNBlock(repeat=4, stride=stride, subblocks=[(3, 256)], index=3,
                         freeze_block=(3 in freeze_blocks))(x)
            if use_pooling:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 data_format=data_format, name='block3_pool')(x)
            # Block - 4.
            x = CNNBlock(repeat=4, stride=stride, subblocks=[(3, 512)], index=4,
                         freeze_block=(4 in freeze_blocks))(x)
            if use_pooling:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                 data_format=data_format, name='block4_pool')(x)
            # Block - 5.
            x = CNNBlock(repeat=4, stride=stride, subblocks=[(3, 512)], index=5,
                         freeze_block=(5 in freeze_blocks))(x)
        else:
            raise NotImplementedError('A VGG with nlayers=%d is not implemented.' % nlayers)

    if add_head:
        # Add final Max Pooling layer if there are FC layers. Otherwise return the
        # feature extractor trunk with a stride of 16
        if use_pooling:
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                             data_format=data_format, name='block5_pool')(x)
        # Classification block.
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        x = Dense(nclasses, activation='softmax', name='output_fc')(x)

    # Naming model.
    model_name = 'vgg%d' % nlayers
    if not use_pooling:
        model_name += '_nopool'
    if use_batch_norm:
        model_name += '_bn'
    # Set up keras model object.
    model = Model(inputs=inputs, outputs=x, name=model_name)

    return model
