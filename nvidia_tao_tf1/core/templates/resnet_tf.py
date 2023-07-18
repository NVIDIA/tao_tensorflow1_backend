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
"""Modulus model templates for ResNets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras

from nvidia_tao_tf1.core.templates.utils_tf import add_activation
from nvidia_tao_tf1.core.templates.utils_tf import add_dense_head
from nvidia_tao_tf1.core.templates.utils_tf import arg_scope
from nvidia_tao_tf1.core.templates.utils_tf import CNNBlock
from nvidia_tao_tf1.core.templates.utils_tf import get_batchnorm_axis


def ResNet(nlayers,
           input_tensor=None,
           use_batch_norm=False,
           data_format='channels_first',
           add_head=False,
           head_activation='softmax',
           nclasses=None,
           kernel_regularizer=None,
           bias_regularizer=None,
           activation_type='relu',
           activation_kwargs=None,
           all_projections=True,
           freeze_blocks=None,
           freeze_bn=False,
           use_pooling=False,
           use_bias=False):
    """
    Construct a fixed-depth vanilla ResNet, based on the architectures from the original paper [1].

    Args:
        nlayers (int): the number of layers in the desired ResNet (e.g. 18, 34, ..., 152).
        input_tensor (tensor): the input tensor.
        use_batch_norm (bool): whether batchnorm should be added after each convolution.
        data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
        add_head (bool): whether to add the original [1] classification head. Note that if you
            don't include the head, the actual number of layers in the model produced by this
            function is 'nlayers-1`.
        head_activation (string): Activation function for classification head.
        nclasses (int): the number of classes to be added to the classification head. Can be `None`
            if unused.
        kernel_regularizer: regularizer to apply to kernels.
        bias_regularizer: regularizer to apply to biases.
        activation_type (str): Type of activation.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        all_projections (bool): whether to implement cnn subblocks with all shortcuts connections
            forced as 1x1 convolutional layers as mentioned in [1] to enable full pruning of
            ResNets. If set as False, the template instantiated will be the classic ResNet template
            as in [1] with shortcut connections as skip connections when there is no stride change
            and 1x1 convolutional layers (projection layers) when there is a stride change.
            Note: The classic template cannot be fully pruned. Only the first N-1 number of layers
            in the ResNet subblock can be pruned. All other layers must be added to exclude layers
            list while pruning, including conv1 layer.
        freeze_bn(bool): Whether or not to freeze the BN layers.
        freeze_blocks(list): the list of blocks in the model to be frozen.
        use_pooling (bool): whether to use MaxPooling2D layer after first conv layer or use a
        stride of 2 for first convolutional layer in subblock
        use_bias(bool): Whether or not to use bias for the conv layers.
    Returns:
        Model: the output model after applying the ResNet on top of input `x`.

    [1] Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
    """
    if freeze_blocks is None:
        freeze_blocks = []
    # Determine proper input shape
    if data_format == 'channels_first':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        inputs = keras.layers.Input(shape=input_shape)
    else:
        inputs = input_tensor

    freeze0 = 0 in freeze_blocks
    freeze1 = 1 in freeze_blocks
    freeze2 = 2 in freeze_blocks
    freeze3 = 3 in freeze_blocks
    freeze4 = 4 in freeze_blocks

    activation_kwargs = activation_kwargs or {}

    x = keras.layers.Conv2D(64, (7, 7),
                            strides=(2, 2),
                            padding='same',
                            data_format=data_format,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            name='conv1',
                            trainable=not freeze0,
                            use_bias=use_bias)(inputs)

    if use_batch_norm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(axis=get_batchnorm_axis(data_format),
                                                name='bn_conv1')(x, training=False)
        else:
            x = keras.layers.BatchNormalization(axis=get_batchnorm_axis(data_format),
                                                name='bn_conv1')(x)

    x = add_activation(activation_type, **activation_kwargs)(x)
    first_stride = 2  # Setting stride 1st convolutional subblock.
    last_stride = 1  # Setting stride last convolutional subblock.
    if use_pooling:
        x = keras.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=(2, 2), padding='same',
                                      data_format=data_format)(x)
        first_stride = 1
        last_stride = 2

    # Define a block functor which can create blocks.
    with arg_scope(
        [CNNBlock],
            use_batch_norm=use_batch_norm,
            all_projections=all_projections,
            use_shortcuts=True,
            data_format=data_format,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            freeze_bn=freeze_bn,
            activation_kwargs={},
            use_bias=use_bias):
        if nlayers == 10:
            x = CNNBlock(repeat=1, stride=first_stride,
                         subblocks=[(3, 64), (3, 64)],
                         index=1, freeze_block=freeze1)(x)
            x = CNNBlock(repeat=1, stride=2,
                         subblocks=[(3, 128), (3, 128)],
                         index=2, freeze_block=freeze2)(x)
            x = CNNBlock(repeat=1, stride=2,
                         subblocks=[(3, 256), (3, 256)],
                         index=3, freeze_block=freeze3)(x)
            x = CNNBlock(repeat=1, stride=last_stride,
                         subblocks=[(3, 512), (3, 512)],
                         index=4, freeze_block=freeze4)(x)
        elif nlayers == 18:
            x = CNNBlock(repeat=2, stride=first_stride,
                         subblocks=[(3, 64), (3, 64)],
                         index=1, freeze_block=freeze1)(x)
            x = CNNBlock(repeat=2, stride=2,
                         subblocks=[(3, 128), (3, 128)],
                         index=2, freeze_block=freeze2)(x)
            x = CNNBlock(repeat=2, stride=2,
                         subblocks=[(3, 256), (3, 256)],
                         index=3, freeze_block=freeze3)(x)
            x = CNNBlock(repeat=2, stride=last_stride,
                         subblocks=[(3, 512), (3, 512)],
                         index=4, freeze_block=freeze4)(x)
        elif nlayers == 34:
            x = CNNBlock(repeat=3, stride=first_stride,
                         subblocks=[(3, 64), (3, 64)],
                         index=1, freeze_block=freeze1)(x)
            x = CNNBlock(repeat=4, stride=2,
                         subblocks=[(3, 128), (3, 128)],
                         index=2, freeze_block=freeze2)(x)
            x = CNNBlock(repeat=6, stride=2,
                         subblocks=[(3, 256), (3, 256)],
                         index=3, freeze_block=freeze3)(x)
            x = CNNBlock(repeat=3, stride=last_stride,
                         subblocks=[(3, 512), (3, 512)],
                         index=4, freeze_block=freeze4)(x)
        elif nlayers == 50:
            x = CNNBlock(repeat=3, stride=first_stride,
                         subblocks=[(1, 64), (3, 64), (1, 256)],
                         index=1, freeze_block=freeze1)(x)
            x = CNNBlock(repeat=4, stride=2,
                         subblocks=[(1, 128), (3, 128), (1, 512)],
                         index=2, freeze_block=freeze2)(x)
            x = CNNBlock(repeat=6, stride=2,
                         subblocks=[(1, 256), (3, 256), (1, 1024)],
                         index=3, freeze_block=freeze3)(x)
            x = CNNBlock(repeat=3, stride=last_stride,
                         subblocks=[(1, 512), (3, 512), (1, 2048)],
                         index=4, freeze_block=freeze4)(x)
        elif nlayers == 101:
            x = CNNBlock(repeat=3, stride=first_stride,
                         subblocks=[(1, 64), (3, 64), (1, 256)],
                         index=1, freeze_block=freeze1)(x)
            x = CNNBlock(repeat=4, stride=2,
                         subblocks=[(1, 128), (3, 128), (1, 512)],
                         index=2, freeze_block=freeze2)(x)
            x = CNNBlock(repeat=23, stride=2,
                         subblocks=[(1, 256), (3, 256), (1, 1024)],
                         index=3, freeze_block=freeze3)(x)
            x = CNNBlock(repeat=3, stride=last_stride,
                         subblocks=[(1, 512), (3, 512), (1, 2048)],
                         index=4, freeze_block=freeze4)(x)
        elif nlayers == 152:
            x = CNNBlock(repeat=3, stride=first_stride,
                         subblocks=[(1, 64), (3, 64), (1, 256)],
                         index=1, freeze_block=freeze1)(x)
            x = CNNBlock(repeat=8, stride=2,
                         subblocks=[(1, 128), (3, 128), (1, 512)],
                         index=2, freeze_block=freeze2)(x)
            x = CNNBlock(repeat=36, stride=2,
                         subblocks=[(1, 256), (3, 256), (1, 1024)],
                         index=3, freeze_block=freeze3)(x)
            x = CNNBlock(repeat=3, stride=last_stride,
                         subblocks=[(1, 512), (3, 512), (1, 2048)],
                         index=4, freeze_block=freeze4)(x)
        else:
            raise NotImplementedError('A resnet with nlayers=%d is not implemented.' % nlayers)

    # Add AveragePooling2D layer if use_pooling is enabled after resnet block.
    if use_pooling:
        x = keras.layers.AveragePooling2D(pool_size=(7, 7),
                                          data_format=data_format,
                                          padding='same')(x)

    # Naming model.
    model_name = 'resnet%d' % nlayers
    if not use_pooling:
        model_name += '_nopool'
    if use_batch_norm:
        model_name += '_bn'
    # Set up keras model object.
    model = keras.models.Model(inputs=inputs, outputs=x, name=model_name)

    # Add a dense head of nclasses if enabled.
    if add_head:
        model = add_dense_head(model, inputs, nclasses, head_activation)
    return model
