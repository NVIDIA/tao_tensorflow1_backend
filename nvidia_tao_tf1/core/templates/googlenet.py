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

"""Maglev model templates for GoogLeNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense, Flatten
from keras.layers import Dropout
# from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.models import Model


from nvidia_tao_tf1.core.templates.utils import arg_scope
from nvidia_tao_tf1.core.templates.utils import get_batchnorm_axis
from nvidia_tao_tf1.core.templates.utils import InceptionV1Block


def GoogLeNet(inputs, use_batch_norm=True, data_format=None, add_head=False,
              nclasses=1000, kernel_regularizer=None, bias_regularizer=None,
              activation_type='relu', freeze_blocks=None, freeze_bn=False,
              use_bias=True):
    """
    Construct GoogLeNet, based on the architectures from the original paper [1].

    Args:
        inputs (tensor): the input tensor.
        use_batch_norm (bool): whether batchnorm or Local Response BatchNormalization
                               if True: batchnorm should be added after each convolution.
                               if False: LRN is added as defined in paper [1].
                                         LRN is not supported in pruning and model export.
        data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
        add_head (bool): whether to add the original [1] classification head. Note that if you
            don't include the head, the actual number of layers in the model produced by this
            function is 'nlayers-3`, as they don't include the last 3 FC layers.
        nclasses (int): the number of classes to be added to the classification head. Can be `None`
            if unused.
        kernel_regularizer: regularizer to apply to kernels.
        bias_regularizer: regularizer to apply to biases.
        freeze_blocks(list): the blocks in the model to be frozen.
        freeze_bn(bool): whether or not to freeze the BN layer in the model.
        use_bias(bool): Whether or not to use bias for conv layers.
    Returns:
        Model: the output model after applying the GoogLeNet on top of input `x`.

    [1] Going Deeper with Convolutions, Szegedy, Christian, et. al., Proceedings
        of the IEEE Conference on Computer Vision and Pattern Recognition, 2015.
        (https://arxiv.org/abs/1409.4842)
    """
    if data_format is None:
        data_format = K.image_data_format()

    if use_batch_norm:
        bn_axis = get_batchnorm_axis(data_format)

    if freeze_blocks is None:
        freeze_blocks = []

    x = Conv2D(64,
               (7, 7),
               strides=(2, 2),
               padding='same',
               data_format=data_format,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name='conv1',
               trainable=not(0 in freeze_blocks),
               use_bias=use_bias)(inputs)
    if use_batch_norm:
        if freeze_bn:
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation(activation_type)(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                     data_format=data_format, name='pool1')(x)

    # we force use_batch_norm to be True in the model builder
    # TODO: <vpraveen> Uncomment when export for LRN is supported.
    # if not use_batch_norm:
    #     x = Lambda(lambda y, arguments={'type': 'googlenet_lrn',
    #                                     'depth_radius': 5,
    #                                     'bias': 1.0,
    #                                     'alpha': 0.0001,
    #                                     'beta': 0.75,
    #                                     'name': 'lrn1'}:
    #                tf.nn.lrn(y, depth_radius=5,
    #                          bias=1.0,
    #                          alpha=0.0001,
    #                          beta=0.75,
    #                          name='lrn1'))(x)
    x = Conv2D(64,
               (1, 1),
               strides=(1, 1),
               padding='same',
               data_format=data_format,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name='conv2_reduce',
               trainable=not(0 in freeze_blocks),
               use_bias=use_bias)(x)
    if use_batch_norm:
        if freeze_bn:
            x = BatchNormalization(axis=bn_axis, name='bn_conv2_reduce')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, name='bn_conv2_reduce')(x)
    x = Activation(activation_type)(x)

    x = Conv2D(192,
               (3, 3),
               strides=(1, 1),
               padding='same',
               data_format=data_format,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name='conv2',
               trainable=not(0 in freeze_blocks),
               use_bias=use_bias)(x)
    if use_batch_norm:
        if freeze_bn:
            x = BatchNormalization(axis=bn_axis, name='bn_conv2')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, name='bn_conv2')(x)
    x = Activation(activation_type)(x)

    # # we force use_batch_norm to be True in the model builder
    # TODO: <vpraveen> Uncomment when export for LRN is supported.
    # if not use_batch_norm:
    #     x = Lambda(lambda y, arguments={'type': 'googlenet_lrn',
    #                                     'depth_radius': 5,
    #                                     'bias': 1.0,
    #                                     'alpha': 0.0001,
    #                                     'beta': 0.75,
    #                                     'name': 'lrn2'}:
    #                tf.nn.lrn(y, depth_radius=5,
    #                          bias=1.0,
    #                          alpha=0.0001,
    #                          beta=0.75,
    #                          name='lrn2'))(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                     data_format=data_format, name='pool2')(x)

    # Define a block functor which can create blocks.
    with arg_scope([InceptionV1Block],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   activation_type=activation_type,
                   freeze_bn=freeze_bn,
                   use_bias=use_bias):
        # Implementing GoogLeNet architecture.
        # Inception_3a
        x = InceptionV1Block(subblocks=(64, 96, 128, 16, 32, 32),
                             index='3a',
                             trainable=not(1 in freeze_blocks))(x)
        # Inception_3b
        x = InceptionV1Block(subblocks=(128, 128, 192, 32, 96, 64),
                             index='3b',
                             trainable=not(2 in freeze_blocks))(x)
        # Max Pooling
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                         data_format=data_format, name='pool3')(x)
        # Inception_4a
        x = InceptionV1Block(subblocks=(192, 96, 208, 16, 48, 64),
                             index='4a',
                             trainable=not(3 in freeze_blocks))(x)
        # Inception_4b
        x = InceptionV1Block(subblocks=(160, 112, 224, 24, 64, 64),
                             index='4b',
                             trainable=not(4 in freeze_blocks))(x)
        # Inception_4c
        x = InceptionV1Block(subblocks=(128, 128, 256, 24, 64, 64),
                             index='4c',
                             trainable=not(5 in freeze_blocks))(x)
        # Inception_4d
        x = InceptionV1Block(subblocks=(112, 144, 288, 32, 64, 64),
                             index='4d',
                             trainable=not(6 in freeze_blocks))(x)
        # Inception_4e
        x = InceptionV1Block(subblocks=(256, 160, 320, 32, 128, 128),
                             index='4e',
                             trainable=not(7 in freeze_blocks))(x)
        if add_head:
            # Add Max Pooling layer if there is a classification head to be added
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                             data_format=data_format, name='pool4')(x)
        # Inception_5a
        x = InceptionV1Block(subblocks=(256, 160, 320, 32, 128, 128),
                             index='5a',
                             trainable=not(8 in freeze_blocks))(x)
        # Inception_5b
        x = InceptionV1Block(subblocks=(384, 192, 384, 48, 128, 128),
                             index='5b',
                             trainable=not(9 in freeze_blocks))(x)

    if add_head:
        # Classification block.
        # Add Average Pooling layer if there is a classification head to be added
        x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same',
                             data_format=data_format, name='avg_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dropout(0.4, noise_shape=None, seed=None)(x)
        x = Dense(nclasses, activation='softmax', name='output_fc')(x)

    # Naming model.
    model_name = 'Googlenet'
    if use_batch_norm:
        model_name += '_bn'
    # Set up keras model object.
    model = Model(inputs=inputs, outputs=x, name=model_name)

    return model
