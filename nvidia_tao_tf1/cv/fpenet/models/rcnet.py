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
"""Model template for Recombinator Networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.backend as K
from keras.layers import concatenate
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.models import Model
from nvidia_tao_tf1.core.models.templates.utils import CNNBlock


concat_axis_map = {'channels_last': 3, 'channels_first': 1}


def RecombinatorNet(inputs,
                    pooling=True,
                    use_batch_norm=False,
                    data_format='channels_first',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activation_type='relu',
                    activation_kwargs=None,
                    blocks_encoder=None,
                    block_trunk=None,
                    blocks_decoder=None,
                    use_upsampling_layer=True):
    """
    Construct a Recombinator Network template.

    Described in the paper [1].

    Args:
        pooling (bool): whether max-pooling with a stride of 2 should be used.
            If `False`, this stride will be added to the next convolution instead.
        use_batch_norm (bool): whether batchnorm should be added after each convolution.
        data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
        kernel_regularizer (float): regularizer to apply to kernels.
        bias_regularizer (float): regularizer to apply to biases.
        activation_type (str): Type of activation.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        blocks_encoder (list of convolution blocks): A convolution block is a list of (K, C) tuples
            of arbitrary length, where C is the number of output channels and K the kernel size of
            a convolution layer. The encoder reduces the spatial resolution horizontally
            and vertically by a factor of two (stride of 2) per convolution block, respectively.
        block_trunk (one convolution block): This block preserves the spatial resolution of the
            final encoder output and 'refines' it.
        blocks_decoder (list of convolution blocks): The decoder increases the spatial resolution
            horizontally and vertically by a factor of two (upsample factor of 2) per convolution
            block, respectively. blocks_encoder and blocks_decoder must have the same length of
            convolution blocks, while the number of convolution layers per block may vary.
        use_upsampling_layer (bool): use upsampling or deconv layer
    Returns:
        Model: The output model after applying the Recombinator Network on top of input `x`.

    [1] Recombinator Networks: Learning Coarse-to-Fine Feature Aggregation
        (https://arxiv.org/abs/1511.07356)

    """
    assert len(blocks_encoder) == len(
        blocks_decoder
    ), 'Need an equal list length for blocks_encoder and blocks_decoder (number of RCN branches).'
    nbranches = len(blocks_encoder)

    if data_format is None:
        data_format = K.image_data_format()
    activation_kwargs = activation_kwargs or {}

    blocks_encoder = blocks_encoder or [[(3, 64)]] * 4
    block_trunk = block_trunk or [(3, 64), (3, 64), (1, 64)]
    blocks_decoder = blocks_decoder or [[(3, 64), (1, 64)]] * 4

    # Adjust the convolution stride of the encoder depending on the pooling setting:
    if pooling:
        filter_stride_encoder = 1
    else:
        filter_stride_encoder = 2

    concat_axis = concat_axis_map[data_format]

    deconv_kernel_initializer = keras.initializers.he_uniform()

    encoder_outputs = list()
    x = inputs

    # Create the encoder blocks (strided):
    for block in blocks_encoder:
        if not pooling:
            encoder_outputs.append(x)

        x = CNNBlock(
                repeat=1,
                stride=filter_stride_encoder,
                subblocks=block,
                use_batch_norm=use_batch_norm,
                use_shortcuts=False,
                data_format=data_format,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activation_type=activation_type,
                activation_kwargs=activation_kwargs,
                use_bias=not (use_batch_norm))(x)

        if pooling:
            encoder_outputs.append(x)
            x = MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same',
                data_format=data_format)(x)

    # Create the trunk block (unstrided):
    x = CNNBlock(repeat=1,
                 stride=1,
                 subblocks=block_trunk,
                 use_batch_norm=use_batch_norm,
                 use_shortcuts=False,
                 data_format=data_format,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 activation_type=activation_type,
                 activation_kwargs=activation_kwargs,
                 use_bias=not (use_batch_norm))(x)

    # Create the decoder blocks (up-sampling):
    for block in blocks_decoder:

        # upsampling or deconv layer
        if use_upsampling_layer:
            x = UpSampling2D(size=(2, 2),
                             data_format=data_format,
                             trainable=False)(x)
        else:
            kernel_size, filters = block
            # Fixing kernel size for deconv
            kernel_size = (2, 2)
            x = Conv2DTranspose(filters[1],
                                kernel_size=kernel_size,
                                strides=2,
                                output_padding=(0, 0),
                                use_bias=False,
                                kernel_initializer=deconv_kernel_initializer,
                                data_format=data_format)(x)

        concat_input_encoder = encoder_outputs.pop()
        x = concatenate(axis=concat_axis, inputs=[x, concat_input_encoder])
        x = CNNBlock(repeat=1,
                     stride=1,
                     subblocks=block,
                     use_batch_norm=use_batch_norm,
                     use_shortcuts=False,
                     data_format=data_format,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activation_type=activation_type,
                     activation_kwargs=activation_kwargs,
                     use_bias=not (use_batch_norm))(x)

    model_name = 'rcn_%dbranches' % nbranches
    if not pooling:
        model_name += '_nopool'
    if use_batch_norm:
        model_name += '_bn'
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model
