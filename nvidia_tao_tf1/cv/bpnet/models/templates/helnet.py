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

"""Model templates for BpNet HelNets (modified versions of original HelNet)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file

from nvidia_tao_tf1.core.decorators.arg_scope import arg_scope
from nvidia_tao_tf1.core.models.templates.utils import add_activation
from nvidia_tao_tf1.core.models.templates.utils import get_batchnorm_axis
from nvidia_tao_tf1.core.models.templates.utils import performance_test_model

from nvidia_tao_tf1.cv.bpnet.models.templates.utils import CNNBlock

logger = logging.getLogger(__name__)


def HelNet(nlayers,
           inputs,
           mtype='default',
           pooling=False,
           use_last_block=True,
           use_batch_norm=False,
           data_format=None,
           kernel_regularizer=None,
           bias_regularizer=None,
           activation_type='relu',
           activation_kwargs=None,
           block_widths=(64, 128, 256, 512),
           weights=None):
    """
    Construct a HelNet with a set amount of layers.

    The HelNet family is very similar, and in its convolutional core identical, to the ResNet family
    described in [1]. The main differences are: the absence of shortcuts (skip connections); the use
    of a different head; and usually one or two changes in the striding. We've also made the second
    layer (max pool) optional, though it was standard for ResNets described in the paper [1].

    Args:
        nlayers (int): the number of layers desired for this HelNet (e.g. 6, 10, ..., 34).
        inputs (tensor): the input tensor `x`.
        pooling (bool): whether max-pooling with a stride of 2 should be used as the second layer.
            If `False`, this stride will be added to the next convolution instead.
        use_batch_norm (bool): whether batchnorm should be added after each convolution.
        data_format (str): either 'channels_last' (NHWC) or 'channels_f
        irst' (NCHW).
        kernel_regularizer (float): regularizer to apply to kernels.
        bias_regularizer (float): regularizer to apply to biases.
        activation_type (str): Type of activation.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        weights (str): download and load in pretrained weights, f.e. 'imagenet'.
        block_widths (tuple of ints): width i.e. number of features maps in each convolutional block
            in the model.
    Returns:
        Model: the output model after applying the HelNet on top of input `x`.

    [1] Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
    """
    if data_format is None:
        data_format = K.image_data_format()
    activation_kwargs = activation_kwargs or {}

    # Create HelNet-0 model for training diagnostics.
    if nlayers == 0:
        return performance_test_model(inputs, data_format, activation_type)

    if mtype == 'default':
        fl_stride = (2, 2)
        fl_drate = (1, 1)
        third_stride = 2
        third_drate = (1, 1)
    elif mtype == 's8_3rdblock_wdilation':
        fl_stride = (2, 2)
        fl_drate = (1, 1)
        third_stride = 1
        third_drate = (2, 2)
    elif mtype == 's8_3rdblock':
        fl_stride = (2, 2)
        fl_drate = (1, 1)
        third_stride = 1
        third_drate = (1, 1)
    elif mtype == 's8_1stlayer_wdilation':
        fl_stride = (1, 1)
        fl_drate = (2, 2)
        third_stride = 2
        third_drate = (1, 1)
    elif mtype == 's8_1stlayer':
        fl_stride = (1, 1)
        fl_drate = (1, 1)
        third_stride = 2
        third_drate = (1, 1)
    else:
        raise NotImplementedError(
            "Helnet type: {} is not supported.".format(mtype))

    x = Conv2D(64, (7, 7),
               strides=fl_stride,
               dilation_rate=fl_drate,
               padding='same',
               data_format=data_format,
               use_bias=not (use_batch_norm),
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name='conv1')(inputs)
    if use_batch_norm:
        x = BatchNormalization(axis=get_batchnorm_axis(data_format),
                               name='bn_conv1')(x)
    x = add_activation(activation_type, **activation_kwargs)(x)
    if pooling:
        x = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         data_format=data_format)(x)
        first_stride = 1
    else:
        first_stride = 2

    # Define a block functor which can create blocks
    with arg_scope([CNNBlock],
                   use_batch_norm=use_batch_norm,
                   use_shortcuts=False,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   activation_type=activation_type,
                   activation_kwargs=activation_kwargs,
                   use_bias=not (use_batch_norm)):
        if nlayers == 6:
            x = CNNBlock(repeat=1,
                         stride=first_stride,
                         subblocks=[(3, block_widths[0])],
                         index=1)(x)
            x = CNNBlock(repeat=1,
                         stride=2,
                         subblocks=[(3, block_widths[1])],
                         index=2)(x)
            x = CNNBlock(repeat=1,
                         stride=third_stride,
                         subblocks=[(3, block_widths[2])],
                         index=3,
                         first_subblock_dilation_rate=third_drate)(x)
            if use_last_block:
                x = CNNBlock(repeat=1,
                             stride=1,
                             subblocks=[(3, block_widths[3])],
                             index=4)(x)
        elif nlayers == 10:
            x = CNNBlock(repeat=1,
                         stride=first_stride,
                         subblocks=[(3, block_widths[0])] * 2,
                         index=1)(x)
            x = CNNBlock(repeat=1,
                         stride=2,
                         subblocks=[(3, block_widths[1])] * 2,
                         index=2)(x)
            x = CNNBlock(repeat=1,
                         stride=third_stride,
                         subblocks=[(3, block_widths[2])] * 2,
                         index=3,
                         first_subblock_dilation_rate=third_drate)(x)
            if use_last_block:
                x = CNNBlock(repeat=1,
                             stride=1,
                             subblocks=[(3, block_widths[3])] * 2,
                             index=4)(x)
        elif nlayers == 12:
            x = CNNBlock(repeat=1,
                         stride=first_stride,
                         subblocks=[(3, block_widths[0])] * 2,
                         index=1)(x)
            x = CNNBlock(repeat=1,
                         stride=2,
                         subblocks=[(3, block_widths[1])] * 2,
                         index=2)(x)
            x = CNNBlock(repeat=2,
                         stride=third_stride,
                         subblocks=[(3, block_widths[2])] * 2,
                         index=3,
                         first_subblock_dilation_rate=third_drate)(x)
            if use_last_block:
                x = CNNBlock(repeat=1,
                             stride=1,
                             subblocks=[(3, block_widths[3])] * 2,
                             index=4)(x)
        elif nlayers == 18:
            x = CNNBlock(repeat=2,
                         stride=first_stride,
                         subblocks=[(3, block_widths[0])] * 2,
                         index=1)(x)
            x = CNNBlock(repeat=2,
                         stride=2,
                         subblocks=[(3, block_widths[1])] * 2,
                         index=2)(x)
            x = CNNBlock(repeat=2,
                         stride=third_stride,
                         subblocks=[(3, block_widths[2])] * 2,
                         index=3,
                         first_subblock_dilation_rate=third_drate)(x)
            if use_last_block:
                x = CNNBlock(repeat=2,
                             stride=1,
                             subblocks=[(3, block_widths[3])] * 2,
                             index=4)(x)
        elif nlayers == 26:
            x = CNNBlock(repeat=3,
                         stride=first_stride,
                         subblocks=[(3, block_widths[0])] * 2,
                         index=1)(x)
            x = CNNBlock(repeat=4,
                         stride=2,
                         subblocks=[(3, block_widths[1])] * 2,
                         index=2)(x)
            x = CNNBlock(repeat=3,
                         stride=third_stride,
                         subblocks=[(3, block_widths[2])] * 2,
                         index=3,
                         first_subblock_dilation_rate=third_drate)(x)
            if use_last_block:
                x = CNNBlock(repeat=2,
                             stride=1,
                             subblocks=[(3, block_widths[3])] * 2,
                             index=4)(x)
        elif nlayers == 34:
            x = CNNBlock(repeat=3,
                         stride=first_stride,
                         subblocks=[(3, block_widths[0])] * 2,
                         index=1)(x)
            x = CNNBlock(repeat=4,
                         stride=2,
                         subblocks=[(3, block_widths[1])] * 2,
                         index=2)(x)
            x = CNNBlock(repeat=6,
                         stride=third_stride,
                         subblocks=[(3, block_widths[2])] * 2,
                         index=3,
                         first_subblock_dilation_rate=third_drate)(x)
            if use_last_block:
                x = CNNBlock(repeat=3,
                             stride=1,
                             subblocks=[(3, block_widths[3])] * 2,
                             index=4)(x)
        else:
            raise NotImplementedError(
                'A Helnet with nlayers=%d is not implemented.' % nlayers)

    model_name = 'helnet%d_s16' % nlayers
    if pooling:
        model_name += '_nopool'
    if use_batch_norm:
        model_name += '_bn'

    model = Model(inputs=inputs, outputs=x, name=model_name)

    if weights == 'imagenet':
        logger.warning(
            "Imagenet weights can not be used for production models.")
        if nlayers == 18:
            if use_batch_norm:
                weights_path = get_file(
                    'imagenet_helnet18-bn_weights_20170729.h5',
                    'https://s3-us-west-2.amazonaws.com/'
                    '9j2raan2rcev-ai-infra-models/'
                    'imagenet_helnet18-bn_weights_20170729.h5',
                    cache_subdir='models',
                    md5_hash='6a2d59e48d8b9f0b41a2b02a2f3c018e')
            else:
                weights_path = get_file(
                    'imagenet_helnet18-no-bn_weights_20170729.h5',
                    'https://s3-us-west-2.amazonaws.com/'
                    '9j2raan2rcev-ai-infra-models/'
                    'imagenet_helnet18-no-bn_weights_20170729.h5',
                    cache_subdir='models',
                    md5_hash='3282b1e5e7f8e769a034103c455968e6')
            model.load_weights(weights_path, by_name=True)

    return model
