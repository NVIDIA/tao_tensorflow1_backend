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
"""EfficientNet model templates in Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    ZeroPadding2D
)
from tensorflow.keras.models import Model

from nvidia_tao_tf1.core.templates.utils_tf import (
    block,
    CONV_KERNEL_INITIALIZER,
    correct_pad,
    DENSE_KERNEL_INITIALIZER,
    force_stride16,
    round_filters,
    round_repeats,
    swish
)


DEFAULT_BLOCKS_ARGS = (
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
)


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 add_head=True,
                 input_tensor=None,
                 input_shape=None,
                 classes=1000,
                 data_format="channels_first",
                 freeze_bn=False,
                 freeze_blocks=None,
                 use_bias=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 stride16=False,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        add_head: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `add_head` is False.
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `add_head` is True.
        data_format(str): Keras data format.
        freeze_bn(bool): Freeze all the BN layers or not.
        freeze_blocks(list): Block IDs to be frozen in this model.
        use_bias(bool): Use bias or not for Conv layers that are followed by a BN layer.
        kernel_regularizer: The kernel regularizer.
        bias_regularizer: The bias regularizer.
        stride16(bool): Limit the total stride of the model to 16 or not, default is stride 32.
            This is used for DetectNet_v2. All other use cases will use stride 32.
    # Returns
        A Keras model instance.
    """
    # activation_fn defaults to swish if it is None or empty string
    bn_opt = {
        'momentum': 0.99,
        'epsilon': 1e-3
    }
    if activation_fn in [None, ""]:
        activation_fn = swish
    # old_data_format = K.image_data_format()
    assert data_format == 'channels_last'
    K.set_image_data_format(data_format)
    if freeze_blocks is None:
        freeze_blocks = []

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    # Build stem
    x = img_input
    x = ZeroPadding2D(
        padding=correct_pad(x, 3),
        name='stem_conv_pad',
        data_format=data_format,
    )(x)
    x = Conv2D(
        round_filters(32, depth_divisor, width_coefficient),
        3,
        strides=2,
        padding='valid',
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        trainable=not bool(0 in freeze_blocks),
        data_format=data_format,
        name='stem_conv'
    )(x)
    if freeze_bn:
        x = BatchNormalization(axis=bn_axis, name='stem_bn')(x, training=False)
    else:
        x = BatchNormalization(axis=bn_axis, name='stem_bn', **bn_opt)(x)
    x = Activation(activation_fn, name='stem_activation')(x)

    # Build blocks
    blocks_args = deepcopy(list(blocks_args))
    # in stride 16 mode, force the last stride 2 to be 1.
    if stride16:
        force_stride16(blocks_args)
    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'], depth_divisor, width_coefficient)
        args['filters_out'] = round_filters(args['filters_out'], depth_divisor, width_coefficient)

        for j in range(round_repeats(args.pop('repeats'), depth_coefficient)):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(
                x, activation_fn, drop_connect_rate * b / blocks,
                freeze=bool((i + 1) in freeze_blocks),
                freeze_bn=freeze_bn,
                use_bias=use_bias,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                data_format=data_format,
                name='block{}{}_'.format(i + 1, chr(j + 97)),
                **args)
            b += 1

    # Build top
    x = Conv2D(
        round_filters(1280, depth_divisor, width_coefficient),
        1,
        padding='same',
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        trainable=not bool((len(blocks_args) + 1) in freeze_blocks),
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        data_format=data_format,
        name='top_conv'
    )(x)
    if freeze_bn:
        x = BatchNormalization(axis=bn_axis, name='top_bn')(x, training=False)
    else:
        x = BatchNormalization(axis=bn_axis, name='top_bn', **bn_opt)(x)
    x = Activation(activation_fn, name='top_activation')(x)
    if add_head:
        # global pool as: avg pool + flatten for pruning support
        output_shape = x.get_shape().as_list()
        if data_format == 'channels_first':
            pool_size = (output_shape[-2], output_shape[-1])
        else:
            pool_size = (output_shape[-3], output_shape[-2])
        x = AveragePooling2D(
            pool_size=pool_size, name='avg_pool',
            data_format=data_format, padding='valid'
        )(x)
        x = Flatten(name='flatten')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name='top_dropout')(x)
        # head will always not be frozen
        # set the name to 'predictions' to align with that in add_dense_head()
        x = Dense(
            classes,
            activation='softmax',
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='predictions'
        )(x)
    # Create model.
    model = Model(img_input, x, name=model_name)
    # restore previous data format
    # K.set_image_data_format(old_data_format)
    return model


def EfficientNetB0(add_head=True,
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   data_format="channels_first",
                   freeze_bn=False,
                   freeze_blocks=None,
                   use_bias=False,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   stride16=False,
                   activation_type=None,
                   **kwargs):
    """EfficientNet B0."""
    return EfficientNet(1.0, 1.0, 0.2,
                        drop_connect_rate=0,
                        model_name='efficientnet-b0',
                        add_head=add_head,
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        classes=classes,
                        data_format=data_format,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        stride16=stride16,
                        activaton_fn=activation_type,
                        **kwargs)


def EfficientNetB1(add_head=True,
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   data_format="channels_first",
                   freeze_bn=False,
                   freeze_blocks=None,
                   use_bias=False,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   stride16=False,
                   activation_type=None,
                   **kwargs):
    """EfficientNet B1."""
    return EfficientNet(1.0, 1.1, 0.2,
                        model_name='efficientnet-b1',
                        add_head=add_head,
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        classes=classes,
                        data_format=data_format,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        stride16=stride16,
                        activation_fn=activation_type,
                        **kwargs)


def EfficientNetB2(add_head=True,
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   data_format="channels_first",
                   freeze_bn=False,
                   freeze_blocks=None,
                   use_bias=False,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   stride16=False,
                   activation_type=None,
                   **kwargs):
    """EfficientNet B2."""
    return EfficientNet(1.1, 1.2, 0.3,
                        model_name='efficientnet-b2',
                        add_head=add_head,
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        classes=classes,
                        data_format=data_format,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        stride16=stride16,
                        activation_fn=activation_type,
                        **kwargs)


def EfficientNetB3(add_head=True,
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   data_format="channels_first",
                   freeze_bn=False,
                   freeze_blocks=None,
                   use_bias=False,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   stride16=False,
                   activation_type=None,
                   **kwargs):
    """EfficientNet B3."""
    return EfficientNet(1.2, 1.4, 0.3,
                        model_name='efficientnet-b3',
                        add_head=add_head,
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        classes=classes,
                        data_format=data_format,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        stride16=stride16,
                        activation_fn=activation_type,
                        **kwargs)


def EfficientNetB4(add_head=True,
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   data_format="channels_first",
                   freeze_bn=False,
                   freeze_blocks=None,
                   use_bias=False,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   stride16=False,
                   activation_type=None,
                   **kwargs):
    """EfficientNet B4."""
    return EfficientNet(1.4, 1.8, 0.4,
                        model_name='efficientnet-b4',
                        add_head=add_head,
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        classes=classes,
                        data_format=data_format,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        stride16=stride16,
                        activation_fn=activation_type,
                        **kwargs)


def EfficientNetB5(add_head=True,
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   data_format="channels_first",
                   freeze_bn=False,
                   freeze_blocks=None,
                   use_bias=False,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   stride16=False,
                   activation_type=None,
                   **kwargs):
    """EfficientNet B5."""
    return EfficientNet(1.6, 2.2, 0.4,
                        model_name='efficientnet-b5',
                        add_head=add_head,
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        classes=classes,
                        data_format=data_format,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        stride16=stride16,
                        activation_fn=activation_type,
                        **kwargs)


def EfficientNetB6(add_head=True,
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   data_format="channels_first",
                   freeze_bn=False,
                   freeze_blocks=None,
                   use_bias=False,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   stride16=False,
                   activation_type=None,
                   **kwargs):
    """EfficientNet B6."""
    return EfficientNet(1.8, 2.6, 0.5,
                        model_name='efficientnet-b6',
                        add_head=add_head,
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        classes=classes,
                        data_format=data_format,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        stride16=stride16,
                        activation_fn=activation_type,
                        **kwargs)


def EfficientNetB7(add_head=True,
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   data_format="channels_first",
                   freeze_bn=False,
                   freeze_blocks=None,
                   use_bias=False,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   stride16=False,
                   activation_type=None,
                   **kwargs):
    """EfficientNet B7."""
    return EfficientNet(2.0, 3.1, 0.5,
                        model_name='efficientnet-b7',
                        add_head=add_head,
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        classes=classes,
                        data_format=data_format,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        stride16=stride16,
                        activation_fn=activation_type,
                        **kwargs)
