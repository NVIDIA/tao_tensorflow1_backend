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

"""IVA utilities for tf model templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import contextlib
import functools
import inspect
import math
import os
import re
import threading

import tensorflow as tf

from nvidia_tao_tf1.core.models.import_keras import keras as keras_fn
from nvidia_tao_tf1.cv.yolo_v4.layers.split import Split

keras = keras_fn()

bn_axis_map = {'channels_last': 3, 'channels_first': 1}

SUBBLOCK_IDS = ['1x1', '3x3_reduce', '3x3', '5x5_reduce', '5x5', 'pool', 'pool_proj']

_ARGSTACK = [{}]

_DECORATED_OPS = {}


def _get_arg_stack():
    if _ARGSTACK:
        return _ARGSTACK
    _ARGSTACK.append({})
    return _ARGSTACK


def _current_arg_scope():
    stack = _get_arg_stack()
    return stack[-1]


def _key_op(op):
    return getattr(op, "_key_op", str(op))


def _name_op(op):
    return (op.__module__, op.__name__)


def _kwarg_names(func):
    kwargs_length = len(func.__defaults__) if func.__defaults__ else 0
    return func.__code__.co_varnames[-kwargs_length : func.__code__.co_argcount]


def _add_op(op):
    key_op = _key_op(op)
    if key_op not in _DECORATED_OPS:
        _DECORATED_OPS[key_op] = _kwarg_names(op)


def get_batchnorm_axis(data_format):
    """Convert a data_format string to the correct index in a 4 dimensional tensor.

    Args:
        data_format (str): either 'channels_last' or 'channels_first'.
    Returns:
        int: the axis corresponding to the `data_format`.
    """
    return bn_axis_map[data_format]


def has_arg_scope(func):
    """Check whether a func has been decorated with @add_arg_scope or not.

    Args:
        func: function to check.

    Returns:
        a boolean.
    """
    return _key_op(func) in _DECORATED_OPS


@contextlib.contextmanager
def arg_scope(list_ops_or_scope, **kwargs):
    """Store the default arguments for the given set of list_ops.

    For usage, please see examples at top of the file.

    Args:
        list_ops_or_scope: List or tuple of operations to set argument scope for or
            a dictionary containing the current scope. When list_ops_or_scope is a
            dict, kwargs must be empty. When list_ops_or_scope is a list or tuple,
            then every op in it need to be decorated with @add_arg_scope to work.
        **kwargs: keyword=value that will define the defaults for each op in
                            list_ops. All the ops need to accept the given set of arguments.

    Yields:
        the current_scope, which is a dictionary of {op: {arg: value}}
    Raises:
        TypeError: if list_ops is not a list or a tuple.
        ValueError: if any op in list_ops has not be decorated with @add_arg_scope.
    """
    if isinstance(list_ops_or_scope, dict):
        # Assumes that list_ops_or_scope is a scope that is being reused.
        if kwargs:
            raise ValueError(
                "When attempting to re-use a scope by suppling a"
                "dictionary, kwargs must be empty."
            )
        current_scope = list_ops_or_scope.copy()
        try:
            _get_arg_stack().append(current_scope)
            yield current_scope
        finally:
            _get_arg_stack().pop()
    else:
        # Assumes that list_ops_or_scope is a list/tuple of ops with kwargs.
        if not isinstance(list_ops_or_scope, (list, tuple)):
            raise TypeError(
                "list_ops_or_scope must either be a list/tuple or reused"
                "scope (i.e. dict)"
            )
        try:
            current_scope = _current_arg_scope().copy()
            for op in list_ops_or_scope:
                if inspect.isclass(op):
                    # If we decorated a class, use the scope on the initializer
                    op = op.__init__
                key_op = _key_op(op)
                if not has_arg_scope(op):
                    raise ValueError(
                        "%s::%s is not decorated with @add_arg_scope" % _name_op(op)
                    )
                if key_op in current_scope:
                    current_kwargs = current_scope[key_op].copy()
                    current_kwargs.update(kwargs)
                    current_scope[key_op] = current_kwargs
                else:
                    current_scope[key_op] = kwargs.copy()
            _get_arg_stack().append(current_scope)
            yield current_scope
        finally:
            _get_arg_stack().pop()


def add_arg_scope(func):
    """Decorate a function with args so it can be used within an arg_scope.

    Args:
        func: function to decorate.

    Returns:
        A tuple with the decorated function func_with_args().
    """

    @functools.wraps(func)
    def func_with_args(*args, **kwargs):
        current_scope = _current_arg_scope()
        current_args = kwargs
        key_func = _key_op(func)
        if key_func in current_scope:
            current_args = current_scope[key_func].copy()
            current_args.update(kwargs)
        return func(*args, **current_args)

    _add_op(func)
    setattr(func_with_args, "_key_op", _key_op(func))
    setattr(func_with_args, "__doc__", func.__doc__)
    return func_with_args


def arg_scoped_arguments(func):
    """Return the list kwargs that arg_scope can set for a func.

    Args:
        func: function which has been decorated with @add_arg_scope.

    Returns:
        a list of kwargs names.
    """
    assert has_arg_scope(func)
    return _DECORATED_OPS[_key_op(func)]


class subblock_ids(object):
    """A operator to get index of subblock, overload [] operation."""

    def __getitem__(self, key):
        """
        Generate a subblock ID and return.

        Args:
            key (int): an index used to generate the subblock ID.
        """
        cur = key
        subblock_id = ''
        while cur >= 0:
            ch = chr(ord('a') + cur % 26)
            subblock_id = ch + subblock_id
            cur = cur // 26 - 1

        return subblock_id


class InceptionV1Block(object):
    """A functor for creating a Inception v1 block of layers."""

    @add_arg_scope
    def __init__(self,
                 use_batch_norm,
                 data_format,
                 kernel_regularizer,
                 bias_regularizer,
                 subblocks,
                 index,
                 freeze_bn=False,
                 activation_type='relu',
                 use_bias=True,
                 trainable=True,
                 use_td=False):
        """
        Initialization of the block functor object.

        Args:
            use_batch_norm (bool): whether batchnorm should be added after each convolution.
            data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
            kernel_regularizer (float): regularizer to apply to kernels.
            bias_regularizer (float): regularizer to apply to biases.
            subblocks (tuple): A tuple of size 6, defining number of feature-maps for
                subbblocks in an inception block.
                For GoogleNet from "Going deeper with convolutions" by Szegedy, Christian, et. al.
                Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015

                Inception_3a: (64, 96, 128, 16, 32, 32)

                Defines Inception block with following parallel branches
                1) 64 outputs from 1x1 convolutions
                2.1) 96 outputs from 1x1 convolutions --> 2.2) 128 outputs from 3x3 convolutions
                3.1) 16 outputs from 1x1 convolutions --> 3.2) 32 outputs from 5x5 convolutions
                4.1) Max pooling with 3x3 pooling size --> 4.2) 32 outputs from 1x1 convolutions

                the outputs of 1, 2.2, 3.2, and 4.2 are concatenated to produce final output.

            index (int): the index of the block to be created.
            activation_type (str): activation function type.
            freeze_bn(bool): Whether or not to freeze the BN layer.
            use_bias(bool): Whether or not to use bias for Conv/Dense, etc.
            trainable(bool): Whether or not to set the weights to be trainable.
            use_td(bool): Whether or not to wrap the layers into a TimeDistributed layer.
                This is useful in FasterRCNN.
        """
        self.use_batch_norm = use_batch_norm
        self.data_format = data_format
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation_type = activation_type
        self.subblocks = subblocks
        self.index = index
        self.name = 'inception_%s' % index
        self.freeze_bn = freeze_bn
        self.use_bias = use_bias
        self.trainable = trainable
        self.use_td = use_td

    def __call__(self, x):
        """Build the block.

        Args:
            x (tensor): input tensor.

        Returns:
            tensor: the output tensor after applying the block on top of input `x`.
        """
        x = self._subblocks(x, name_prefix=self.name)

        return x

    def _subblocks(self, x, name_prefix=None):
        """
        Stack several convolutions in a specific sequence given by a list of subblocks.

        Args:
            x (tensor): the input tensor.
            name_prefix (str): name prefix for all the layers created in this function.

        Returns:
            tensor: the output tensor after applying the ResNet block on top of input `x`.
        """
        nblocks = len(self.subblocks)
        if(nblocks != 6):
            print("Inception V1 block must have 6 subblocks")
            return(x)

        if self.use_batch_norm:
            bn_axis = get_batchnorm_axis(self.data_format)

        # First branch is 1x1 conv with padding = 0, and stride = 1
        layer = keras.layers.Conv2D(
            self.subblocks[0],
            (1, 1),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='%s_%s' % (name_prefix, SUBBLOCK_IDS[0]),
            use_bias=self.use_bias,
            trainable=self.trainable)
        if self.use_td:
            layer = keras.layers.TimeDistributed(layer)
        x1 = layer(x)

        if self.use_batch_norm:
            _name = '%s_%s_bn' % (name_prefix, SUBBLOCK_IDS[0])
            layer = keras.layers.BatchNormalization(axis=bn_axis, name=_name)
            if self.use_td:
                layer = keras.layers.TimeDistributed(layer)
            if self.freeze_bn:
                x1 = layer(x1, training=False)
            else:
                x1 = layer(x1)
        x1 = keras.layers.Activation(self.activation_type)(x1)

        # Second branch is 1x1 conv with padding = 0, and stride = 1
        layer = keras.layers.Conv2D(
            self.subblocks[1],
            (1, 1),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='%s_%s' % (name_prefix, SUBBLOCK_IDS[1]),
            use_bias=self.use_bias,
            trainable=self.trainable)
        if self.use_td:
            layer = keras.layers.TimeDistributed(layer)
        x2 = layer(x)

        if self.use_batch_norm:
            _name = '%s_%s_bn' % (name_prefix, SUBBLOCK_IDS[1])
            layer = keras.layers.BatchNormalization(axis=bn_axis, name=_name)
            if self.use_td:
                layer = keras.layers.TimeDistributed(layer)
            if self.freeze_bn:
                x2 = layer(x2, training=False)
            else:
                x2 = layer(x2)
        x2 = keras.layers.Activation(self.activation_type)(x2)

        # Second branch is 1x1 conv with padding = 0, and stride = 1 followed by 3x3 conv
        layer = keras.layers.Conv2D(
            self.subblocks[2],
            (3, 3),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='%s_%s' % (name_prefix, SUBBLOCK_IDS[2]),
            use_bias=self.use_bias,
            trainable=self.trainable)
        if self.use_td:
            layer = keras.layers.TimeDistributed(layer)
        x2 = layer(x2)

        if self.use_batch_norm:
            _name = '%s_%s_bn' % (name_prefix, SUBBLOCK_IDS[2])
            layer = keras.layers.BatchNormalization(axis=bn_axis, name=_name)
            if self.use_td:
                layer = keras.layers.TimeDistributed(layer)
            if self.freeze_bn:
                x2 = layer(x2, training=False)
            else:
                x2 = layer(x2)
        x2 = keras.layers.Activation(self.activation_type)(x2)

        # Third branch is 1x1 conv with stride = 1
        layer = keras.layers.Conv2D(
            self.subblocks[3],
            (1, 1),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='%s_%s' % (name_prefix, SUBBLOCK_IDS[3]),
            use_bias=self.use_bias,
            trainable=self.trainable)
        if self.use_td:
            layer = keras.layers.TimeDistributed(layer)
        x3 = layer(x)

        if self.use_batch_norm:
            _name = '%s_%s_bn' % (name_prefix, SUBBLOCK_IDS[3])
            layer = keras.layers.BatchNormalization(axis=bn_axis, name=_name)
            if self.use_td:
                layer = keras.layers.TimeDistributed(layer)
            if self.freeze_bn:
                x3 = layer(x3, training=False)
            else:
                x3 = layer(x3)
        x3 = keras.layers.Activation(self.activation_type)(x3)

        # Third branch is 1x1 conv with padding = 0, and stride = 1 followed by 5x5 conv
        layer = keras.layers.Conv2D(
            self.subblocks[4],
            (5, 5),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='%s_%s' % (name_prefix, SUBBLOCK_IDS[4]),
            use_bias=self.use_bias,
            trainable=self.trainable)
        if self.use_td:
            layer = keras.layers.TimeDistributed(layer)
        x3 = layer(x3)

        if self.use_batch_norm:
            _name = '%s_%s_bn' % (name_prefix, SUBBLOCK_IDS[4])
            layer = keras.layers.BatchNormalization(axis=bn_axis, name=_name)
            if self.use_td:
                layer = keras.layers.TimeDistributed(layer)
            if self.freeze_bn:
                x3 = layer(x3, training=False)
            else:
                x3 = layer(x3)
        x3 = keras.layers.Activation(self.activation_type)(x3)

        # Fourth branch is max pool stride = 1, and a 1x1 conv
        layer = keras.layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            name='%s_%s' % (name_prefix, SUBBLOCK_IDS[5]))
        if self.use_td:
            layer = keras.layers.TimeDistributed(layer)
        x4 = layer(x)

        layer = keras.layers.Conv2D(
            self.subblocks[5],
            (1, 1),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='%s_%s' % (name_prefix, SUBBLOCK_IDS[6]),
            use_bias=self.use_bias,
            trainable=self.trainable)
        if self.use_td:
            layer = keras.layers.TimeDistributed(layer)
        x4 = layer(x4)

        if self.use_batch_norm:
            _name = '%s_%s_bn' % (name_prefix, SUBBLOCK_IDS[6])
            layer = keras.layers.BatchNormalization(axis=bn_axis, name=_name)
            if self.use_td:
                layer = keras.layers.TimeDistributed(layer)
            if self.freeze_bn:
                x4 = layer(x4, training=False)
            else:
                x4 = layer(x4)
        x4 = keras.layers.Activation(self.activation_type)(x4)

        if self.data_format == 'channels_first':
            concat_axis = 1
            if self.use_td:
                concat_axis += 1
        else:
            concat_axis = -1
        layer = keras.layers.Concatenate(axis=concat_axis, name='%s_output' % (name_prefix))
        x = layer([x1, x2, x3, x4])
        return x


def update_config(model, inputs, config, name_pattern=None):
    """
    Update the configuration of an existing model.

    Note that the input tensors to apply the new model to must be different
    from those of the original model. This is because when Keras
    clones a model it retains the original input layer and adds an extra one
    on top.

    In order to update the configuration of only certain layers,
    a name pattern (regular expression) may be provided.

    Args:
        model (Model): the model to update the regularizers of.
        inputs (tensors): the tensor to apply the new model to.
        config (dict): dictionary of layer attributes to update.
        name_pattern (str): pattern to match layers against. Those that
            do not match will not be updated.
    """
    # Loop through all layers and update those that have a regularizer.
    for layer in model.layers:
        if name_pattern is None or re.match(name_pattern, layer.name):
            for name, value in config.items():
                if hasattr(layer, name):
                    setattr(layer, name, value)
    new_model = model  # clone_model(model, [inputs])
    new_model.set_weights(model.get_weights())
    return new_model


def update_regularizers(model, inputs, kernel_regularizer, bias_regularizer, name_pattern=None):
    """
    Update the weight decay regularizers of an existing model.

    Note that the input tensors to apply the new model to must be different
    from those of the original model. This is because when Keras
    clones a model it retains the original input layer and adds an extra one
    on top.

    In order to update the regularizers of only certain layers,
    a name pattern (regular expression) may be provided.

    Args:
        model (Model): the model to update the regularizers of.
        inputs (tensors): the tensor to apply the new model to.
        kernel_regularizer (object): regularizer to apply to kernels.
        bias_regularizer (object): regularizer to apply to biases.
        name_pattern (str): pattern to match layers against. Those that
            do not match will not be updated.
    """
    config = {'bias_regularizer': bias_regularizer,
              'kernel_regularizer': kernel_regularizer}
    return update_config(model, inputs, config, name_pattern)


@add_arg_scope
def _conv_block(inputs, filters, alpha, kernel=(3, 3),
                strides=(1, 1), kernel_regularizer=None,
                bias_regularizer=None, use_batch_norm=True,
                activation_type='relu', data_format='channels_first',
                freeze_bn=False, trainable=True,
                use_bias=False):
    """
    Construct a conv block to be used in MobileNet.

    Args:
        inputs(tensor): The input tensor.
        filters(int): The number of filters.
        alpha(float): The alpha parameter for MobileNet to control the final number of filters.
        kernel(int, tuple): The kernel size, can be a int or a tuple.
        strides(int, tuple): The strides.
        kernel_regularizer: Kernel regularizer to be applied to the block.
        bias_regularizer: Bias regularizer to be applied to the block.
        use_batch_norm(bool): Whether or not to use batch normalization layer.
        activation_type(str): Activation type, can be relu or relu6.
        data_format(str): Data format for Keras, can be channels_first or channels_last.
        freeze_bn(bool): Whether or not to freeze the BN layer.
        trainable(bool): Make the conv layer trainable or not.
        use_bias(bool): Whether or not use bias for the conv layer
                        that is immediately before the BN layers.

    Returns:
        The output tensor of this block.
    """
    channel_axis = get_batchnorm_axis(data_format)
    filters = int(filters * alpha)
    # Use explicit padding here to avoid TF asymmetric padding.
    # This will be fused into Conv layer, and TRT inference is faster than TF asymmetric padding.
    # For accuracy, we found they are almost the same for the two padding styles.
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = keras.layers.Conv2D(
        filters,
        kernel,
        padding='valid',
        use_bias=use_bias,
        strides=strides,
        name='conv1',
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        trainable=trainable)(x)

    if use_batch_norm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(axis=channel_axis,
                                                name='conv1_bn')(x, training=False)
        else:
            x = keras.layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    if activation_type == 'relu6':
        x = keras.layers.ReLU(6., name='conv_block_relu6')(x)
    else:
        x = keras.layers.ReLU(name='conv_block_relu')(x)
    return x


@add_arg_scope
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1),
                          block_id=1, kernel_regularizer=None,
                          bias_regularizer=None, use_batch_norm=True,
                          activation_type='relu', data_format='channels_first',
                          freeze_bn=False, trainable=True,
                          use_bias=False):
    """
    Depthwise conv block as building blocks for MobileNet.

    Args:
        inputs(tensor): The input tensor.
        pointwise_conv_filters(int): The number of depthwise conv filters.
        alpha(float): The alpha parameter for MobileNet.
        depth_multiplier(int): The depth multiplier(defaut: 1)
        strides(int, tuple): The strides, can be a int or a tuple.
        block_id(int): The block_id, used to name the blocks.
        kernel_regularizer: The kernel regularizer.
        bias_regularizer: The bias regularizer.
        use_batch_norm(bool): Whether or not to use batch normalization layer.
        activation_type(str): Activation type, can be relu or relu6.
        data_format(str): Data format for Keras, can be channels_first or channels_last.
        freeze_bn(bool): Whether or not to freeze the BN layer.
        trainable(bool): Make the conv layer trainable or not.
        use_bias(bool): Whether or not use bias for the conv layer
                        that is immediately before the BN layers.

    Returns:
        The output tensor.

    """
    channel_axis = get_batchnorm_axis(data_format)
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    # Also use explicit padding here to avoid TF style padding.
    x = keras.layers.ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = keras.layers.DepthwiseConv2D(
        (3, 3),
        padding='valid',
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=use_bias,
        name='conv_dw_%d' % block_id,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        trainable=trainable)(x)

    if use_batch_norm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(
                axis=channel_axis,
                name='conv_dw_%d_bn' % block_id)(x, training=False)
        else:
            x = keras.layers.BatchNormalization(axis=channel_axis,
                                                name='conv_dw_%d_bn' % block_id)(x)

    if activation_type == 'relu6':
        x = keras.layers.ReLU(6., name='conv_dw_%d_relu6' % block_id)(x)
    else:
        x = keras.layers.ReLU(name='conv_dw_%d_relu' % block_id)(x)

    x = keras.layers.Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding='valid',
        use_bias=use_bias,
        strides=(1, 1),
        name='conv_pw_%d' % block_id,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        trainable=trainable)(x)

    if use_batch_norm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(
                axis=channel_axis,
                name='conv_pw_%d_bn' % block_id)(x, training=False)
        else:
            x = keras.layers.BatchNormalization(
                axis=channel_axis,
                name='conv_pw_%d_bn' % block_id)(x)

    if activation_type == 'relu6':
        x = keras.layers.ReLU(6., name='conv_pw_relu6_%d' % block_id)(x)
    else:
        x = keras.layers.ReLU(name='conv_pw_relu_%d' % block_id)(x)
    return x


@add_arg_scope
def _leaky_conv(inputs, filters, alpha=0.1, kernel=(3, 3),
                strides=(1, 1), kernel_regularizer=None,
                bias_regularizer=None, use_batch_norm=True,
                padding='same', data_format='channels_first',
                freeze_bn=False, trainable=True, force_relu=False,
                use_bias=False, name='conv1', use_td=False):
    """
    Construct a leaky relu conv block to be used in DarkNet.

    Args:
        inputs(tensor): The input tensor.
        filters(int): The number of filters.
        alpha(float): leaky rate for LeakyReLU
        kernel(int, tuple): The kernel size, can be a int or a tuple.
        strides(int, tuple): The strides.
        padding(str): same or valid.
        kernel_regularizer: Kernel regularizer to be applied to the block.
        bias_regularizer: Bias regularizer to be applied to the block.
        use_batch_norm(bool): Whether or not to use batch normalization layer.
        activation_type(str): Activation type, can be relu or relu6.
        data_format(str): Data format for Keras, can be channels_first or channels_last.
        freeze_bn(bool): Whether or not to freeze the BN layer.
        trainable(bool): Make the conv layer trainable or not.
        force_relu(bool): For ReLU activation.
        use_bias(bool): Whether or not use bias for the conv layer
                        that is immediately before the BN layers.
        name(str): name of the layer.
        use_td(bool): use TimeDistributed wrapper or not, default is False.

    Returns:
        The output tensor of this block.
    """
    channel_axis = get_batchnorm_axis(data_format)
    _layer = keras.layers.Conv2D(
        filters,
        kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=None,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_bias=use_bias,
        trainable=trainable,
        name=name)
    if use_td:
        _layer = keras.layers.TimeDistributed(_layer)
    x = _layer(inputs)

    if use_batch_norm:
        _layer = keras.layers.BatchNormalization(axis=channel_axis, name=name+'_bn')
        if use_td:
            _layer = keras.layers.TimeDistributed(_layer)
        if freeze_bn:
            x = _layer(x, training=False)
        else:
            x = _layer(x)

    if force_relu:
        # still use _lrelu as name
        x = keras.layers.ReLU(name=name+'_lrelu')(x)
    else:
        x = keras.layers.LeakyReLU(alpha=alpha, name=name+'_lrelu')(x)
    return x


@add_arg_scope
def _mish_conv(inputs, filters, kernel=(3, 3),
               strides=(1, 1), kernel_regularizer=None,
               bias_regularizer=None, use_batch_norm=True,
               padding='same', data_format='channels_first',
               freeze_bn=False, trainable=True, force_relu=False,
               use_bias=False, name='conv1', use_td=False,
               activation="leaky_relu"):
    """
    Construct a mish conv block to be used in DarkNet.

    Args:
        inputs(tensor): The input tensor.
        filters(int): The number of filters.
        kernel(int, tuple): The kernel size, can be a int or a tuple.
        strides(int, tuple): The strides.
        padding(str): same or valid.
        kernel_regularizer: Kernel regularizer to be applied to the block.
        bias_regularizer: Bias regularizer to be applied to the block.
        use_batch_norm(bool): Whether or not to use batch normalization layer.
        data_format(str): Data format for Keras, can be channels_first or channels_last.
        freeze_bn(bool): Whether or not to freeze the BN layer.
        trainable(bool): Make the conv layer trainable or not.
        force_relu(bool): Whether to use ReLU instead of Mish
        use_bias(bool): Whether or not use bias for the conv layer
                        that is immediately before the BN layers.
        name(str): name of the layer.
        use_td(bool): use TimeDistributed wrapper or not, default is False.
        activation(str): Activation type.

    Returns:
        The output tensor of this block.
    """
    channel_axis = get_batchnorm_axis(data_format)
    _layer = keras.layers.Conv2D(filters,
                                 kernel,
                                 strides=strides,
                                 padding=padding,
                                 data_format=data_format,
                                 activation=None,
                                 kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 use_bias=use_bias,
                                 trainable=trainable,
                                 name=name)
    if use_td:
        _layer = keras.layers.TimeDistributed(_layer)
    x = _layer(inputs)

    if use_batch_norm:
        _layer = keras.layers.BatchNormalization(
            axis=channel_axis, name=name+'_bn'
        )
        if use_td:
            _layer = keras.layers.TimeDistributed(_layer)
        if freeze_bn:
            x = _layer(x, training=False)
        else:
            x = _layer(x)

    if force_relu:
        # TODO(@zhimengf): This should be deprecated in the future
        # Use the general yolov4_config.activation parameter instead
        # For now, let's keep it for backward compatibility of spec
        x = keras.layers.ReLU(name=name+'_mish')(x)
    elif activation == "mish":
        x = keras.layers.Activation(mish, name=name+'_mish')(x)
    elif activation == "relu":
        # still use _mish as name
        x = keras.layers.ReLU(name=name+'_mish')(x)
    else:
        # default case: LeakyReLU
        x = keras.layers.LeakyReLU(alpha=0.1, name=name+'_mish')(x)
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@add_arg_scope
def _inverted_res_block(inputs, expansion, stride, alpha, filters,
                        block_id, kernel_regularizer=None, bias_regularizer=None,
                        use_batch_norm=True, activation_type='relu',
                        data_format='channels_first', all_projections=True,
                        trainable=True, freeze_bn=False,
                        use_bias=False):
    """
    Inverted residual block as building blocks for MobileNet V2.

    Args:
        inputs(tensor): Input tensor.
        expansion(float): Expansion factor of the filter numbers.
        stride(int, tuple): Stride of this block.
        alpha(float): alpha parameter.
        filters(int): Number of filters.
        block_id(int): block id for this block, as a name.
        kernel_regularizer: Kernel regularizer to be applied.
        bias_regularizer: Bias regularizer to be applied.
        use_batch_norm(bool): Whether or not to use BN layers.
        activation_type(str): Activation type, can be relu or relu6.
        data_format(str): Data format, can be channels_first or channels_last.
        all_projections(bool): Whether to use all projection layers to replace the shortcuts.
        freeze_bn(bool): Whether or not to freeze the BN layer.
        trainable(bool): Make the conv layer trainable or not.
        use_bias(bool): Whether or not use bias for the conv layer
                        that is immediately before the BN layers.

    Returns:
        The output tensor.

    """
    channel_axis = get_batchnorm_axis(data_format)
    in_channels = inputs._keras_shape[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = keras.layers.Conv2D(
                expansion * in_channels,
                kernel_size=1,
                padding='valid',
                use_bias=use_bias,
                activation=None,
                name=prefix + 'expand',
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                trainable=trainable)(x)

        if use_batch_norm:
            if freeze_bn:
                x = keras.layers.BatchNormalization(
                        epsilon=1e-3,
                        axis=channel_axis,
                        momentum=0.999,
                        name=prefix + 'expand_bn')(x, training=False)
            else:
                x = keras.layers.BatchNormalization(
                        epsilon=1e-3,
                        axis=channel_axis,
                        momentum=0.999,
                        name=prefix + 'expand_bn')(x)
        if activation_type == 'relu6':
            x = keras.layers.ReLU(6., name='re_lu_%d' % (block_id + 1))(x)
        else:
            x = keras.layers.ReLU(name='re_lu_%d' % (block_id + 1))(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    # Use explicit padding
    x = keras.layers.ZeroPadding2D((1, 1), name=prefix + 'depthwise_pad')(x)
    x = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            activation=None,
            use_bias=use_bias,
            padding='valid',
            name=prefix + 'depthwise',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=trainable)(x)
    if use_batch_norm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(
                    epsilon=1e-3,
                    axis=channel_axis,
                    momentum=0.999,
                    name=prefix + 'depthwise_bn')(x, training=False)
        else:
            x = keras.layers.BatchNormalization(
                    epsilon=1e-3,
                    axis=channel_axis,
                    momentum=0.999,
                    name=prefix + 'depthwise_bn')(x)

    if activation_type == 'relu6':
        x = keras.layers.ReLU(6., name=prefix + 'relu6')(x)
    else:
        x = keras.layers.ReLU(name=prefix + 'relu')(x)
    # Project
    x = keras.layers.Conv2D(
            pointwise_filters,
            kernel_size=1,
            padding='valid',
            use_bias=use_bias,
            activation=None,
            name=prefix + 'project',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=trainable)(x)
    if use_batch_norm:
        if freeze_bn:
            x = keras.layers.BatchNormalization(
                    axis=channel_axis,
                    epsilon=1e-3,
                    momentum=0.999,
                    name=prefix + 'project_bn')(x, training=False)
        else:
            x = keras.layers.BatchNormalization(
                    axis=channel_axis,
                    epsilon=1e-3,
                    momentum=0.999,
                    name=prefix + 'project_bn')(x)

    if in_channels == pointwise_filters and stride == 1:
        if all_projections:
            inputs_projected = keras.layers.Conv2D(
                    in_channels,
                    kernel_size=1,
                    padding='valid',
                    use_bias=False,
                    activation=None,
                    name=prefix + 'projected_inputs',
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    trainable=trainable)(inputs)
            return keras.layers.Add(name=prefix + 'add')([inputs_projected, x])
        return keras.layers.Add(name=prefix + 'add')([inputs, x])

    return x


def get_uid(base_name):
    """Return a unique ID."""
    get_uid.lock.acquire()
    if base_name not in get_uid.seqn:
        get_uid.seqn[base_name] = 0
    uid = get_uid.seqn[base_name]
    get_uid.seqn[base_name] += 1
    get_uid.lock.release()
    return uid


get_uid.seqn = {}
get_uid.lock = threading.Lock()


def add_activation(activation_type, **kwargs):
    """
    Create an activation layer based on activation type and additional arguments.

    Note that the needed kwargs depend on the activation type.

    Args:
        activation_type (str): String to indicate activation type.
        kwargs (dict): Additional keyword arguments depending on the activation type.

    Returns:
        activation_layer (a subclass of keras.layers.Layer): The layer type
            depends on activation_type.
    """
    if activation_type == 'relu-n':
        max_value = kwargs.get('max_value', None)
        activation_layer = keras.layers.ReLU(max_value=max_value)

    elif activation_type == 'lrelu':
        alpha = kwargs['alpha']
        activation_layer = keras.layers.LeakyReLU(alpha=alpha)
    elif activation_type == 'elu':
        alpha = kwargs['alpha']
        activation_layer = keras.layers.ELU(alpha=alpha)

    else:
        activation_layer = keras.layers.Activation(activation_type, **kwargs)

    return activation_layer


class CNNBlock(object):
    """A functor for creating a block of layers."""

    @add_arg_scope
    def __init__(self,
                 use_batch_norm,
                 use_shortcuts,
                 data_format,
                 kernel_regularizer,
                 bias_regularizer,
                 repeat,
                 stride,
                 subblocks,
                 index=None,
                 activation_type='relu',
                 freeze_bn=False,
                 freeze_block=False,
                 activation_kwargs=None,
                 dilation_rate=(1, 1),
                 all_projections=False,
                 use_bias=True,
                 use_td=False):
        """
        Initialization of the block functor object.

        Args:
            use_batch_norm (bool): whether batchnorm should be added after each convolution.
            use_shortcuts (bool): whether shortcuts should be used. A typical ResNet by definition
                uses shortcuts, but these can be toggled off to use the same ResNet topology without
                the shortcuts.
            data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
            kernel_regularizer (float): regularizer to apply to kernels.
            bias_regularizer (float): regularizer to apply to biases.
            repeat (int): repeat number.
            stride (int): The filter stride to be applied only to the first subblock (typically used
                for downsampling). Strides are set to 1 for all layers beyond the first subblock.
            subblocks (list of tuples): A list of tuples defining settings for each consecutive
                convolution. Example:
                    `[(3, 64), (3, 64)]`
                The two items in each tuple represents the kernel size and the amount of filters in
                a convolution, respectively. The convolutions are added in the order of the list.
            index (int): the index of the block to be created.
            activation_type (str): activation function type.
            activation_kwargs (dict): Additional activation keyword arguments to be fed to
                the add_activation function.
            dilation_rate (int or (int, int)): An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            all_projections (bool): A boolean flag to determinte whether all shortcut connections
                should be implemented as projection layers to facilitate full pruning or not.
            use_bias (bool): whether the layer uses a bias vector.
            use_td (bool): Whether or not to wrap the layers into a TimeDistributed layer
                to make it work for 5D tensors.
        """
        self.use_batch_norm = use_batch_norm
        self.use_shortcuts = use_shortcuts
        self.all_projections = all_projections
        self.data_format = data_format
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation_type = activation_type
        self.activation_kwargs = activation_kwargs or {}
        self.dilation_rate = dilation_rate
        self.repeat = repeat
        self.stride = stride
        self.use_bias = use_bias
        self.subblocks = subblocks
        self.subblock_ids = subblock_ids()
        self.freeze_bn = freeze_bn
        self.freeze_block = freeze_block
        self.use_td = use_td
        if index is not None:
            self.name = 'block_%d' % index
        else:
            self.name = 'block_%d' % (get_uid('block') + 1)

    def __call__(self, x):
        """Build the block.

        Args:
            x (tensor): input tensor.

        Returns:
            tensor: the output tensor after applying the block on top of input `x`.
        """
        for i in range(self.repeat):
            name = '%s%s_' % (self.name, self.subblock_ids[i])
            if i == 0:
                # Set the stride only on the first layer.
                stride = self.stride
                dimension_changed = True
            else:
                stride = 1
                dimension_changed = False

            x = self._subblocks(x,
                                stride,
                                dimension_changed,
                                name_prefix=name,
                                freeze=self.freeze_block,
                                use_td=self.use_td)

        return x

    def _subblocks(self, x, stride, dimension_changed, name_prefix=None, freeze=False,
                   use_td=False):
        """
        Stack several convolutions in a specific sequence given by a list of subblocks.

        Args:
            x (tensor): the input tensor.
            stride (int): The filter stride to be applied only to the first subblock (typically used
                for downsampling). Strides are set to 1 for all layers beyond the first subblock.
            dimension_changed (bool): This indicates whether the dimension has been changed for this
                block. If this is true, then we need to account for the change, or else we will be
                unable to re-add the shortcut tensor due to incompatible dimensions. This can be
                solved by applying a (1x1) convolution [1]. (The paper also notes the possibility of
                zero-padding the shortcut tensor to match any larger output dimension, but this is
                not implemented.)
            name_prefix (str): name prefix for all the layers created in this function.
            freeze (bool): Whether or not to freeze this block.
            use_td (bool): Whether or not to wrap layers into a TimeDistributed layer to make it
                work for 5D tensors.

        Returns:
            tensor: the output tensor after applying the ResNet block on top of input `x`.
        """
        bn_axis = get_batchnorm_axis(self.data_format)

        shortcut = x
        nblocks = len(self.subblocks)
        for i in range(nblocks):
            kernel_size, filters = self.subblocks[i]
            if i == 0:
                strides = (stride, stride)
            else:
                strides = (1, 1)
            layer = keras.layers.Conv2D(
                    filters, (kernel_size, kernel_size),
                    strides=strides,
                    padding='same',
                    dilation_rate=self.dilation_rate,
                    data_format=self.data_format,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name='%sconv_%d' % (name_prefix, i + 1),
                    trainable=not freeze)
            if use_td:
                layer = keras.layers.TimeDistributed(layer)
            x = layer(x)
            if self.use_batch_norm:
                layer = keras.layers.BatchNormalization(
                    axis=bn_axis,
                    name='%sbn_%d' % (name_prefix, i + 1)
                )
                if use_td:
                    layer = keras.layers.TimeDistributed(layer)
                if self.freeze_bn:
                    x = layer(x, training=False)
                else:
                    x = layer(x)
            if i != nblocks - 1:  # All except last conv in block.
                x = add_activation(self.activation_type,
                                   name='%s%s_%d' % (name_prefix, self.activation_type, i + 1))(x)

        if self.use_shortcuts:
            if self.all_projections:
                # Implementing shortcut connections as 1x1 projection layers irrespective of
                # dimension change.
                layer = keras.layers.Conv2D(
                    filters, (1, 1),
                    strides=(stride, stride),
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name='%sconv_shortcut' % name_prefix,
                    trainable=not freeze)
                if use_td:
                    layer = keras.layers.TimeDistributed(layer)
                shortcut = layer(shortcut)
                if self.use_batch_norm:
                    _name = '%sbn_shortcut' % name_prefix
                    layer = keras.layers.BatchNormalization(axis=bn_axis,
                                                            name=_name)
                    if use_td:
                        layer = keras.layers.TimeDistributed(layer)
                    if self.freeze_bn:
                        shortcut = layer(shortcut, training=False)
                    else:
                        shortcut = layer(shortcut)
            else:
                # Add projection layers to shortcut only if there is a change in dimesion.
                if dimension_changed:  # Dimension changed.
                    layer = keras.layers.Conv2D(
                                filters, (1, 1),
                                strides=(stride, stride),
                                data_format=self.data_format,
                                dilation_rate=self.dilation_rate,
                                use_bias=self.use_bias,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                name='%sconv_shortcut' % name_prefix,
                                trainable=not freeze)
                    if use_td:
                        layer = keras.layers.TimeDistributed(layer)
                    shortcut = layer(shortcut)
                    if self.use_batch_norm:
                        _name = '%sbn_shortcut' % name_prefix
                        layer = keras.layers.BatchNormalization(
                            axis=bn_axis,
                            name=_name)
                        if use_td:
                            layer = keras.layers.TimeDistributed(layer)
                        if self.freeze_bn:
                            shortcut = layer(shortcut, training=False)
                        else:
                            shortcut = layer(shortcut)
            x = keras.layers.add([x, shortcut])

        x = add_activation(self.activation_type,
                           name='%s%s' % (name_prefix, self.activation_type))(x)

        return x


@add_arg_scope
def fire_module(inputs, block_id, squeeze, expand, kernel_regularizer=None,
                bias_regularizer=None, data_format='channels_first',
                trainable=True):
    """
    The squeeze net fire module architecture.

    For details, see https://arxiv.org/pdf/1602.07360.pdf

    Args:
        inputs(tensor): Input tensor.
        block_id(int): Block id for current module
        squeeze(int): number of filters for squeeze conv layer
        expand(int): number of filters for expand conv layers (1x1 and 3x3)
        kernel_regularizer: Kernel regularizer applied to the model.
        bias_regularizer: Bias regularizer applied to the model.
        data_format(str): Data format, can be channels_first or channels_last.
        trainable(bool): whether to make the conv layer trainable or not.

    Returns:
        The output tensor.
    """
    concat_axis = 1 if data_format == 'channels_first' else 3

    x = keras.layers.Conv2D(
            squeeze,
            kernel_size=(1, 1),
            padding='same',
            name='fire' + str(block_id) + '_squeeze_conv',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            data_format=data_format,
            trainable=trainable)(inputs)
    x = keras.layers.Activation('relu', name='fire' + str(block_id) + '_squeeze')(x)
    b_1x1 = keras.layers.Conv2D(
                expand,
                kernel_size=(1, 1),
                padding='same',
                name='fire' + str(block_id) + '_expand_conv1x1',
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                data_format=data_format,
                trainable=trainable)(x)
    b_1x1 = keras.layers.Activation('relu', name='fire' + str(block_id) + '_expand_1x1')(b_1x1)
    b_3x3 = keras.layers.Conv2D(
                expand,
                kernel_size=(3, 3),
                padding='same',
                name='fire' + str(block_id) + '_expand_conv3x3',
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                data_format=data_format,
                trainable=trainable)(x)
    b_3x3 = keras.layers.Activation('relu', name='fire' + str(block_id) + '_expand_3x3')(b_3x3)
    return keras.layers.Concatenate(axis=concat_axis, name='fire' + str(block_id))([b_1x1, b_3x3])


def swish(x):
    """Swish activation function.

    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    return x * keras.backend.sigmoid(x)


def mish(x):
    """Mish activation function.

    See details: https://arxiv.org/pdf/1908.08681.pdf

    Args:
        x: input tensor
    Returns:
        mish(x) = x * tanh(ln(1 + e^x))
    """

    return x * tf.math.tanh(tf.math.softplus(x))


initializer_distribution = "normal"
if os.getenv("TF_KERAS", "0") == "1":
    initializer_distribution = "untruncated_normal"
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': initializer_distribution
    }
}


DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if keras.backend.image_data_format() == 'channels_first' else 1
    input_size = keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def round_filters(filters, divisor, width_coefficient):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


DEFAULT_DATA_FORMAT = "channels_last" if os.getenv("TF_KERAS", "0") == "1" else "channels_first"


def block(inputs, activation_fn=swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True, freeze=False,
          freeze_bn=False, use_td=False, kernel_regularizer=None,
          bias_regularizer=None, use_bias=False, data_format=DEFAULT_DATA_FORMAT):
    """A mobile inverted residual block.

    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
        freeze(bool): Freeze this block or not.
        freeze_bn(bool): Freeze all the BN layers in this block or not.
        use_td(bool): Use TimeDistributed wrapper layers for this block or not.
            This is used to support 5D input tensors, e.g. in FasterRCNN use case.
        kernel_regularizer: The kernel regularizer.
        bias_regularizer: The bias regularizer.
        use_bias(bool): Use bias or not for Conv layers followed by a BN layer.
    # Returns
        output tensor for the block.
    """
    bn_opt = {
        'momentum': 0.99,
        'epsilon': 1e-3
    }
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        layer = keras.layers.Conv2D(
            filters,
            1,
            padding='same',
            use_bias=use_bias,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=not freeze,
            data_format=data_format,
            name=name + 'expand_conv'
        )
        if use_td:
            layer = keras.layers.TimeDistributed(layer)
        x = layer(inputs)
        layer = keras.layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn',
                                                **bn_opt)
        if use_td:
            layer = keras.layers.TimeDistributed(layer)
        if freeze_bn:
            x = layer(x, training=False)
        else:
            x = layer(x)
        x = keras.layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        layer = keras.layers.ZeroPadding2D(
            padding=correct_pad(x, kernel_size),
            data_format=data_format,
            name=name + 'dwconv_pad'
        )
        if use_td:
            layer = keras.layers.TimeDistributed(layer)
        x = layer(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    layer = keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=use_bias,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        data_format=data_format,
        trainable=not freeze,
        name=name + 'dwconv'
    )
    if use_td:
        layer = keras.layers.TimeDistributed(layer)
    x = layer(x)
    layer = keras.layers.BatchNormalization(axis=bn_axis, name=name + 'bn',
                                            **bn_opt)
    if use_td:
        layer = keras.layers.TimeDistributed(layer)
    if freeze_bn:
        x = layer(x, training=False)
    else:
        x = layer(x)
    x = keras.layers.Activation(activation_fn, name=name + 'activation')(x)
    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        # Global pooling is needed if we are going to support dynamic
        # input shape(e.g., in FasterRCNN) for this backbone
        # AveragePooling2D requires static input shape, hence cannot work with
        # dynamic shapes
        if use_td:
            # GlobalAveragePooling2D cannot work well with TimeDistributed layer
            # because when converted to UFF, GlobalAveragePooling2D becomes Mean
            # Op in UFF, and it cannot handle 5D input by itself like Conv2D does.
            # So we rely on some manual shape transforms, so it sees 4D input
            # (N, R*C, H, W), and reshape back to (N, R, C, 1, 1) after global pooling.
            R, C, H, W = x.get_shape().as_list()[1:]
            assert None not in (R, C, H, W), (
                "Expect R, C, H, W all not None. While got {}".format((R, C, H, W))
            )
            # Another issue is for pruning. Reshape cannot follow a pruned layer
            # in modulus pruning due to dimension change after pruning.
            # while for current special case, we essentially reshape to (N, -1, H, W)
            # whenever the filter number C changes or not during pruning.
            # So in this case, the logic is still correct even if the number C is changed.
            # But we cannot hard-code the target shape to (R*C, H, W) in case C changes.
            # Instead, the target shape is actually (N, -1, H, W) whenever C changes or not.
            se = keras.layers.Reshape((-1, H, W), name=name + 'pre_pool_reshape')(x)
            se = keras.layers.GlobalAveragePooling2D(
                data_format=data_format, name=name + 'se_squeeze')(se)
            layer = keras.layers.Reshape((R, -1, 1, 1), name=name + 'post_pool_reshape')
            se = layer(se)
        else:
            se = keras.layers.GlobalAveragePooling2D(
                data_format=data_format, name=name + 'se_squeeze')(x)
            # _, cc = se.get_shape()
            se_shape = (1, 1, -1) if data_format == 'channels_last' else (-1, 1, 1)
            se = keras.layers.Reshape(se_shape, name=name + 'se_reshape')(se)
        layer = keras.layers.Conv2D(
            filters_se,
            1,
            padding='same',
            activation=activation_fn,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=not freeze,
            data_format=data_format,
            name=name + 'se_reduce'
        )
        if use_td:
            layer = keras.layers.TimeDistributed(layer)
        se = layer(se)
        layer = keras.layers.Conv2D(
            filters,
            1,
            padding='same',
            activation='sigmoid',
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            data_format=data_format,
            trainable=not freeze,
            name=name + 'se_expand'
        )
        if use_td:
            layer = keras.layers.TimeDistributed(layer)
        se = layer(se)
        x = keras.layers.Multiply(name=name + 'se_excite')([x, se])

    # Output phase
    layer = keras.layers.Conv2D(
        filters_out,
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        trainable=not freeze,
        data_format=data_format,
        name=name + 'project_conv'
    )
    if use_td:
        layer = keras.layers.TimeDistributed(layer)
    x = layer(x)
    layer = keras.layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn',
                                            **bn_opt)
    if use_td:
        layer = keras.layers.TimeDistributed(layer)
    if freeze_bn:
        x = layer(x, training=False)
    else:
        x = layer(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            layer = keras.layers.Dropout(
                drop_rate,
                noise_shape=(None, 1, 1, 1),
                name=name + 'drop',
            )
            if use_td:
                layer = keras.layers.TimeDistributed(layer)
            x = layer(x)
        x = keras.layers.Add(name=name + 'add')([x, inputs])

    return x


def force_stride16(block_args):
    """Force the block args to make the model have stride 16."""
    last_block = -1
    for idx, block in enumerate(block_args):
        if block['strides'] == 2:
            last_block = idx
    assert last_block >= 0, (
        "Cannot find stride 2 in the block args."
    )
    # pop the layer with last stride 2 and following layers
    # to keep the total stride of 16
    block_args = block_args[:last_block]


def add_dense_head(model, inputs, nclasses, activation):
    """
    Create a model that stacks a dense head on top of a another model. It is also flattened.

    Args:
        model (Model): the model on top of which the head should be created.
        inputs (tensor): the inputs (tensor) to the previously supplied model.
        nclasses (int): the amount of outputs of the dense map
        activation (string): activation function to use e.g. 'softmax' or 'linear'.

    Returns:
        Model: A model with the head stacked on top of the `model` input.
    """
    x = model.outputs[0]
    head_name = "head_fc%d" % (nclasses)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(nclasses, activation=activation, name=head_name)(x)
    model = keras.models.Model(
        inputs=inputs, outputs=x, name="%s_fc%d" % (model.name, nclasses)
    )
    return model


def csp_tiny_block(x, num_filters, name, trainable=True, kernel_regularizer=None,
                   bias_regularizer=None, data_format="channels_first",
                   freeze_bn=False, force_relu=False, use_bias=False, use_td=False,
                   use_batch_norm=True, activation="leaky_relu"):
    """Building block for CSPDarkNet tiny."""
    concat_axis = 1 if data_format == "channels_first" else -1
    with arg_scope(
        [_mish_conv],
        use_batch_norm=use_batch_norm,
        data_format=data_format,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        padding='same',
        freeze_bn=freeze_bn,
        use_bias=use_bias,
        force_relu=force_relu,
        trainable=trainable,
        activation=activation
    ):
        x = _mish_conv(x, num_filters, kernel=(3, 3), name=name+"_conv_0")
        route = x
        x = Split(groups=2, group_id=1, name=name+"_split_0")(x)
        x = _mish_conv(x, num_filters // 2, kernel=(3, 3), name=name+"_conv_1")
        route_1 = x
        x = _mish_conv(x, num_filters // 2, kernel=(3, 3), name=name+"_conv_2")
        x = keras.layers.Concatenate(axis=concat_axis, name=name+"_concat_0")([x, route_1])
        x = _mish_conv(x, num_filters, kernel=(1, 1), name=name+"_conv_3")
        x = keras.layers.Concatenate(axis=concat_axis, name=name+"_concat_1")([route, x])
        x = keras.layers.MaxPooling2D(
            pool_size=[2, 2], name=name+"_pool_0",
            data_format=data_format)(x)
        return x
