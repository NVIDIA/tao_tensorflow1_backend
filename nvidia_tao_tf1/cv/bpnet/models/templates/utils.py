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
"""Modulus utilities for model templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import keras

from nvidia_tao_tf1.core.decorators.arg_scope import add_arg_scope
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.utils import add_activation, get_batchnorm_axis
from nvidia_tao_tf1.core.models.templates.utils import SUBBLOCK_IDS
from nvidia_tao_tf1.core.utils import get_uid

logger = logging.getLogger(__name__)


def add_input(channels=3,
              height=256,
              width=256,
              name='inputs',
              data_format='channels_first'):
    """
    Build sample input for testing.

    Args:
        name (str): Name of the input tensor. Default value is 'inputs'
        data_format (str): Expected tensor format, either `channels_first` or `channels_last`.
            Default value is `channels_first`.
        channels, height, width (all int): Input image dimentions.
    """

    # Set sample inputs.
    if data_format == 'channels_first':
        shape = (channels, height, width)
    elif data_format == 'channels_last':
        shape = (height, width, channels)
    else:
        raise ValueError(
            'Provide either `channels_first` or `channels_last` for `data_format`.'
        )
    input_tensor = keras.layers.Input(shape=shape, name=name)
    return input_tensor


class CNNBlock(object):
    """A functor for creating a block of layers.

    Modified version of modulus. The difference is in the way dilation rate is being used. The one in modulus will apply
    it to all layers. Here we add another argument 'first_subblock_dilation_rate' and pass to the
    subblock function where dilation is applied only to the first subblock, similar to stride.
    Dilations are set to 'dilation_rate' for all layers beyond the first subblock.
    """

    @add_arg_scope
    def __init__(
        self,
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
        activation_kwargs=None,
        dilation_rate=(1, 1),
        first_subblock_dilation_rate=None,
        all_projections=False,
        use_bias=True,
        name_prefix=None,
        quantize=False,
        bitwidth=8,
    ):
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
            first_subblock_dilation_rate (int): The dilation to be applied only to first subblock
                (typically used instead of downsampling). Dilations are set to 'dilation_rate'
                for all layers beyond the first subblock.
            all_projections (bool): A boolean flag to determinte whether all shortcut connections
                should be implemented as projection layers to facilitate full pruning or not.
            use_bias (bool): whether the layer uses a bias vector.
            name_prefix (str): Prefix the name with this value.
            quantize (bool): A boolean flag to determine whether to use quantized conv2d or not.
            bitwidth (integer): quantization bitwidth.
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
        self.first_subblock_dilation_rate = first_subblock_dilation_rate
        self.repeat = repeat
        self.stride = stride
        self.use_bias = use_bias
        self.subblocks = subblocks
        self.subblock_ids = SUBBLOCK_IDS()
        self.quantize = quantize
        self.bitwidth = bitwidth
        if index is not None:
            self.name = "block_%d" % index
        else:
            self.name = "block_%d" % (get_uid("block") + 1)
        if name_prefix is not None:
            self.name = name_prefix + "_" + self.name

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
                first_subblock_dilation_rate = self.first_subblock_dilation_rate
                dimension_changed = True
            else:
                stride = 1
                first_subblock_dilation_rate = None
                dimension_changed = False

            x = self._subblocks(x,
                                stride,
                                first_subblock_dilation_rate,
                                dimension_changed,
                                name_prefix=name)

        return x

    def _subblocks(self,
                   x,
                   stride,
                   first_subblock_dilation_rate,
                   dimension_changed,
                   name_prefix=None):
        """
        Stack several convolutions in a specific sequence given by a list of subblocks.

        Args:
            x (tensor): the input tensor.
            stride (int): The filter stride to be applied only to the first subblock (typically used
                for downsampling). Strides are set to 1 for all layers beyond the first subblock.
            first_subblock_dilation_rate (int): The dilation to be applied only to first subblock
                (typically used instead of downsampling). Dilations are set to 'dilation_rate'
                for all layers beyond the first subblock.
            dimension_changed (bool): This indicates whether the dimension has been changed for this
                block. If this is true, then we need to account for the change, or else we will be
                unable to re-add the shortcut tensor due to incompatible dimensions. This can be
                solved by applying a (1x1) convolution [1]. (The paper also notes the possibility of
                zero-padding the shortcut tensor to match any larger output dimension, but this is
                not implemented.)
            name_prefix (str): name prefix for all the layers created in this function.

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

            if i == 0 and self.first_subblock_dilation_rate is not None:
                # if first block, use dilation rate from the first_subblock_dilation_rate
                dilation_rate = self.first_subblock_dilation_rate
            else:
                # if not fist block, use the common dilation rate
                dilation_rate = self.dilation_rate

            # Keras doesn't support dilation_rate != 1 if stride != 1.
            if strides != (1, 1) and dilation_rate != (1, 1):
                dilation_rate = (1, 1)
                logger.warning(
                    "Dilation rate {} is incompatible with stride {}. "
                    "Setting dilation rate to {} for layer {}conv_{}.".format(
                        self.dilation_rate, strides, dilation_rate,
                        name_prefix, i + 1))
            if self.quantize:
                x = QuantizedConv2D(
                    filters,
                    (kernel_size, kernel_size),
                    strides=strides,
                    padding="same",
                    dilation_rate=dilation_rate,
                    data_format=self.data_format,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    bitwidth=self.bitwidth,
                    name="%sconv_%d" % (name_prefix, i + 1),
                )(x)
            else:
                x = keras.layers.Conv2D(
                    filters,
                    (kernel_size, kernel_size),
                    strides=strides,
                    padding="same",
                    dilation_rate=dilation_rate,
                    data_format=self.data_format,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="%sconv_%d" % (name_prefix, i + 1),
                )(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=bn_axis,
                                                    name="%sbn_%d" %
                                                    (name_prefix, i + 1))(x)
            if i != nblocks - 1:  # All except last conv in block.
                x = add_activation(self.activation_type,
                                   **self.activation_kwargs)(x)

        if self.use_shortcuts:
            if self.all_projections:
                # Implementing shortcut connections as 1x1 projection layers irrespective of
                # dimension change.
                if self.quantize:
                    shortcut = QuantizedConv2D(
                        filters,
                        (1, 1),
                        strides=(stride, stride),
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate,
                        use_bias=self.use_bias,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        bitwidth=self.bitwidth,
                        name="%sconv_shortcut" % name_prefix,
                    )(shortcut)
                else:
                    shortcut = keras.layers.Conv2D(
                        filters,
                        (1, 1),
                        strides=(stride, stride),
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate,
                        use_bias=self.use_bias,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name="%sconv_shortcut" % name_prefix,
                    )(shortcut)
                if self.use_batch_norm:
                    shortcut = keras.layers.BatchNormalization(
                        axis=bn_axis,
                        name="%sbn_shortcut" % name_prefix)(shortcut)
            else:
                # Add projection layers to shortcut only if there is a change in dimesion.
                if dimension_changed:  # Dimension changed.
                    if self.quantize:
                        shortcut = QuantizedConv2D(
                            filters,
                            (1, 1),
                            strides=(stride, stride),
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate,
                            use_bias=self.use_bias,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            bitwidth=self.bitwidth,
                            name="%sconv_shortcut" % name_prefix,
                        )(shortcut)
                    else:
                        shortcut = keras.layers.Conv2D(
                            filters,
                            (1, 1),
                            strides=(stride, stride),
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate,
                            use_bias=self.use_bias,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            name="%sconv_shortcut" % name_prefix,
                        )(shortcut)
                    if self.use_batch_norm:
                        shortcut = keras.layers.BatchNormalization(
                            axis=bn_axis,
                            name="%sbn_shortcut" % name_prefix)(shortcut)

            x = keras.layers.add([x, shortcut])

        x = add_activation(self.activation_type, **self.activation_kwargs)(x)

        return x
