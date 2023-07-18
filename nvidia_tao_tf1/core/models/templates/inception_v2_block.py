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

"""Maglev utilities for model templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from nvidia_tao_tf1.core.decorators.arg_scope import add_arg_scope, arg_scope
from nvidia_tao_tf1.core.models.templates.utils import conv2D_bn_activation

if os.environ.get("TF_KERAS"):
    from tensorflow import keras
else:
    import keras

bn_axis_map = {"channels_last": 3, "channels_first": 1}

SUBBLOCK_IDS = [
    "br0_1x1",
    ["br1_1x1", "br1_3x3"],
    ["br2_1x1", "br2_3x3a", "br2_3x3b"],
    ["br3_1x1", "br3_pool"],
]


class InceptionV2Block(object):
    """A functor for creating a Inception v2 module."""

    @add_arg_scope
    def __init__(
        self,
        use_batch_norm=False,
        data_format=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        use_bias=None,
        subblocks=None,
        block_name_prefix=None,
        icp_block_index=None,
        activation_type="relu",
    ):
        """
        Initialization of the block functor object.

        Args:
            use_batch_norm (bool): whether batchnorm should be added after each convolution.
            data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
            kernel_regularizer (float): regularizer to apply to kernels.
            bias_regularizer (float): regularizer to apply to biases.
            subblocks (list): A list of list with size 4, defining number of feature-maps for
                subbblocks in an inception v2 block.
            This is a slightly modified version of Inception v2 module:
            "Rethinking the Inception Architecture for Computer Vision"
             by Szegedy, Christian, et. al.

            Inception_v2: [[32], [32, 32], [32, 32, 32], 32]

            Define Inception block with following parallel branches
            1) 32 outputs from 1x1 convolutions
            2.1) 32 outputs from 1x1 convolutions --> 2.2) 32 outputs from 3x3 convolutions
            3.1) 32 outputs from 1x1 convolutions --> 3.2) 32 outputs from 3x3 convolutions
            --> 3.3) 32 outputs from 3x3 convolutions
            4.1) 32 outputs from 1x1 convolutions --> 4.2) Average pooling with 3x3 pooling
            The fourth branch is slightly different from the original model.
            The original model is 3x3 max pooling followed by 1x1 convolution; whereas
            this implementation performs 1x1 convolution followed by 3x3 average pooling.
            This change speeds up the inference run-time performance.

            The outputs of 1, 2.2, 3.3, and 4.2 are concatenated to produce final output.

            block_name_prefix (str): name prefix for the whole block.
            icp_block_index (int): the index of the block to be created.
            activation_type (str): activation function type.
        """
        self.use_batch_norm = use_batch_norm
        self.data_format = data_format
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        if use_bias is None:
            self.use_bias = not (use_batch_norm)
        else:
            self.use_bias = use_bias
        self.activation_type = activation_type
        self.subblocks = subblocks
        self.icp_block_index = 0 if icp_block_index is None else icp_block_index
        self.block_name_prefix = "" if block_name_prefix is None else block_name_prefix
        self.name = "%sicp%d" % (self.block_name_prefix, self.icp_block_index)

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
        if nblocks != 4:
            print("Inception V2 block must have 4 subblocks/paralle_branches")
            return x

        with arg_scope(
            [conv2D_bn_activation],
            use_batch_norm=self.use_batch_norm,
            data_format=self.data_format,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
        ):

            # The first branch is 1x1 conv with padding = 0, stride = 1.
            x1 = conv2D_bn_activation(
                x,
                filters=self.subblocks[0],
                kernel_size=(1, 1),
                strides=(1, 1),
                layer_name="%s_%s" % (name_prefix, SUBBLOCK_IDS[0]),
            )

            # The second branch is 1x1 conv with padding = 0, stride = 1.
            x2 = conv2D_bn_activation(
                x,
                filters=self.subblocks[1][0],
                kernel_size=(1, 1),
                strides=(1, 1),
                layer_name="%s_%s" % (name_prefix, SUBBLOCK_IDS[1][0]),
            )

            # This second branch is 1x1 conv with padding = 0, stride = 1 followed by 3x3 conv.
            x2 = conv2D_bn_activation(
                x2,
                filters=self.subblocks[1][1],
                kernel_size=(3, 3),
                strides=(1, 1),
                layer_name="%s_%s" % (name_prefix, SUBBLOCK_IDS[1][1]),
            )

            # The third branch is 1x1 conv with stride = 1.
            x3 = conv2D_bn_activation(
                x,
                filters=self.subblocks[2][0],
                kernel_size=(1, 1),
                strides=(1, 1),
                layer_name="%s_%s" % (name_prefix, SUBBLOCK_IDS[2][0]),
            )

            # The third branch is 1x1 conv with padding = 0, stride = 1 followed by 3x3 conv.
            x3 = conv2D_bn_activation(
                x3,
                filters=self.subblocks[2][1],
                kernel_size=(3, 3),
                strides=(1, 1),
                layer_name="%s_%s" % (name_prefix, SUBBLOCK_IDS[2][1]),
            )

            x3 = conv2D_bn_activation(
                x3,
                filters=self.subblocks[2][2],
                kernel_size=(3, 3),
                strides=(1, 1),
                layer_name="%s_%s" % (name_prefix, SUBBLOCK_IDS[2][2]),
            )

            # The fourth branch is a 1x1 conv followed by a 3x3 average pool stride = 1.
            # This is different from the original paper. The 1x1 can be performed in parallel
            # at inference time for all 4 branches.
            x4 = conv2D_bn_activation(
                x,
                filters=self.subblocks[3],
                kernel_size=(1, 1),
                strides=(1, 1),
                layer_name="%s_%s" % (name_prefix, SUBBLOCK_IDS[3][0]),
            )
            x4 = keras.layers.AveragePooling2D(
                pool_size=(3, 3),
                strides=(1, 1),
                padding="same",
                name="%s_%s" % (name_prefix, SUBBLOCK_IDS[3][1]),
            )(x4)

        # Concat layer.
        if self.data_format == "channels_first":
            x = keras.layers.Concatenate(
                axis=1, name="%s_concatenated" % (name_prefix)
            )([x1, x2, x3, x4])
        else:
            x = keras.layers.Concatenate(
                axis=-1, name="%s_concatenated" % (name_prefix)
            )([x1, x2, x3, x4])
        return x
