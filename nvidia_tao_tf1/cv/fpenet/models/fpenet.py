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
"""FpeNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.models import Model

from nvidia_tao_tf1.core.models.templates.utils import CNNBlock
from nvidia_tao_tf1.core.models.templates.utils import get_batchnorm_axis
from nvidia_tao_tf1.cv.fpenet.models.custom.softargmax import Softargmax
from nvidia_tao_tf1.cv.fpenet.models.rcnet import RecombinatorNet


class FpeNetModel(object):
    """FpeNet model definition."""

    def __init__(self,
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
                 block_post_decoder=None,
                 nkeypoints=None,
                 beta=0.1,
                 additional_conv_layer=True,
                 use_upsampling_layer=True):
        """__init__ method.

        Construct a Fiducial Points Estimator network with a Softargmax activation function.
        Based on Recombinator Networks.
        Described in the paper (in particular in its appendix) [1].

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
            blocks_encoder (list of convolution blocks): A convolution block is a list of (K, C)
                tuples of arbitrary length (aka convolution subblocks), where C is the number of
                output channels and K the kernel size of a convolution layer. The encoder reduces
                the spatial resolution horizontally and vertically by a factor of two (stride of 2)
                per convolution block, respectively.
            block_trunk (one convolution block): This block preserves the spatial resolution of the
                final encoder output and 'refines' it.
            blocks_decoder (list of convolution blocks): The decoder increases the spatial
                resolution horizontally and vertically by a factor of two (upsample factor of 2) per
                convolution block, respectively. blocks_encoder and blocks_decoder must have the
                same length of convolution blocks, while the number of convolution layers per block
                may vary.
            block_post_decoder (one convolution block): This optional block preserves the spatial
                resolution of the final decoder output and 'refines' it before predicting the
                key-points feature maps.
            nkeypoints (int): Number of key points to be predicted. A 1x1 convolution layer is
                appended to the final decoder output with nkeypoints output channels, and a
                corresponding Softargmax operator is added.
            beta (float): Softargmax coefficient used for multiplying the key-point maps after
                subtracting the channel-wise maximum.
            additional_conv_layer (bool): additional convolutional layer in the end.
            use_upsampling_layer (bool): upsamping layer or decconv layer

        Returns:
            Model: The output model after applying the Fiducial Point Estimator net on top of
            input `x`.

        [1] Improving Landmark Localization with Semi-Supervised Learning
            (https://arxiv.org/abs/1709.01591)

        """
        # Check whether data format is supported.
        if data_format not in ["channels_first", "channels_last"]:
            raise ValueError("Unsupported data_format: {}.".format(data_format))

        self._pooling = pooling
        self._use_batch_norm = use_batch_norm
        self._data_format = data_format
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activation_type = activation_type
        self._activation_kwargs = activation_kwargs or {}
        self._blocks_encoder = blocks_encoder or [[(3, 64)]] * 4
        self._block_trunk = block_trunk or [(3, 64), (3, 64), (1, 64)]
        self._blocks_decoder = blocks_decoder or [[(3, 64), (1, 64)]] * 4
        self._block_post_decoder = block_post_decoder or list()
        self._nkeypoints = nkeypoints
        self._beta = beta
        self._additional_conv_layer = additional_conv_layer
        self._use_upsampling_layer = use_upsampling_layer

    def conv2D_bn_activation(
                            self,
                            x,
                            use_batch_norm,
                            filters,
                            kernel_size,
                            strides=(1, 1),
                            activation_type="relu",
                            activation_kwargs=None,
                            data_format=None,
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            layer_name=None,
                            use_bias=True,
                            trainable=True
                            ):
        """
        Add a conv layer, followed by batch normalization and activation.

        Args:
            x (tensor): the inputs (tensor) to the convolution layer.
            use_batch_norm (bool): use batch norm.
            filters (int): the number of filters.
            kernel_size: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            activation_type (str): activation function name, e.g., 'relu'.
            activation_kwargs (dict): Additional activation keyword arguments to be fed to
                the add_activation function.
            data_format (str): either 'channels_last' or 'channels_first'.
            kernel_regularizer (`regularizer`): regularizer for the kernels.
            bias_regularizer (`regularizer`): regularizer for the biases.
            layer_name(str): layer name prefix.
            use_bias(bool): whether or not use bias in convolutional layer.

        Returns:
            x (tensor): the output tensor of the convolution layer.
        """
        if layer_name is not None:
            layer_name = "%s_m%d" % (layer_name, filters)

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding="same",
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   name=layer_name,
                   use_bias=use_bias,
                   trainable=trainable)(x)
        if use_batch_norm:
            if layer_name is not None:
                layer_name += "_bn"
            x = BatchNormalization(axis=get_batchnorm_axis(data_format),
                                   name=layer_name,
                                   trainable=trainable)(x)
        if activation_type:
            # activation_kwargs = activation_kwargs or {}
            x = Activation(activation_type)(x)
            # x = add_activation(activation_type, **activation_kwargs)(x)
        return x

    def construct(self, imageface):
        """Create a template for a Fiducial Points Estimator network.

        Args:
            image_face (Tensor): Input tensor for face.
        """
        # First construct the RCN back-bone:
        rcn_model = RecombinatorNet(inputs=imageface,
                                    pooling=self._pooling,
                                    use_batch_norm=self._use_batch_norm,
                                    data_format=self._data_format,
                                    kernel_regularizer=self._kernel_regularizer,
                                    bias_regularizer=self._bias_regularizer,
                                    activation_type=self._activation_type,
                                    activation_kwargs=self._activation_kwargs,
                                    blocks_encoder=self._blocks_encoder,
                                    block_trunk=self._block_trunk,
                                    blocks_decoder=self._blocks_decoder,
                                    use_upsampling_layer=self._use_upsampling_layer)

        # Grab the output tensor of our RCN back-bone:
        rcn_output = rcn_model.outputs[0]

        # If specified, add a convolution block on top of the RCN back-bone ...
        if self._block_post_decoder:
            block = [(3, 128), (1, 64)]
            rcn_output = CNNBlock(
                use_batch_norm=self._use_batch_norm,
                use_shortcuts=False,
                repeat=1,
                stride=1,
                subblocks=block,
                activation_type=self._activation_type,
                use_bias=not (self._use_batch_norm),
                data_format=self._data_format,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activation_kwargs=self._activation_kwargs)(rcn_output)

        # additional conv layer as head of RCN
        if self._additional_conv_layer is True:
            rcn_output = self.conv2D_bn_activation(
                rcn_output,
                use_batch_norm=False,
                filters=64,
                kernel_size=1,
                strides=(1, 1),
                activation_type=self._activation_type,
                layer_name='conv_keypoints',
                use_bias=True,
                data_format=self._data_format,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activation_kwargs=self._activation_kwargs)

        # add final 1x1 convolution for predicting the target number of key points
        conv_keypoints = self.conv2D_bn_activation(
            rcn_output,
            use_batch_norm=False,
            filters=self._nkeypoints,
            kernel_size=1,
            strides=(1, 1),
            activation_type=None,
            layer_name='conv_keypoints',
            use_bias=True,
            data_format=self._data_format,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activation_kwargs=self._activation_kwargs)

        # Grab the output shape of the tensor produced by the previous conv operator
        # (important to add `as_list()` so that a snapshot of the shape is taken during build time
        # and not derived from the graph during run time).
        conv_keypoints_shape = conv_keypoints.get_shape().as_list()

        # Add a Softargmax activation:
        softargmax, confidence = Softargmax(
            conv_keypoints_shape,
            beta=self._beta,
            data_format=self._data_format,
            name='softargmax')(conv_keypoints)

        # Derive a model name from the number of key points:
        keypoints_model_name = 'fpe_%s_%dkpts' % (rcn_model.name,
                                                  self._nkeypoints)

        outputs_list = [softargmax, confidence, conv_keypoints]

        rcn_keypoints_model = Model(inputs=imageface,
                                    outputs=outputs_list,
                                    name=keypoints_model_name)

        return rcn_keypoints_model
