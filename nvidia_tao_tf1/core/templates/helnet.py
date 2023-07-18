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
"""Modulus model templates for HelNets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from nvidia_tao_tf1.core.decorators.arg_scope import arg_scope
from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.utils import add_activation
from nvidia_tao_tf1.core.models.templates.utils import CNNBlock
from nvidia_tao_tf1.core.models.templates.utils import get_batchnorm_axis
from nvidia_tao_tf1.core.models.templates.utils import performance_test_model

if os.environ.get("TF_KERAS"):
    from tensorflow import keras
else:
    import keras

logger = logging.getLogger(__name__)


def HelNet(
    nlayers,
    inputs,
    pooling=False,
    use_batch_norm=False,
    use_bias=None,
    data_format=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    activation_type="relu",
    activation_kwargs=None,
    first_filter_kernel_size=7,
    dilation_rate=(1, 1),
    block_repeats=None,
    block_widths=(64, 128, 256, 512),
    block_strides=(2, 2, 2, 1),
    quantize=False,
    bitwidth=8,
    weights=None,
):
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
        first_filter_kernel_size (int): kernel size of the first filter in network.
        dilation_rate (int or (int, int)): An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
        block_repeats (tuple of ints): number of times to repeat each convolutional block.
        block_widths (tuple of ints): width i.e. number of features maps in each convolutional block
            in the model.
        quantize (bool): Flag for using QuantizedConv2D and ReLU6.
        bitwidth (int): number of quantization bits.
        strides (tuple of ints): the convolution stride for the first conv of each block
    Returns:
        Model: the output model after applying the HelNet on top of input `x`.

    [1] Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
    """
    if quantize:
        if activation_kwargs is None:
            activation_kwargs = {"max_value": 6.0}
        else:
            activation_kwargs.update({"max_value": 6.0})
        activation_type = "relu-n"

    if use_bias is None:
        use_bias = not (use_batch_norm)

    if data_format is None:
        data_format = keras.backend.image_data_format()
    activation_kwargs = activation_kwargs or {}

    if block_repeats is None:
        if nlayers == 6:
            block_repeats = (1, 1, 1, 1)
        elif nlayers == 10:
            block_repeats = (1, 1, 1, 1)
        elif nlayers == 12:
            block_repeats = (1, 1, 2, 1)
        elif nlayers == 18:
            block_repeats = (2, 2, 2, 2)
        elif nlayers == 26:
            block_repeats = (3, 4, 3, 2)
        elif nlayers == 34:
            block_repeats = (3, 4, 6, 3)
        else:
            raise NotImplementedError(
                "A Helnet with nlayers=%d is not implemented." % nlayers
            )

    # Create HelNet-0 model for training diagnostics.
    if nlayers == 0:
        return performance_test_model(inputs, data_format, activation_type)

    if quantize:
        x = QuantizedConv2D(
            64,
            (first_filter_kernel_size, first_filter_kernel_size),
            strides=(2, 2),
            padding="same",
            data_format=data_format,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bitwidth=bitwidth,
            name="conv1",
        )(inputs)
    else:
        x = keras.layers.Conv2D(
            64,
            (first_filter_kernel_size, first_filter_kernel_size),
            strides=(2, 2),
            padding="same",
            data_format=data_format,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="conv1",
        )(inputs)
    if use_batch_norm:
        x = keras.layers.BatchNormalization(
            axis=get_batchnorm_axis(data_format), name="bn_conv1"
        )(x)
    x = add_activation(activation_type, **activation_kwargs)(x)
    if pooling:
        if quantize:
            if use_batch_norm:
                qdq_name = "conv1_bn_act_qdq"
            else:
                qdq_name = "conv1_act_qdq"
            x = QDQ(name=qdq_name)(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), padding="same", data_format=data_format
        )(x)
        first_sride = 1
    else:
        first_sride = block_strides[0]

    # Define a block functor which can create blocks
    with arg_scope(
        [CNNBlock],
        use_batch_norm=use_batch_norm,
        use_shortcuts=False,
        data_format=data_format,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activation_type=activation_type,
        activation_kwargs=activation_kwargs,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        quantize=quantize,
        bitwidth=bitwidth,
    ):
        if nlayers == 6:
            x = CNNBlock(
                repeat=block_repeats[0],
                stride=first_sride,
                subblocks=[(3, block_widths[0])],
                index=1,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[1],
                stride=block_strides[1],
                subblocks=[(3, block_widths[1])],
                index=2,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[2],
                stride=block_strides[2],
                subblocks=[(3, block_widths[2])],
                index=3,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[3],
                stride=block_strides[3],
                subblocks=[(3, block_widths[3])],
                index=4,
            )(x)
        elif nlayers == 10:
            x = CNNBlock(
                repeat=block_repeats[0],
                stride=first_sride,
                subblocks=[(3, block_widths[0])] * 2,
                index=1,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[1],
                stride=block_strides[1],
                subblocks=[(3, block_widths[1])] * 2,
                index=2,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[2],
                stride=block_strides[2],
                subblocks=[(3, block_widths[2])] * 2,
                index=3,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[3],
                stride=block_strides[3],
                subblocks=[(3, block_widths[3])] * 2,
                index=4,
            )(x)
        elif nlayers == 12:
            x = CNNBlock(
                repeat=block_repeats[0],
                stride=first_sride,
                subblocks=[(3, block_widths[0])] * 2,
                index=1,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[1],
                stride=block_strides[1],
                subblocks=[(3, block_widths[1])] * 2,
                index=2,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[2],
                stride=block_strides[2],
                subblocks=[(3, block_widths[2])] * 2,
                index=3,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[3],
                stride=block_strides[3],
                subblocks=[(3, block_widths[3])] * 2,
                index=4,
            )(x)
        elif nlayers == 18:
            x = CNNBlock(
                repeat=block_repeats[0],
                stride=first_sride,
                subblocks=[(3, block_widths[0])] * 2,
                index=1,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[1],
                stride=block_strides[1],
                subblocks=[(3, block_widths[1])] * 2,
                index=2,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[2],
                stride=block_strides[2],
                subblocks=[(3, block_widths[2])] * 2,
                index=3,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[3],
                stride=block_strides[3],
                subblocks=[(3, block_widths[3])] * 2,
                index=4,
            )(x)
        elif nlayers == 26:
            x = CNNBlock(
                repeat=block_repeats[0],
                stride=first_sride,
                subblocks=[(3, block_widths[0])] * 2,
                index=1,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[1],
                stride=block_strides[1],
                subblocks=[(3, block_widths[1])] * 2,
                index=2,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[2],
                stride=block_strides[2],
                subblocks=[(3, block_widths[2])] * 2,
                index=3,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[3],
                stride=block_strides[3],
                subblocks=[(3, block_widths[3])] * 2,
                index=4,
            )(x)
        elif nlayers == 34:
            x = CNNBlock(
                repeat=block_repeats[0],
                stride=first_sride,
                subblocks=[(3, block_widths[0])] * 2,
                index=1,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[1],
                stride=block_strides[1],
                subblocks=[(3, block_widths[1])] * 2,
                index=2,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[2],
                stride=block_strides[2],
                subblocks=[(3, block_widths[2])] * 2,
                index=3,
            )(x)
            x = CNNBlock(
                repeat=block_repeats[3],
                stride=block_strides[3],
                subblocks=[(3, block_widths[3])] * 2,
                index=4,
            )(x)
        else:
            raise NotImplementedError(
                "A Helnet with nlayers=%d is not implemented." % nlayers
            )

    model_name = "helnet%d_s16" % nlayers
    if pooling:
        model_name += "_nopool"
    if use_batch_norm:
        model_name += "_bn"

    model = keras.models.Model(inputs=inputs, outputs=x, name=model_name)

    if weights == "imagenet":
        logger.warning("Imagenet weights can not be used for production models.")
        if nlayers == 18:
            if use_batch_norm:
                weights_path = keras.utils.data_utils.get_file(
                    "imagenet_helnet18-bn_weights_20170729.h5",
                    "https://s3-us-west-2.amazonaws.com/"
                    "9j2raan2rcev-ai-infra-models/"
                    "imagenet_helnet18-bn_weights_20170729.h5",
                    cache_subdir="models",
                    md5_hash="6a2d59e48d8b9f0b41a2b02a2f3c018e",
                )
            else:
                weights_path = keras.utils.data_utils.get_file(
                    "imagenet_helnet18-no-bn_weights_20170729.h5",
                    "https://s3-us-west-2.amazonaws.com/"
                    "9j2raan2rcev-ai-infra-models/"
                    "imagenet_helnet18-no-bn_weights_20170729.h5",
                    cache_subdir="models",
                    md5_hash="3282b1e5e7f8e769a034103c455968e6",
                )
            model.load_weights(weights_path, by_name=True)

    return model
