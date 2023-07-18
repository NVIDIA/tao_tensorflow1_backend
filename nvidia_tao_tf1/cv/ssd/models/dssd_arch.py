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

"""IVA DSSD model constructor based on SSD layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Add, BatchNormalization, Conv2D, Conv2DTranspose, Multiply, \
                         ReLU, ZeroPadding2D


def _deconv_module(tensor_small,
                   tensor_large,
                   tensor_large_shape,
                   module_index=0,
                   data_format='channels_first',
                   kernel_regularizer=None,
                   bias_regularizer=None):
    '''
    Deconv module of DSSD. output is a tensor with same shape as tensor_large.

    Args:
        tensor_small: a keras tensor for small feature map
        tensor_large: a keras tensor for immediate larger feature map in backbone
        tensor_large_shape: [c, h, w] of large tensor
        module_index: int representing the index of the module
        data_format: data format
        kernel_regularizer: keras regularizer for kernel
        bias_regularizer: keras regularizer for bias

    Returns:
        deconv_tensor: tensor representing feature maps used for prediction
    '''

    bn_axis = 1 if data_format == 'channels_first' else 3
    x = Conv2D(int(tensor_large_shape[0]),
               kernel_size=3,
               strides=1,
               padding='same',
               data_format=data_format,
               activation=None,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               use_bias=False,
               name='dssd_large_conv_0_block_'+str(module_index))(tensor_large)
    x = BatchNormalization(axis=bn_axis, name='dssd_large_bn_0_block_'+str(module_index))(x)
    x = ReLU(name='dssd_large_relu_0_block_'+str(module_index))(x)
    x = Conv2D(int(tensor_large_shape[0]),
               kernel_size=3,
               strides=1,
               padding='same',
               data_format=data_format,
               activation=None,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               use_bias=False,
               name='dssd_large_conv_1_block_'+str(module_index))(x)
    x = BatchNormalization(axis=bn_axis, name='dssd_large_bn_1_block_'+str(module_index))(x)
    y = Conv2DTranspose(int(tensor_large_shape[0]), (2, 2), strides=(2, 2),
                        padding='same', output_padding=None,
                        data_format=data_format, dilation_rate=(1, 1),
                        activation=None, use_bias=False,
                        kernel_regularizer=kernel_regularizer,
                        name='dssd_small_deconv_block_'+str(module_index))(tensor_small)

    if data_format == "channels_first":
        h_upsampled = tensor_small._keras_shape[2]*2
        w_upsampled = tensor_small._keras_shape[3]*2
    elif data_format == "channels_last":
        h_upsampled = tensor_small._keras_shape[1]*2
        w_upsampled = tensor_small._keras_shape[2]*2

    # perform a trick to match size of x and y
    if h_upsampled == tensor_large_shape[1]:
        # keep size unchanging
        h_pad = 1
        h_kernel = 3
    elif h_upsampled > tensor_large_shape[1]:
        # make spatial size - 1
        h_pad = 0
        h_kernel = 2
    else:
        # make spatial size + 1
        h_pad = 1
        h_kernel = 2

    if w_upsampled == tensor_large_shape[2]:
        # keep size unchanged
        w_pad = 1
        w_kernel = 3
    elif w_upsampled > tensor_large_shape[2]:
        # make spatial size - 1
        w_pad = 0
        w_kernel = 2
    else:
        # make sptial size + 1
        w_pad = 1
        w_kernel = 2

    y = ZeroPadding2D(padding=(h_pad, w_pad), data_format=data_format,
                      name='dssd_pad_'+str(module_index))(y)

    y = Conv2D(int(tensor_large_shape[0]),
               kernel_size=(h_kernel, w_kernel),
               strides=1,
               padding='valid',
               data_format=data_format,
               activation=None,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               use_bias=False,
               name='dssd_small_conv_block_'+str(module_index))(y)

    y = BatchNormalization(axis=bn_axis, name='dssd_small_bn_block_'+str(module_index))(y)

    # finally... We multiply the small and large fmaps
    x = Multiply(name='dssd_mul_block_'+str(module_index))([x, y])
    x = ReLU(name='dssd_relu_block_'+str(module_index))(x)
    return x


def _pred_module(feature_map,
                 module_index=0,
                 num_channels=0,
                 data_format='channels_first',
                 kernel_regularizer=None,
                 bias_regularizer=None):
    '''
    predict module.

    Args:
        feature_map: keras tensor for feature maps used for prediction.
        module_index: the index of module
        num_channels: the number of output feature map channels, use 0 to skip pred_module
        data_format: data format
        kernel_regularizer: keras regularizer for kernel
        bias_regularizer: keras regularizer for bias

    Returns:
        pred_map: a keras tensor with channel number defined by num_channels (if not zero) and
            map size same as feature_map.
    '''

    if num_channels == 0:
        return feature_map
    assert num_channels in [256, 512, 1024], "num_channels only supports 0, 256, 512, 1024"
    bn_axis = 1 if data_format == 'channels_first' else 3

    x = Conv2D(num_channels // 4,
               kernel_size=1,
               strides=1,
               padding='same',
               data_format=data_format,
               activation=None,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               use_bias=False,
               name='ssd_mpred_conv_0_block_'+str(module_index))(feature_map)
    x = BatchNormalization(axis=bn_axis, name='ssd_mpred_bn_0_block_'+str(module_index))(x)
    x = ReLU(name='ssd_mpred_relu_0_block_'+str(module_index))(x)
    x = Conv2D(num_channels // 4,
               kernel_size=1,
               strides=1,
               padding='same',
               data_format=data_format,
               activation=None,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               use_bias=False,
               name='ssd_mpred_conv_1_block_'+str(module_index))(x)
    x = BatchNormalization(axis=bn_axis, name='ssd_mpred_bn_1_block_'+str(module_index))(x)
    x = ReLU(name='ssd_mpred_relu_1_block_'+str(module_index))(x)
    x = Conv2D(num_channels,
               kernel_size=1,
               strides=1,
               padding='same',
               data_format=data_format,
               activation=None,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               use_bias=False,
               name='ssd_mpred_conv_2_block_'+str(module_index))(x)
    x = BatchNormalization(axis=bn_axis, name='ssd_mpred_bn_2_block_'+str(module_index))(x)
    y = Conv2D(num_channels,
               kernel_size=1,
               strides=1,
               padding='same',
               data_format=data_format,
               activation=None,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               use_bias=False,
               name='ssd_mpred_conv_3_block_'+str(module_index))(feature_map)
    y = BatchNormalization(axis=bn_axis, name='ssd_mpred_bn_3_block_'+str(module_index))(y)
    x = Add(name='ssd_mpred_add_block_'+str(module_index))([x, y])
    x = ReLU(name='ssd_mpred_relu_3_block_'+str(module_index))(x)
    return x


def generate_dssd_layers(ssd_layers,
                         data_format='channels_first',
                         kernel_regularizer=None,
                         bias_regularizer=None):
    '''
    Get DSSD layers from SSD layers.

    Args:
        ssd_layers: SSD layers from SSD feature maps.
        data_format: data format
        kernel_regularizer: keras regularizer for kernel
        bias_regularizer: keras regularizer for bias

    Returns:
        dssd_layers: DSSD layers each of which has same shape with that in ssd_layers.
    '''

    # NCHW or NHWC
    l_vals = list(zip(*[l.shape[1:] for l in ssd_layers]))
    if data_format == 'channels_first':
        l_c, l_h, l_w = l_vals
    else:
        l_h, l_w, l_c = l_vals

    results = [ssd_layers[-1]]

    for idx, i in enumerate(reversed(range(len(ssd_layers)-1))):
        dssd_layer = _deconv_module(results[-1],
                                    ssd_layers[i],
                                    [l_c[i], l_h[i], l_w[i]],  # [c, h, w]
                                    module_index=idx,
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer)
        results.append(dssd_layer)

    # return large fmap first.
    return results[::-1]


def attach_pred_layers(dssd_layers,
                       num_channels,
                       data_format='channels_first',
                       kernel_regularizer=None,
                       bias_regularizer=None):
    '''
    Get pred module attached feature map.

    Args:
        dssd_layers: keras tensor for feature maps right before prediction module
        num_channels: the number of output feature map channels, use 0 to skip pred_module
        data_format: data format
        kernel_regularizer: keras regularizer for kernel
        bias_regularizer: keras regularizer for bias

    Returns:
        pred_map: a keras tensor with channel number defined by num_channels (if not zero) and
            map size same as feature_map.
    '''

    results = []
    for idx, l in enumerate(dssd_layers):
        pred_layer = _pred_module(l, idx, num_channels, data_format, kernel_regularizer,
                                  bias_regularizer)
        results.append(pred_layer)

    return results
