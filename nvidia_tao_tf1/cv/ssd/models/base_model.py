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

"""IVA SSD model construction wrapper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Conv2D
from nvidia_tao_tf1.cv.ssd.models.dssd_arch import attach_pred_layers, generate_dssd_layers
from nvidia_tao_tf1.cv.ssd.models.ssd_backbones import get_raw_ssd_model


def _shrink_module(fmap_outputs,
                   num_output_channels,
                   data_format,
                   kernel_regularizer):
    '''Reduce fmaps in fmap_outputs to num_output_channels.

    Args:
        fmap_outputs: List of keras tensors.
        num_output_channels: Output channels each output needs, length equal to fmap_outputs
        data_format: data_format
        kernel_regularizer: kernel_regularizer

    Returns:
        feature_maps: length n list of keras tensors, each of which is output tensor of SSD module
    '''
    feature_maps = []

    for idx, fmap in enumerate(fmap_outputs):
        if num_output_channels[idx] != 0:
            fmap = Conv2D(num_output_channels[idx],
                          kernel_size=1,
                          strides=1,
                          dilation_rate=1,
                          padding='same',
                          data_format=data_format,
                          activation=None,
                          kernel_regularizer=kernel_regularizer,
                          use_bias=False,
                          name='ssd_shrink_block_%d' % (idx))(fmap)
        feature_maps.append(fmap)
    return feature_maps


def get_base_model(input_tensor,
                   arch,
                   nlayers,
                   is_dssd,
                   pred_num_channels,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   freeze_blocks=None,
                   freeze_bn=None):
    '''Wrapper function to get base model.

    Args:
        input_tensor: Image tensor.
        arch, nlayers: feature extractor config.
        is_dssd: whether dssd arch is needed
        pred_num_channels: #channels for pred module
        kernel_regularizer: kernel_regularizer
        bias_regularizer: bias_regularizer
        freeze_blocks: set blocks to non-trainable in extractors (kept pretrained weights forever)
        freeze_bn: freeze bn in pretrained feature extractors.

    Returns:
        pred_layers: list of pred tensors.
    '''

    ssd_layers = get_raw_ssd_model(input_tensor,
                                   arch,
                                   nlayers,
                                   kernel_regularizer,
                                   bias_regularizer,
                                   freeze_blocks,
                                   freeze_bn)

    if is_dssd:
        shrinkage = [0, 0, 0, 0, 0, 0]
        if arch in ['mobilenet_v1', 'mobilenet_v2', 'squeezenet']:
            shrinkage = [256, 0, 0, 0, 0, 0]
        if arch == 'resnet' and nlayers in [10]:
            shrinkage = [256, 0, 0, 0, 0, 0]
        ssd_layers = _shrink_module(ssd_layers, shrinkage, 'channels_first', kernel_regularizer)
        # attach dssd module
        ssd_layers = generate_dssd_layers(ssd_layers,
                                          data_format='channels_first',
                                          kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer)

    pred_layers = attach_pred_layers(ssd_layers,
                                     pred_num_channels,
                                     data_format='channels_first',
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer)

    return pred_layers
