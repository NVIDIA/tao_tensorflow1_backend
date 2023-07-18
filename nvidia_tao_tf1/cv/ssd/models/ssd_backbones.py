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

from keras.layers import BatchNormalization, Conv2D, ReLU
from nvidia_tao_tf1.cv.common.models.backbones import get_backbone


def _expansion_module(input_tensor,
                      expansion_config,
                      data_format,
                      kernel_regularizer,
                      bias_regularizer):
    '''Add expansion layers to input_tensor.

    Args:
        input_tensor: keras tensor to expand on
        expansion_config: list of length n (n is the final feature maps needed). Each element is a
            sublist for convs inside. Each element in this sublist is 5-element tuple (or list) for
            (num_filters, kernel_size, strides, dilation, need_bn). The paper VGG16 SSD300 expansion
            can be generated using `expansion_config = [[(1024, 3, 1, 6, False),
            (1024, 1, 1, 1, False)], [(256, 1, 1, 1, False), (512, 3, 2, 1, False)],
            [(128, 1, 1, 1, False), (256, 3, 2, 1, False)],
            [(128, 1, 1, 1, False), (256, 3, 1, 1, False)],
            [(128, 1, 1, 1, False), (256, 3, 1, 1, False)]]`
        data_format: data_format
        kernel_regularizer: kernel_regularizer
        bias_regularizer: bias_regularizer

    Returns:
        feature_maps: length n list of keras tensors, each of which is tensor right after the last
            conv layers in sublist of expansion_config.
    '''

    x = input_tensor
    feature_maps = []
    bn_axis = 1 if data_format == 'channels_first' else 3

    for b_id, block in enumerate(expansion_config):
        for c_id, conv in enumerate(block):
            n, k, s, d, bn = conv
            x = Conv2D(n,
                       kernel_size=k,
                       strides=s,
                       dilation_rate=d,
                       padding='same',
                       data_format=data_format,
                       activation=None,
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       use_bias=(not bn),
                       name='ssd_expand_block_%d_conv_%d' % (b_id, c_id))(x)
            if bn:
                x = BatchNormalization(axis=bn_axis,
                                       name='ssd_expand_block_%d_bn_%d' % (b_id, c_id))(x)
            x = ReLU(name='ssd_expand_block_%d_relu_%d' % (b_id, c_id))(x)
        feature_maps.append(x)
    return feature_maps


def get_raw_ssd_model(input_tensor,
                      arch,
                      nlayers,
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      freeze_blocks=None,
                      freeze_bn=None):
    '''Return feature maps same as original SSD paper.

    Args:
        input_tensor: image tensor
        arch: feature extractor arch
        nlayers: arch layers
        kernel_regularizer: kernel_regularizer
        bias_regularizer: bias_regularizer
        freeze_blocks: freeze_blocks
        freeze_bn: freeze_bn

    Returns:
        layers: (default 6) keras tensors for feature maps.

    '''

    base_model = get_backbone(input_tensor,
                              arch,
                              data_format='channels_first',
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              freeze_blocks=freeze_blocks,
                              freeze_bn=freeze_bn,
                              nlayers=nlayers,
                              use_batch_norm=True,
                              use_pooling=False,
                              use_bias=False,
                              all_projections=True,
                              dropout=1e-3,
                              force_relu=True)

    large_net_expansion = [[(512, 1, 1, 1, False), (512, 3, 1, 1, True)],
                           [(256, 1, 1, 1, False), (512, 3, 2, 1, True)],
                           [(128, 1, 1, 1, False), (256, 3, 2, 1, True)],
                           [(128, 1, 1, 1, False), (256, 3, 2, 1, True)],
                           [(128, 1, 1, 1, False), (256, 3, 2, 1, True)]]

    mid_net_expansion = [[(256, 1, 1, 1, False), (256, 3, 1, 1, True)],
                         [(128, 1, 1, 1, False), (256, 3, 2, 1, True)],
                         [(64, 1, 1, 1, False), (128, 3, 2, 1, True)],
                         [(64, 1, 1, 1, False), (128, 3, 2, 1, True)],
                         [(64, 1, 1, 1, False), (128, 3, 2, 1, True)]]

    sm_net_expansion = [[],
                        [(64, 1, 1, 1, False), (128, 3, 2, 1, True)],
                        [(64, 1, 1, 1, False), (128, 3, 2, 1, True)],
                        [(64, 1, 1, 1, False), (128, 3, 2, 1, True)],
                        [(64, 1, 1, 1, False), (128, 3, 2, 1, True)]]

    if arch == 'resnet':
        exp_config = large_net_expansion
        if nlayers == 10:
            fmaps = ['block_2a_relu']
            exp_config = sm_net_expansion
        elif nlayers == 18:
            fmaps = ['block_2b_relu']
            exp_config = mid_net_expansion
        elif nlayers == 34:
            fmaps = ['block_2d_relu']
            exp_config = mid_net_expansion
        elif nlayers == 50:
            fmaps = ['block_2d_relu']
        elif nlayers == 101:
            fmaps = ['block_2d_relu']
        else:
            raise ValueError("ResNet-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '10, 18, 34, 50, 101'))
    elif arch == 'vgg':
        exp_config = large_net_expansion
        if nlayers == 16:
            fmaps = ['block_4c_relu']
        elif nlayers == 19:
            fmaps = ['block_4d_relu']
        else:
            raise ValueError("ResNet-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '16, 19'))
    elif arch == 'darknet':
        if nlayers == 19:
            exp_config = mid_net_expansion
            fmaps = ['b3_conv3_lrelu', 'b4_conv5_lrelu']
        elif nlayers == 53:
            exp_config = large_net_expansion
            fmaps = ['b3_add8', 'b4_add8']
        else:
            raise ValueError("DarkNet-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '19, 53'))
        exp_config = exp_config[:-1]
    elif arch == 'mobilenet_v1':
        fmaps = ['conv_pw_relu_5']
        exp_config = sm_net_expansion
    elif arch == 'mobilenet_v2':
        fmaps = ['re_lu_7']
        exp_config = sm_net_expansion
    elif arch == 'squeezenet':
        fmaps = ['fire8']
        exp_config = sm_net_expansion
    elif arch == 'googlenet':
        fmaps = ['inception_3b_output']
        exp_config = large_net_expansion
    elif arch == 'efficientnet_b0':
        fmaps = ['block4a_expand_activation', 'block6a_expand_activation']
        exp_config = large_net_expansion
        exp_config = exp_config[:-1]
    else:
        raise ValueError("{} architecture is currently not implemented\n".
                         format(arch))

    exp_fmaps = _expansion_module(base_model.layers[-1].output,
                                  exp_config,
                                  data_format='channels_first',
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer)

    return [base_model.get_layer(l).output for l in fmaps] + exp_fmaps
