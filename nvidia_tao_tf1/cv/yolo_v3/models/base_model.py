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

"""IVA YOLO model construction wrapper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import UpSampling2D

from nvidia_tao_tf1.core.templates.utils import _leaky_conv
from nvidia_tao_tf1.cv.common.models.backbones import get_backbone


def get_base_model(input_tensor,
                   arch,
                   nlayers,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   freeze_blocks=None,
                   freeze_bn=None,
                   force_relu=False):
    '''Return feature maps for YOLOv3.

    Args:
        input_tensor: image tensor
        arch: feature extractor arch
        nlayers: arch layers
        kernel_regularizer: kernel_regularizer
        bias_regularizer: bias_regularizer
        freeze_blocks: freeze_blocks
        freeze_bn: freeze_bn
        force_relu: Replace LeakyReLU with ReLU.

    Returns:
        the return is two tuples. First one is three tensors for three feature layers. second one is
            two integer tuple, corresponding to upsample0 and upsample1 num_filters.
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
                              force_relu=force_relu)

    def additional_conv(nchannels):
        return _leaky_conv(base_model.layers[-1].output,
                           nchannels, alpha=0.1, kernel=3, strides=2,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           use_batch_norm=True, force_relu=force_relu,
                           name='yolo_expand_conv1')

    if arch == 'resnet':
        if nlayers == 10:
            fmaps = (additional_conv(512),
                     base_model.layers[-10].output, base_model.layers[-19].output)
            map_size = (128, 64)
        elif nlayers == 18:
            fmaps = (additional_conv(512),
                     base_model.layers[-19].output, base_model.layers[-37].output)
            map_size = (128, 64)
        elif nlayers == 34:
            fmaps = (additional_conv(512),
                     base_model.layers[-28].output, base_model.layers[-82].output)
            map_size = (128, 64)
        elif nlayers == 50:
            # Extract layers[-43] as the feature layer for large feature map (to detect sm object)
            lg_map = base_model.layers[-43].output
            lg_map = UpSampling2D(2, data_format='channels_first', name='expand_upsample')(lg_map)
            fmaps = (additional_conv(1024), base_model.layers[-7].output, lg_map)
            map_size = (256, 128)
        elif nlayers == 101:
            # Extract layers[-43] as the feature layer for large feature map (to detect sm object)
            # there's too many stride 16 layers. We take one and upsample it to stride 8.
            lg_map = base_model.layers[-43].output
            lg_map = UpSampling2D(2, data_format='channels_first', name='expand_upsample')(lg_map)
            fmaps = (additional_conv(1024), base_model.layers[-7].output, lg_map)
            map_size = (256, 128)
        else:
            raise ValueError("ResNet-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '10, 18, 34, 50, 101'))
    elif arch == 'vgg':
        if nlayers == 16:
            fmaps = (base_model.layers[-1].output, base_model.layers[-10].output,
                     base_model.layers[-19].output)
            map_size = (256, 128)

        elif nlayers == 19:
            fmaps = (base_model.layers[-1].output, base_model.layers[-13].output,
                     base_model.layers[-25].output)
            map_size = (256, 128)
        else:
            raise ValueError("ResNet-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '16, 19'))
    elif arch == 'efficientnet_b0':
        lg_map = base_model.get_layer('block5a_expand_activation').output
        lg_map = UpSampling2D(2, data_format='channels_first', name='expand_upsample')(lg_map)
        z = _leaky_conv(lg_map, 256, alpha=0.1, kernel=3,
                        strides=1, kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer, use_batch_norm=True,
                        force_relu=force_relu, name='expand_conv3')
        fmaps = (base_model.layers[-1].output,
                 base_model.get_layer('block6a_expand_activation').output,
                 z)
        map_size = (256, 128)

    elif arch == 'darknet':
        if nlayers == 19:
            fmaps = (base_model.layers[-1].output, base_model.layers[-17].output,
                     base_model.layers[-33].output)
            map_size = (256, 128)
        elif nlayers == 53:
            fmaps = (base_model.layers[-1].output, base_model.layers[-32].output,
                     base_model.layers[-91].output)
            map_size = (256, 128)
        else:
            raise ValueError("DarkNet-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '19, 53'))
    elif arch == 'mobilenet_v1':
        fmaps = (additional_conv(512), base_model.layers[-39].output, base_model.layers[-53].output)
        map_size = (128, 64)
    elif arch == 'mobilenet_v2':
        fmaps = (additional_conv(96), base_model.layers[-32].output, base_model.layers[-74].output)
        map_size = (32, 16)
    elif arch == 'squeezenet':
        # Quite a bit work here...
        x = additional_conv(128)

        y = _leaky_conv(base_model.layers[-1].output, 128, alpha=0.1, kernel=1,
                        strides=1, kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer, use_batch_norm=True,
                        force_relu=force_relu, name='expand_conv2')

        z = _leaky_conv(base_model.layers[-9].output, 128, alpha=0.1, kernel=1,
                        strides=1, kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer, use_batch_norm=True,
                        force_relu=force_relu, name='expand_conv3')
        fmaps = (x, y, z)
        map_size = (64, 64)
    elif arch == 'googlenet':
        # Quite a bit work here...
        x = additional_conv(1024)

        y = _leaky_conv(base_model.layers[-21].output, 512, alpha=0.1, kernel=3,
                        strides=1, kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer, use_batch_norm=True,
                        force_relu=force_relu, name='expand_conv2')

        lg_map = base_model.layers[-41].output
        lg_map = UpSampling2D(2, data_format='channels_first', name='expand_upsample')(lg_map)
        z = _leaky_conv(lg_map, 256, alpha=0.1, kernel=3,
                        strides=1, kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer, use_batch_norm=True,
                        force_relu=force_relu, name='expand_conv3')
        fmaps = (x, y, z)
        map_size = (256, 128)

    else:
        raise ValueError("{} architecture is currently not implemented\n".
                         format(arch))

    return fmaps, map_size
