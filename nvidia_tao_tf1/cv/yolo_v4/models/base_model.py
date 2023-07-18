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

"""IVA YOLOv4 model construction wrapper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers

from nvidia_tao_tf1.core.templates.utils import _leaky_conv
from nvidia_tao_tf1.cv.common.models.backbones import get_backbone


def tiny_neck_featuremaps(
    model,
):
    """helper function to build the neck featuremap of yolov4-tiny."""
    feat_names = ["conv_5_mish", "conv_4_conv_3_mish"]
    feats = [model.get_layer(n).output for n in feat_names]
    feat_sizes = [128]
    return feats, feat_sizes


def tiny3l_neck_featuremaps(
    model,
):
    """helper function to build the neck featuremap of yolov4-tiny-3l."""
    feat_names = ["conv_5_mish", "conv_4_conv_3_mish", "conv_3_conv_3_mish"]
    feats = [model.get_layer(n).output for n in feat_names]
    feat_sizes = [128, 64]
    return feats, feat_sizes


def get_base_model(input_tensor,
                   arch,
                   nlayers,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   freeze_blocks=None,
                   freeze_bn=None,
                   force_relu=False,
                   activation="leaky_relu"):
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
        activation(str): Activation type.

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
                              force_relu=force_relu,
                              activation=activation)
    if arch == "cspdarknet_tiny":
        return tiny_neck_featuremaps(
            base_model,
        )
    if arch == "cspdarknet_tiny_3l":
        return tiny3l_neck_featuremaps(
            base_model,
        )
    base_featuremap = base_model.layers[-1].output
    neck = base_featuremap
    neck = _leaky_conv(
        neck, 512, kernel=1, strides=1,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_batch_norm=True,
        force_relu=force_relu,
        name="yolo_neck_1"
    )
    neck = _leaky_conv(
        neck, 1024, kernel=3, strides=1,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_batch_norm=True,
        force_relu=force_relu,
        name="yolo_neck_2"
    )
    neck = _leaky_conv(
        neck, 512, kernel=1, strides=1,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_batch_norm=True,
        force_relu=force_relu,
        name="yolo_neck_3"
    )
    # SPP module

    x1 = layers.MaxPooling2D(pool_size=5, strides=1, data_format='channels_first',
                             padding='same', name='yolo_spp_pool_1')(neck)
    x2 = layers.MaxPooling2D(pool_size=9, strides=1, data_format='channels_first',
                             padding='same', name='yolo_spp_pool_2')(neck)
    x3 = layers.MaxPooling2D(pool_size=13, strides=1, data_format='channels_first',
                             padding='same', name='yolo_spp_pool_3')(neck)

    x = layers.Concatenate(axis=1,
                           name='yolo_spp_concat')([x1, x2, x3, neck])

    # Detector will use leaky-relu activation
    # See darknet yolov4.cfg
    x_spp = _leaky_conv(
        x, 512, kernel=1, strides=1,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_batch_norm=True,
        force_relu=force_relu,
        name='yolo_spp_conv'
    )

    def additional_conv(nchannels):
        return _leaky_conv(
            x_spp, nchannels, kernel=3, strides=2,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            use_batch_norm=True,
            force_relu=force_relu,
            name='yolo_expand_conv1'
        )

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
            lg_map = layers.UpSampling2D(2, data_format='channels_first',
                                         name='expand_upsample')(lg_map)
            fmaps = (additional_conv(1024), base_model.layers[-7].output, lg_map)
            map_size = (256, 128)
        elif nlayers == 101:
            # Extract layers[-43] as the feature layer for large feature map (to detect sm object)
            # there's too many stride 16 layers. We take one and upsample it to stride 8.
            lg_map = base_model.layers[-43].output
            lg_map = layers.UpSampling2D(2, data_format='channels_first',
                                         name='expand_upsample')(lg_map)
            fmaps = (additional_conv(1024), base_model.layers[-7].output, lg_map)
            map_size = (256, 128)
        else:
            raise ValueError("ResNet-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '10, 18, 34, 50, 101'))
    elif arch == 'vgg':
        if nlayers == 16:
            fmaps = (x_spp, base_model.layers[-10].output,
                     base_model.layers[-19].output)
            map_size = (256, 128)

        elif nlayers == 19:
            fmaps = (x_spp, base_model.layers[-13].output,
                     base_model.layers[-25].output)
            map_size = (256, 128)
        else:
            raise ValueError("VGG-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '16, 19'))
    elif arch == 'darknet':
        if nlayers == 19:
            fmaps = (x_spp, base_model.get_layer('b4_conv5').output,
                     base_model.get_layer('b3_conv3').output)
            map_size = (256, 128)
        elif nlayers == 53:
            fmaps = (x_spp, base_model.get_layer('b4_add7').output,
                     base_model.get_layer('b3_add7').output)
            map_size = (256, 128)
        else:
            raise ValueError("DarkNet-{} architecture is currently not implemented\n"
                             "Please choose out of the following:\n{}.".
                             format(nlayers, '19, 53'))
    elif arch == 'cspdarknet':
        fmaps = (x_spp, base_model.get_layer('b4_final_trans').output,
                 base_model.get_layer('b3_final_trans').output)
        map_size = (256, 128)
    elif arch == 'efficientnet_b0':
        lg_map = base_model.get_layer('block5a_expand_activation').output
        lg_map = layers.UpSampling2D(2, data_format='channels_first',
                                     name='expand_upsample')(lg_map)
        z = _leaky_conv(
            lg_map, 256, kernel=3,
            strides=1, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, use_batch_norm=True,
            force_relu=force_relu, name='expand_conv3'
        )
        fmaps = (base_featuremap,
                 base_model.get_layer('block6a_expand_activation').output,
                 z)
        map_size = (256, 128)
    elif arch == 'mobilenet_v1':
        fmaps = (additional_conv(512), base_model.layers[-39].output, base_model.layers[-53].output)
        map_size = (128, 64)
    elif arch == 'mobilenet_v2':
        fmaps = (additional_conv(96), base_model.layers[-32].output, base_model.layers[-74].output)
        map_size = (32, 16)
    elif arch == 'squeezenet':
        # Quite a bit work here...
        x = additional_conv(128)
        y = _leaky_conv(
            base_featuremap, 128, kernel=1,
            strides=1, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, use_batch_norm=True,
            force_relu=force_relu, name='expand_conv2'
        )
        z = _leaky_conv(
            base_model.layers[-9].output, 128, kernel=1,
            strides=1, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, use_batch_norm=True,
            force_relu=force_relu, name='expand_conv3'
        )
        fmaps = (x, y, z)
        map_size = (64, 64)
    elif arch == 'googlenet':
        # Quite a bit work here...
        x = additional_conv(1024)
        y = _leaky_conv(
            base_model.layers[-21].output, 512, kernel=3,
            strides=1, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, use_batch_norm=True,
            force_relu=force_relu, name='expand_conv2'
        )
        lg_map = base_model.layers[-41].output
        lg_map = layers.UpSampling2D(2, data_format='channels_first',
                                     name='expand_upsample')(lg_map)
        z = _leaky_conv(
            lg_map, 256, kernel=3,
            strides=1, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, use_batch_norm=True,
            force_relu=force_relu, name='expand_conv3'
        )
        fmaps = (x, y, z)
        map_size = (256, 128)
    else:
        raise ValueError("{} architecture is currently not implemented\n".
                         format(arch))

    return fmaps, map_size
