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

"""IVA YOLO base architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Concatenate, Conv2D, Permute, Reshape, UpSampling2D
from keras.models import Model

from nvidia_tao_tf1.core.models.quantize_keras_model import create_quantized_keras_model
from nvidia_tao_tf1.core.templates.utils import _leaky_conv, arg_scope
from nvidia_tao_tf1.cv.yolo_v3.layers.yolo_anchor_box_layer import YOLOAnchorBox
from nvidia_tao_tf1.cv.yolo_v3.models.base_model import get_base_model


def YOLO_FCN(feature_layers,  # pylint: disable=W0102
             data_format='channels_first',
             use_batch_norm=True,
             kernel_regularizer=None,
             bias_regularizer=None,
             alpha=0.1,
             use_bias=False,
             num_anchors=[3, 3, 3],
             num_classes=80,
             arch_conv_blocks=2,
             force_relu=False):
    '''
    Build FCN (fully convolutional net) part of YOLO.

    Args:
        feature_layers: two elements' list. First element is a tuple of size 3, containing three
            keras tensors as three feature maps. Second element is a tuple of size 2, containing
            number of channels upsampled layers need to have (this should be half of the number of
            channels of the 2x larger feature map).
        data_format: currently only 'channels_first' is tested and supported
        use_batch_norm: whether to use batch norm in FCN build. Note this should be consistent with
            feature extractor.
        kernel_regularizer, bias_regularizer: keras regularizer object or None
        alpha: Alpha for leakyReLU in FCN build. Note this is value does not apply to feature
            extractor. if x is negative, lReLU(x) = alpha * x
        use_bias: whether to use bias for conv layers. If use_batch_norm is true, this should be
            false.
        num_anchors: Number of anchors of different sizes in each feature maps. first element is
            for smallest feature map (i.e. to detect large objects). Last element is for largest
            feature map (i.e. to detect small objects).
        num_classes: Number of all possible classes. E.g. if you have `person, bag, face`, the value
            should be 3.
        arch_conv_blocks: How many leaky conv blocks to attach before detection layer.
        force_relu: whether to use ReLU instead of LeakyReLU
    Returns:
        [det_bgobj, det_mdobj, det_smobj]: Three keras tensors for big/mid/small objects detection.
            Those tensors can be processed to get detection boxes.
    '''

    concat_axis = 1 if data_format == 'channels_first' else -1
    concat_num_filters = feature_layers[1]
    last_layer = feature_layers[0][0]
    last_conv_filters = [i * (num_classes + 5) for i in num_anchors]
    assert arch_conv_blocks < 3, "arch_conv_blocks can only be 0, 1 or 2."

    with arg_scope([_leaky_conv],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   alpha=alpha,
                   padding='same',
                   freeze_bn=False,
                   use_bias=use_bias,
                   force_relu=force_relu):

        x = _leaky_conv(last_layer, filters=concat_num_filters[0] * 2,
                        kernel=1, strides=1, name='yolo_conv1_1')

        if arch_conv_blocks > 0:
            x = _leaky_conv(x, filters=concat_num_filters[0] * 4,
                            kernel=3, strides=1, name='yolo_conv1_2')
            x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                            kernel=1, strides=1, name='yolo_conv1_3')
        if arch_conv_blocks > 1:
            x = _leaky_conv(x, filters=concat_num_filters[0] * 4,
                            kernel=3, strides=1, name='yolo_conv1_4')
            x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                            kernel=1, strides=1, name='yolo_conv1_5')

        bg_leaky = _leaky_conv(x, filters=concat_num_filters[0] * 4,
                               kernel=3, strides=1, name='yolo_conv1_6')

        x = _leaky_conv(x, filters=concat_num_filters[0], kernel=1, strides=1, name='yolo_conv2')
        x = UpSampling2D(2, data_format=data_format, name='upsample0')(x)
        x = Concatenate(axis=concat_axis)([x, feature_layers[0][1]])
        x = _leaky_conv(x, filters=concat_num_filters[0],
                        kernel=1, strides=1, name='yolo_conv3_1')

        if arch_conv_blocks > 0:
            x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                            kernel=3, strides=1, name='yolo_conv3_2')
            x = _leaky_conv(x, filters=concat_num_filters[0],
                            kernel=1, strides=1, name='yolo_conv3_3')

        if arch_conv_blocks > 1:
            x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                            kernel=3, strides=1, name='yolo_conv3_4')
            x = _leaky_conv(x, filters=concat_num_filters[0],
                            kernel=1, strides=1, name='yolo_conv3_5')

        md_leaky = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                               kernel=3, strides=1, name='yolo_conv3_6')

        x = _leaky_conv(x, filters=concat_num_filters[1], kernel=1, strides=1, name='yolo_conv4')
        x = UpSampling2D(2, data_format=data_format, name='upsample1')(x)
        x = Concatenate(axis=concat_axis)([x, feature_layers[0][2]])
        x = _leaky_conv(x, filters=concat_num_filters[1],
                        kernel=1, strides=1, name='yolo_conv5_1')

        if arch_conv_blocks > 0:
            x = _leaky_conv(x, filters=concat_num_filters[1] * 2,
                            kernel=3, strides=1, name='yolo_conv5_2')
            x = _leaky_conv(x, filters=concat_num_filters[1],
                            kernel=1, strides=1, name='yolo_conv5_3')
        if arch_conv_blocks > 1:
            x = _leaky_conv(x, filters=concat_num_filters[1] * 2,
                            kernel=3, strides=1, name='yolo_conv5_4')
            x = _leaky_conv(x, filters=concat_num_filters[1],
                            kernel=1, strides=1, name='yolo_conv5_5')

        sm_leaky = _leaky_conv(x, filters=concat_num_filters[1] * 2,
                               kernel=3, strides=1, name='yolo_conv5_6')

    det_smobj = Conv2D(filters=last_conv_filters[2],
                       kernel_size=1,
                       strides=1,
                       padding='same',
                       data_format=data_format,
                       activation=None,
                       bias_regularizer=bias_regularizer,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=True,
                       name='conv_sm_object')(sm_leaky)
    det_bgobj = Conv2D(filters=last_conv_filters[0],
                       kernel_size=1,
                       strides=1,
                       padding='same',
                       data_format=data_format,
                       activation=None,
                       bias_regularizer=bias_regularizer,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=True,
                       name='conv_big_object')(bg_leaky)
    det_mdobj = Conv2D(filters=last_conv_filters[1],
                       kernel_size=1,
                       strides=1,
                       padding='same',
                       data_format=data_format,
                       activation=None,
                       bias_regularizer=bias_regularizer,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=True,
                       name='conv_mid_object')(md_leaky)

    return [det_bgobj, det_mdobj, det_smobj]


def YOLO(input_tensor,  # pylint: disable=W0102
         arch,
         nlayers,
         kernel_regularizer=None,
         bias_regularizer=None,
         freeze_blocks=None,
         freeze_bn=None,
         anchors=[[(0.279, 0.216), (0.375, 0.476), (0.897, 0.784)],
                  [(0.072, 0.147), (0.149, 0.108), (0.142, 0.286)],
                  [(0.024, 0.031), (0.038, 0.072), (0.079, 0.055)]],
         num_classes=80,
         arch_conv_blocks=2,
         qat=True,
         force_relu=False):
    '''
    Build YOLO v3 Network.

    Args:
        input_tensor: Keras tensor created by Input layer
        arch: architecture of feature extractors. E.g. resnet18, resnet10, darknet53
        kernel_regularizer, bias_regularizer: keras regularizer object or None
        freeze_blocks: blocks to freeze during training. The meaning of `block` is arch-specific
        freeze_bn: whether to freeze batch norm layer **for feature extractors**
        anchors: List of 3 elements indicating the anchor boxes shape on feature maps. first element
            is for smallest feature map (i.e. to detect large objects). Last element is for largest
            feature map (i.e. to detect small objects). Each element is a list of tuples of size 2,
            in the format of (w, h). The length of the list can be any integer larger than 0.
        num_classes: Number of all possible classes. E.g. if you have `person, bag, face`, the value
            should be 3
        arch_conv_blocks: Number of optional conv blocks to attach after each feature map.
        qat (bool): If `True`, build an quantization aware model.
        force_relu(bool): If `True`, change all LeakyReLU to ReLU

    Returns:
        model: A keras YOLO v3 model with encoded box detections as output.
    '''

    assert len(anchors) == 3
    num_anchors = [len(i) for i in anchors]
    feature_layers = get_base_model(input_tensor, arch, nlayers, kernel_regularizer,
                                    bias_regularizer, freeze_blocks, freeze_bn, force_relu)

    yolo_fcn = YOLO_FCN(feature_layers,
                        data_format='channels_first',
                        use_batch_norm=True,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        alpha=0.1,
                        use_bias=False,
                        num_anchors=num_anchors,
                        num_classes=num_classes,
                        arch_conv_blocks=arch_conv_blocks,
                        force_relu=force_relu)
    if qat:
        raw_model = Model(inputs=input_tensor, outputs=yolo_fcn)
        qat_model = create_quantized_keras_model(raw_model)
        yolo_fcn = [qat_model.get_layer('conv_big_object').output,
                    qat_model.get_layer('conv_mid_object').output,
                    qat_model.get_layer('conv_sm_object').output]

    bgdet = Reshape((-1, num_classes + 5),
                    name="bg_reshape")(Permute((2, 3, 1), name="bg_permute")(yolo_fcn[0]))
    mddet = Reshape((-1, num_classes + 5),
                    name="md_reshape")(Permute((2, 3, 1), name="md_permute")(yolo_fcn[1]))
    smdet = Reshape((-1, num_classes + 5),
                    name="sm_reshape")(Permute((2, 3, 1), name="sm_permute")(yolo_fcn[2]))

    # build YOLO v3 anchor layers for corresponding feature maps. Anchor shapes are defined in args.
    bg_anchor = YOLOAnchorBox(anchors[0], name="bg_anchor")(yolo_fcn[0])
    md_anchor = YOLOAnchorBox(anchors[1], name="md_anchor")(yolo_fcn[1])
    sm_anchor = YOLOAnchorBox(anchors[2], name="sm_anchor")(yolo_fcn[2])
    bgdet = Concatenate(axis=-1, name="encoded_bg")([bg_anchor, bgdet])
    mddet = Concatenate(axis=-1, name="encoded_md")([md_anchor, mddet])
    smdet = Concatenate(axis=-1, name="encoded_sm")([sm_anchor, smdet])
    results = Concatenate(axis=-2, name="encoded_detections")([bgdet, mddet, smdet])
    return Model(inputs=input_tensor, outputs=results, name="YOLOv3")
