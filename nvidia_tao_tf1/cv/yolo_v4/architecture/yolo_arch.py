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

"""IVA YOLOv4 base architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Concatenate, Conv2D, Permute, Reshape, UpSampling2D
from keras.models import Model

from nvidia_tao_tf1.core.models.quantize_keras_model import create_quantized_keras_model
from nvidia_tao_tf1.core.templates.utils import _leaky_conv, arg_scope
from nvidia_tao_tf1.cv.yolo_v3.layers.yolo_anchor_box_layer import YOLOAnchorBox
from nvidia_tao_tf1.cv.yolo_v4.layers.bbox_postprocessing_layer import BBoxPostProcessingLayer
from nvidia_tao_tf1.cv.yolo_v4.models.base_model import get_base_model


def YOLO_FCN(feature_layers,  # pylint: disable=W0102
             data_format='channels_first',
             use_batch_norm=True,
             kernel_regularizer=None,
             bias_regularizer=None,
             use_bias=False,
             num_anchors=[3, 3, 3],
             num_classes=80,
             force_relu=False,
             activation="leaky_relu"):
    '''
    Build FCN (fully convolutional net) part of YOLOv4.

    Args:
        feature_layers: two elements' list. First element is a tuple of size 3, containing three
            keras tensors as three feature maps. Second element is a tuple of size 2, containing
            number of channels upsampled layers need to have (this should be half of the number of
            channels of the 2x larger feature map).
        data_format: currently only 'channels_first' is tested and supported
        use_batch_norm: whether to use batch norm in FCN build. Note this should be consistent with
            feature extractor.
        kernel_regularizer, bias_regularizer: keras regularizer object or None
        use_bias: whether to use bias for conv layers. If use_batch_norm is true, this should be
            false.
        num_anchors: Number of anchors of different sizes in each feature maps. first element is
            for smallest feature map (i.e. to detect large objects). Last element is for largest
            feature map (i.e. to detect small objects).
        num_classes: Number of all possible classes. E.g. if you have `person, bag, face`, the value
            should be 3.
        force_relu: whether to use ReLU instead of LeakyReLU
        activation(str): Activation type.

    Returns:
        [det_bgobj, det_mdobj, det_smobj]: Three keras tensors for big/mid/small objects detection.
            Those tensors can be processed to get detection boxes.
    '''

    concat_axis = 1 if data_format == 'channels_first' else -1
    concat_num_filters = feature_layers[1]
    x = feature_layers[0][0]
    last_conv_filters = [i * (num_classes + 5) for i in num_anchors]
    with arg_scope([_leaky_conv],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   padding='same',
                   freeze_bn=False,
                   use_bias=use_bias,
                   force_relu=force_relu):
        x = _leaky_conv(x, filters=concat_num_filters[0] * 4,
                        kernel=3, strides=1, name='yolo_conv1_2')
        x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                        kernel=1, strides=1, name='yolo_conv1_3')
        x = _leaky_conv(x, filters=concat_num_filters[0], kernel=1, strides=1, name='yolo_conv2')
        x_branch_0 = x
        x = UpSampling2D(2, data_format=data_format, name='upsample0')(x)
        x_branch = _leaky_conv(
            feature_layers[0][1], filters=concat_num_filters[0],
            kernel=1, strides=1,
            name="yolo_x_branch")
        x = Concatenate(axis=concat_axis)([x, x_branch])
        x = _leaky_conv(x, filters=concat_num_filters[0],
                        kernel=1, strides=1, name='yolo_conv3_1')
        x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                        kernel=3, strides=1, name='yolo_conv3_2')
        x = _leaky_conv(x, filters=concat_num_filters[0],
                        kernel=1, strides=1, name='yolo_conv3_3')
        x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                        kernel=3, strides=1, name='yolo_conv3_4')
        x = _leaky_conv(x, filters=concat_num_filters[0],
                        kernel=1, strides=1, name='yolo_conv3_5')
        x_next_next = x
        x = _leaky_conv(x_next_next, filters=concat_num_filters[1], kernel=1,
                        strides=1, name='yolo_conv4')
        x = UpSampling2D(2, data_format=data_format, name='upsample1')(x)
        x_branch_2 = _leaky_conv(
            feature_layers[0][2], filters=concat_num_filters[1],
            kernel=1, strides=1,
            name="yolo_x_branch_2")
        x = Concatenate(axis=concat_axis)([x, x_branch_2])
        x = _leaky_conv(x, filters=concat_num_filters[1],
                        kernel=1, strides=1, name='yolo_conv5_1')
        x = _leaky_conv(x, filters=concat_num_filters[1] * 2,
                        kernel=3, strides=1, name='yolo_conv5_2')
        x = _leaky_conv(x, filters=concat_num_filters[1],
                        kernel=1, strides=1, name='yolo_conv5_3')
        x = _leaky_conv(x, filters=concat_num_filters[1] * 2,
                        kernel=3, strides=1, name='yolo_conv5_4')
        x = _leaky_conv(x, filters=concat_num_filters[1],
                        kernel=1, strides=1, name='yolo_conv5_5')
        sm_branch = x
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
    with arg_scope([_leaky_conv],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   padding='same',
                   freeze_bn=False,
                   use_bias=use_bias,
                   force_relu=force_relu):
        sm_branch_conv = _leaky_conv(
            sm_branch, filters=concat_num_filters[0],
            kernel=3, strides=2,
            name="yolo_sm_branch_conv")
        x = Concatenate(axis=concat_axis)([x_next_next, sm_branch_conv])
        x = _leaky_conv(x, filters=concat_num_filters[0],
                        kernel=1, strides=1, name='yolo_conv3_5_1')
        x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                        kernel=3, strides=1, name='yolo_conv3_4_2')
        x = _leaky_conv(x, filters=concat_num_filters[0],
                        kernel=1, strides=1, name='yolo_conv3_5_1_1')
        md_leaky = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                               kernel=3, strides=1, name='yolo_conv3_6')
    md_leaky = _leaky_conv(md_leaky, 256, alpha=0.1, kernel=1, strides=1,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           use_batch_norm=True, force_relu=force_relu,
                           name='md_leaky_conv512')
    md_leaky_down = _leaky_conv(md_leaky, 512, alpha=0.1, kernel=3, strides=2,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                use_batch_norm=True, force_relu=force_relu,
                                name='md_leaky_conv512_down')
    md_leaky = _leaky_conv(md_leaky, 512, alpha=0.1, kernel=3, strides=1,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           use_batch_norm=True, force_relu=force_relu,
                           name='md_leaky_conv1024')
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
    with arg_scope([_leaky_conv],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   padding='same',
                   freeze_bn=False,
                   use_bias=use_bias,
                   force_relu=force_relu):
        x = Concatenate(axis=concat_axis)([x_branch_0, md_leaky_down])
        x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                        kernel=1, strides=1, name='yolo_conv1_3_1')
        x = _leaky_conv(x, filters=concat_num_filters[0] * 4,
                        kernel=3, strides=1, name='yolo_conv1_4')
        x = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                        kernel=1, strides=1, name='yolo_conv1_5')
        bg_leaky = _leaky_conv(x, filters=concat_num_filters[0] * 4,
                               kernel=3, strides=1, name='yolo_conv1_6')
    bg_leaky = _leaky_conv(bg_leaky, 512, alpha=0.1, kernel=1, strides=1,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           use_batch_norm=True, force_relu=force_relu,
                           name='bg_leaky_conv512')
    bg_leaky = _leaky_conv(bg_leaky, 1024, alpha=0.1, kernel=3, strides=1,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           use_batch_norm=True, force_relu=force_relu,
                           name='bg_leaky_conv1024')
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
    return [det_bgobj, det_mdobj, det_smobj]


def YOLO_FCN_Tiny(  # pylint: disable=W0102
    feature_layers,
    data_format='channels_first',
    use_batch_norm=True,
    kernel_regularizer=None,
    bias_regularizer=None,
    use_bias=False,
    num_anchors=[3, 3, 3],
    num_classes=80,
    force_relu=False,
    activation="leaky_relu"
):
    '''
    Build FCN (fully convolutional net) part of YOLOv4.

    Args:
        feature_layers: two elements' list. First element is a tuple of size 3, containing three
            keras tensors as three feature maps. Second element is a tuple of size 2, containing
            number of channels upsampled layers need to have (this should be half of the number of
            channels of the 2x larger feature map).
        data_format: currently only 'channels_first' is tested and supported
        use_batch_norm: whether to use batch norm in FCN build. Note this should be consistent with
            feature extractor.
        kernel_regularizer, bias_regularizer: keras regularizer object or None
        use_bias: whether to use bias for conv layers. If use_batch_norm is true, this should be
            false.
        num_anchors: Number of anchors of different sizes in each feature maps. first element is
            for smallest feature map (i.e. to detect large objects). Last element is for largest
            feature map (i.e. to detect small objects).
        num_classes: Number of all possible classes. E.g. if you have `person, bag, face`, the value
            should be 3.
        force_relu: whether to use ReLU instead of LeakyReLU
        activation(str): Activation type.

    Returns:
        [det_bgobj, det_mdobj, det_smobj]: Three keras tensors for big/mid/small objects detection.
            Those tensors can be processed to get detection boxes.
    '''

    concat_axis = 1 if data_format == 'channels_first' else -1
    concat_num_filters = feature_layers[1]
    last_layer = feature_layers[0][0]
    last_conv_filters = [i * (num_classes + 5) for i in num_anchors]
    with arg_scope([_leaky_conv],
                   use_batch_norm=use_batch_norm,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   padding='same',
                   freeze_bn=False,
                   use_bias=use_bias,
                   force_relu=force_relu):
        x = _leaky_conv(last_layer, filters=concat_num_filters[0] * 2,
                        kernel=1, strides=1, name='yolo_conv1_1')
        bg_mish = _leaky_conv(x, filters=concat_num_filters[0] * 4,
                              kernel=3, strides=1, name='yolo_conv1_6')
        x = _leaky_conv(x, filters=concat_num_filters[0], kernel=1, strides=1, name='yolo_conv2')
        x = UpSampling2D(2, data_format=data_format, name='upsample0')(x)
        x = Concatenate(axis=concat_axis)([x, feature_layers[0][1]])
        md_mish = _leaky_conv(x, filters=concat_num_filters[0] * 2,
                              kernel=3, strides=1, name='yolo_conv3_6')
        if len(num_anchors) > 2:
            # tiny-3l
            x = _leaky_conv(md_mish, filters=concat_num_filters[1], kernel=1,
                            strides=1, name='yolo_conv4')
            x = UpSampling2D(2, data_format=data_format, name='upsample1')(x)
            x = Concatenate(axis=concat_axis)([x, feature_layers[0][2]])
            sm_mish = _leaky_conv(x, filters=concat_num_filters[1] * 2,
                                  kernel=3, strides=1, name='yolo_conv5_6')
    if len(num_anchors) > 2:
        det_smobj = Conv2D(filters=last_conv_filters[2],
                           kernel_size=1,
                           strides=1,
                           padding='same',
                           data_format=data_format,
                           activation=None,
                           bias_regularizer=bias_regularizer,
                           kernel_regularizer=kernel_regularizer,
                           use_bias=True,
                           name='conv_sm_object')(sm_mish)
    det_bgobj = Conv2D(filters=last_conv_filters[0],
                       kernel_size=1,
                       strides=1,
                       padding='same',
                       data_format=data_format,
                       activation=None,
                       bias_regularizer=bias_regularizer,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=True,
                       name='conv_big_object')(bg_mish)
    det_mdobj = Conv2D(filters=last_conv_filters[1],
                       kernel_size=1,
                       strides=1,
                       padding='same',
                       data_format=data_format,
                       activation=None,
                       bias_regularizer=bias_regularizer,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=True,
                       name='conv_mid_object')(md_mish)
    if len(num_anchors) == 2:
        return [det_bgobj, det_mdobj]
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
         grid_scale_xy=[1.05, 1.1, 1.1],
         num_classes=80,
         qat=True,
         force_relu=False,
         activation="leaky_relu"):
    '''
    Build YOLO v4 Network.

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
        grid_scale_xy: List of 3 floats indicating how much the grid scale should be (to eliminate
            grid sensitivity. See YOLOv4 paper for details)
        num_classes: Number of all possible classes. E.g. if you have `person, bag, face`, the value
            should be 3
        qat (bool): If `True`, build an quantization aware model.
        force_relu(bool): If `True`, change all LeakyReLU to ReLU.
        activation(str): Activation type.

    Returns:
        model: A keras YOLO v4 model with encoded box detections as output.
    '''

    assert len(anchors) in [2, 3]
    num_anchors = [len(i) for i in anchors]
    feature_layers = get_base_model(
        input_tensor, arch, nlayers, kernel_regularizer,
        bias_regularizer, freeze_blocks, freeze_bn, force_relu,
        activation=activation
    )
    is_tiny = bool(arch in ["cspdarknet_tiny", "cspdarknet_tiny_3l"])
    if is_tiny:
        yolo_fcn = YOLO_FCN_Tiny(
            feature_layers,
            data_format='channels_first',
            use_batch_norm=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            use_bias=False,
            num_anchors=num_anchors,
            num_classes=num_classes,
            force_relu=force_relu,
            activation=activation
        )
    else:
        yolo_fcn = YOLO_FCN(
            feature_layers,
            data_format='channels_first',
            use_batch_norm=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            use_bias=False,
            num_anchors=num_anchors,
            num_classes=num_classes,
            force_relu=force_relu,
            activation=activation
        )
    if qat:
        raw_model = Model(inputs=input_tensor, outputs=yolo_fcn)
        qat_model = create_quantized_keras_model(raw_model)
        if len(anchors) == 3:
            yolo_fcn = [qat_model.get_layer('conv_big_object').output,
                        qat_model.get_layer('conv_mid_object').output,
                        qat_model.get_layer('conv_sm_object').output]
        else:
            yolo_fcn = [qat_model.get_layer('conv_big_object').output,
                        qat_model.get_layer('conv_mid_object').output]

    # [pred_y, pred_x, pred_h, pred_w, object, cls...]
    bgdet = Reshape((-1, num_classes + 5),
                    name="bg_reshape")(Permute((2, 3, 1), name="bg_permute")(yolo_fcn[0]))
    bgdet = BBoxPostProcessingLayer(grid_scale_xy[0], name="bg_bbox_processor")(bgdet)
    mddet = Reshape((-1, num_classes + 5),
                    name="md_reshape")(Permute((2, 3, 1), name="md_permute")(yolo_fcn[1]))
    mddet = BBoxPostProcessingLayer(grid_scale_xy[1], name="md_bbox_processor")(mddet)
    if len(anchors) == 3:
        smdet = Reshape((-1, num_classes + 5),
                        name="sm_reshape")(Permute((2, 3, 1), name="sm_permute")(yolo_fcn[2]))
        smdet = BBoxPostProcessingLayer(grid_scale_xy[2], name="sm_bbox_processor")(smdet)

    # build YOLO v3 anchor layers for corresponding feature maps. Anchor shapes are defined in args.
    bg_anchor = YOLOAnchorBox(anchors[0], name="bg_anchor")(yolo_fcn[0])
    md_anchor = YOLOAnchorBox(anchors[1], name="md_anchor")(yolo_fcn[1])
    if len(anchors) == 3:
        sm_anchor = YOLOAnchorBox(anchors[2], name="sm_anchor")(yolo_fcn[2])
    bgdet = Concatenate(axis=-1, name="encoded_bg")([bg_anchor, bgdet])
    mddet = Concatenate(axis=-1, name="encoded_md")([md_anchor, mddet])
    if len(anchors) == 3:
        smdet = Concatenate(axis=-1, name="encoded_sm")([sm_anchor, smdet])
    if len(anchors) == 3:
        results = Concatenate(axis=-2, name="encoded_detections")([bgdet, mddet, smdet])
    else:
        results = Concatenate(axis=-2, name="encoded_detections")([bgdet, mddet])
    return Model(inputs=input_tensor, outputs=results, name="YOLOv4")
