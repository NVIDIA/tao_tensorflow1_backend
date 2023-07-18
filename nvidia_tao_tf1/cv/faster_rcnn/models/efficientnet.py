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
"""EfficientNet models for FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

from keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dropout,
    Flatten,
    TimeDistributed,
    ZeroPadding2D
)

from nvidia_tao_tf1.core.templates.utils import (
    block,
    CONV_KERNEL_INITIALIZER,
    correct_pad,
    round_filters,
    round_repeats,
    swish
)
from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


DEFAULT_BLOCKS_ARGS = (
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
)


class EfficientNet(FrcnnModel):
    '''EfficientNet as backbones for FasterRCNN model.

    This is EfficientNet class that use FrcnnModel class as base class and do some customization
    specific to EfficientNet backbone. Methods here will override those functions in FrcnnModel
    class.
    '''

    def __init__(self, nlayers, batch_size_per_gpu,
                 rpn_stride, regularizer_type,
                 weight_decay, freeze_bn, freeze_blocks,
                 dropout_rate, drop_connect_rate,
                 conv_bn_share_bias, all_projections,
                 use_pooling, anchor_sizes, anchor_ratios,
                 roi_pool_size, roi_pool_2x, num_classes,
                 std_scaling, rpn_pre_nms_top_N, rpn_post_nms_top_N,
                 rpn_nms_iou_thres, gt_as_roi, rcnn_min_overlap,
                 rcnn_max_overlap, rcnn_train_bs, rcnn_bbox_std,
                 rpn_train_bs, lambda_rpn_class, lambda_rpn_regr,
                 lambda_rcnn_class, lambda_rcnn_regr,
                 backbone, results_dir, enc_key, lr_config,
                 enable_qat=False, **kwargs):
        '''Initialize the EfficientNet FasterRCNN model.'''
        super(EfficientNet, self).__init__(
            nlayers, batch_size_per_gpu,
            rpn_stride, regularizer_type,
            weight_decay, freeze_bn, freeze_blocks,
            dropout_rate, drop_connect_rate,
            conv_bn_share_bias, all_projections,
            use_pooling, anchor_sizes, anchor_ratios,
            roi_pool_size, roi_pool_2x, num_classes,
            std_scaling, rpn_pre_nms_top_N, rpn_post_nms_top_N,
            rpn_nms_iou_thres, gt_as_roi, rcnn_min_overlap,
            rcnn_max_overlap, rcnn_train_bs, rcnn_bbox_std,
            rpn_train_bs, lambda_rpn_class, lambda_rpn_regr,
            lambda_rcnn_class, lambda_rcnn_regr,
            backbone, results_dir, enc_key, lr_config, enable_qat,
            **kwargs
        )
        # constant parameters specific to EfficientNet
        # activation fn defaults to swish if unspecified.
        if self.activation_type in [None, ""]:
            self.activation_type = swish
        self.depth_divisor = 8
        if self.nlayers == "b0":
            self.width_coefficient = 1.0
            self.depth_coefficient = 1.0
        elif self.nlayers == "b1":
            self.width_coefficient = 1.0
            self.depth_coefficient = 1.1
        elif self.nlayers == "b2":
            self.width_coefficient = 1.1
            self.depth_coefficient = 1.2
        elif self.nlayers == "b3":
            self.width_coefficient = 1.2
            self.depth_coefficient = 1.4
        elif self.nlayers == "b4":
            self.width_coefficient = 1.4
            self.depth_coefficient = 1.8
        elif self.nlayers == "b5":
            self.width_coefficient = 1.6
            self.depth_coefficient = 2.2
        elif self.nlayers == "b6":
            self.width_coefficient = 1.8
            self.depth_coefficient = 2.6
        elif self.nlayers == "b7":
            self.width_coefficient = 2.0
            self.depth_coefficient = 3.1
        else:
            raise ValueError("Unsupported EfficientNet {} architecture.".format(self.nlayers))

    def backbone(self, x):
        '''backbone of the ResNet FasterRCNN model.'''
        bn_axis = 1
        x = ZeroPadding2D(
            padding=correct_pad(x, 3),
            name='stem_conv_pad'
        )(x)
        x = Conv2D(
            round_filters(32, self.depth_divisor, self.width_coefficient),
            3,
            strides=2,
            padding='valid',
            use_bias=not self.conv_bn_share_bias,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=self.kernel_reg,
            trainable=not bool(0 in self.freeze_blocks),
            name='stem_conv'
        )(x)
        if self.freeze_bn:
            x = BatchNormalization(axis=bn_axis, name='stem_bn')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, name='stem_bn')(x)
        x = Activation(self.activation_type, name='stem_activation')(x)
        blocks_args = deepcopy(list(DEFAULT_BLOCKS_ARGS))
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args))
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(
                args['filters_in'],
                self.depth_divisor,
                self.width_coefficient
            )
            args['filters_out'] = round_filters(
                args['filters_out'],
                self.depth_divisor,
                self.width_coefficient
            )
            for j in range(round_repeats(args.pop('repeats'), self.depth_coefficient)):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                # skip the last two blocks, will use it in RCNN body
                if i < len(blocks_args) - 2:
                    x = block(
                        x, self.activation_type,
                        self.drop_connect_rate * b / blocks,
                        freeze=bool((i + 1) in self.freeze_blocks),
                        freeze_bn=self.freeze_bn,
                        use_bias=not self.conv_bn_share_bias,
                        kernel_regularizer=self.kernel_reg,
                        name='block{}{}_'.format(i + 1, chr(j + 97)),
                        **args
                    )
                b += 1
        return x

    def rcnn_body(self, x):
        '''RCNN body.'''
        data_format = 'channels_first'
        bn_axis = 1
        _stride = 2 if self.roi_pool_2x else 1
        blocks_args = deepcopy(list(DEFAULT_BLOCKS_ARGS))
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args))
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(
                args['filters_in'],
                self.depth_divisor,
                self.width_coefficient
            )
            args['filters_out'] = round_filters(
                args['filters_out'],
                self.depth_divisor,
                self.width_coefficient
            )
            for j in range(round_repeats(args.pop('repeats'), self.depth_coefficient)):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                elif i == len(blocks_args) - 2:
                    args['strides'] = _stride
                # only use the last two blocks
                if i >= len(blocks_args) - 2:
                    x = block(
                        x, self.activation_type,
                        self.drop_connect_rate * b / blocks,
                        freeze=bool((i + 1) in self.freeze_blocks),
                        freeze_bn=self.freeze_bn,
                        use_bias=not self.conv_bn_share_bias,
                        kernel_regularizer=self.kernel_reg,
                        name='block{}{}_'.format(i + 1, chr(j + 97)),
                        use_td=True,
                        **args
                    )
                b += 1
        # Build top
        layer = Conv2D(
            round_filters(1280, self.depth_divisor, self.width_coefficient),
            1,
            padding='same',
            use_bias=not self.conv_bn_share_bias,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            trainable=not bool((len(blocks_args) + 1) in self.freeze_blocks),
            kernel_regularizer=self.kernel_reg,
            name='top_conv'
        )
        x = TimeDistributed(layer)(x)
        layer = BatchNormalization(axis=bn_axis, name='top_bn')
        if self.freeze_bn:
            x = TimeDistributed(layer)(x, training=False)
        else:
            x = TimeDistributed(layer)(x)
        x = Activation(self.activation_type, name='top_activation')(x)
        layer = AveragePooling2D(
            pool_size=self.roi_pool_size, name='avg_pool',
            data_format=data_format, padding='valid'
        )
        x = TimeDistributed(layer)(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        x = TimeDistributed(Flatten(name='flatten'), name="time_distributed_flatten")(x)
        if self.dropout_rate > 0:
            x = TimeDistributed(Dropout(self.dropout_rate, name='top_dropout'))(x)
        return x
