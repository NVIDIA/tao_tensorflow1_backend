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
"""ResNet models for FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, AveragePooling2D, BatchNormalization, \
                         Conv2D, Flatten, \
                         MaxPooling2D, TimeDistributed

from nvidia_tao_tf1.core.templates.utils import arg_scope, CNNBlock
from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


class ResNet(FrcnnModel):
    '''ResNet as backbones for FasterRCNN model.

    This is ResNet class that use FrcnnModel class as base class and do some customization
    specific to ResNet backbone. Methods here will override those functions in FrcnnModel class.
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
                 backbone, results_dir, enc_key, lr_config, enable_qat=False,
                 **kwargs):
        '''Initialize the ResNet FasterRCNN model.'''
        assert nlayers in [10, 18, 34, 50, 101], 'Number of layers for ResNet can ' \
            'only be 10, 18, 34, 50, 101, got {}'.format(nlayers)
        super(ResNet, self).__init__(nlayers, batch_size_per_gpu,
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
                                     **kwargs)

    def backbone(self, input_images):
        '''backbone of the ResNet FasterRCNN model.'''
        bn_axis = 1
        data_format = 'channels_first'

        freeze0 = bool(0 in self.freeze_blocks)
        freeze1 = bool(1 in self.freeze_blocks)
        freeze2 = bool(2 in self.freeze_blocks)
        freeze3 = bool(3 in self.freeze_blocks)

        x = Conv2D(
            64, (7, 7),
            use_bias=not self.conv_bn_share_bias,
            strides=(2, 2),
            padding='same',
            kernel_regularizer=self.kernel_reg,
            name='conv1', trainable=not freeze0)(input_images)

        if self.freeze_bn:
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x, training=False)
        else:
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)

        x = Activation('relu')(x)

        if self.use_pooling:
            x = MaxPooling2D(
                    pool_size=(3, 3), strides=(2, 2), padding='same')(x)
            first_stride = 1
        else:
            first_stride = 2

        with arg_scope(
                [CNNBlock],
                use_batch_norm=True,
                all_projections=self.all_projections,
                use_shortcuts=True,
                data_format=data_format,
                kernel_regularizer=self.kernel_reg,
                bias_regularizer=None,
                activation_type='relu',
                freeze_bn=self.freeze_bn,
                activation_kwargs={},
                use_bias=not self.conv_bn_share_bias):
            if self.nlayers == 10:
                x = CNNBlock(repeat=1, stride=first_stride,
                             subblocks=[(3, 64), (3, 64)],
                             index=1, freeze_block=freeze1)(x)
                x = CNNBlock(repeat=1, stride=2,
                             subblocks=[(3, 128), (3, 128)],
                             index=2, freeze_block=freeze2)(x)
                x = CNNBlock(repeat=1, stride=2,
                             subblocks=[(3, 256), (3, 256)],
                             index=3, freeze_block=freeze3)(x)
            elif self.nlayers == 18:
                x = CNNBlock(repeat=2, stride=first_stride,
                             subblocks=[(3, 64), (3, 64)],
                             index=1, freeze_block=freeze1)(x)
                x = CNNBlock(repeat=2, stride=2,
                             subblocks=[(3, 128), (3, 128)],
                             index=2, freeze_block=freeze2)(x)
                x = CNNBlock(repeat=2, stride=2,
                             subblocks=[(3, 256), (3, 256)],
                             index=3, freeze_block=freeze3)(x)
            elif self.nlayers == 34:
                x = CNNBlock(repeat=3, stride=first_stride,
                             subblocks=[(3, 64), (3, 64)],
                             index=1, freeze_block=freeze1)(x)
                x = CNNBlock(repeat=4, stride=2,
                             subblocks=[(3, 128), (3, 128)],
                             index=2, freeze_block=freeze2)(x)
                x = CNNBlock(repeat=6, stride=2,
                             subblocks=[(3, 256), (3, 256)],
                             index=3, freeze_block=freeze3)(x)
            elif self.nlayers == 50:
                x = CNNBlock(repeat=3, stride=first_stride,
                             subblocks=[(1, 64), (3, 64), (1, 256)],
                             index=1, freeze_block=freeze1)(x)
                x = CNNBlock(repeat=4, stride=2,
                             subblocks=[(1, 128), (3, 128), (1, 512)],
                             index=2, freeze_block=freeze2)(x)
                x = CNNBlock(repeat=6, stride=2,
                             subblocks=[(1, 256), (3, 256), (1, 1024)],
                             index=3, freeze_block=freeze3)(x)
            elif self.nlayers == 101:
                x = CNNBlock(repeat=3, stride=first_stride,
                             subblocks=[(1, 64), (3, 64), (1, 256)],
                             index=1, freeze_block=freeze1)(x)
                x = CNNBlock(repeat=4, stride=2,
                             subblocks=[(1, 128), (3, 128), (1, 512)],
                             index=2, freeze_block=freeze2)(x)
                x = CNNBlock(repeat=23, stride=2,
                             subblocks=[(1, 256), (3, 256), (1, 1024)],
                             index=3, freeze_block=freeze3)(x)
            elif self.nlayers == 152:
                x = CNNBlock(repeat=3, stride=first_stride,
                             subblocks=[(1, 64), (3, 64), (1, 256)],
                             index=1, freeze_block=freeze1)(x)
                x = CNNBlock(repeat=8, stride=2,
                             subblocks=[(1, 128), (3, 128), (1, 512)],
                             index=2, freeze_block=freeze2)(x)
                x = CNNBlock(repeat=36, stride=2,
                             subblocks=[(1, 256), (3, 256), (1, 1024)],
                             index=3, freeze_block=freeze3)(x)
            else:
                raise NotImplementedError('''A resnet with nlayers=%d is
                                           not implemented.''' % self.nlayers)
        return x

    def rcnn_body(self, x):
        '''RCNN body.'''
        data_format = 'channels_first'
        _stride = 2 if self.roi_pool_2x else 1

        with arg_scope([CNNBlock],
                       use_batch_norm=True,
                       all_projections=self.all_projections,
                       use_shortcuts=True,
                       data_format=data_format,
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       activation_type='relu',
                       freeze_bn=self.freeze_bn,
                       activation_kwargs={},
                       use_bias=not self.conv_bn_share_bias,
                       use_td=True):
            if self.nlayers == 10:
                x = CNNBlock(repeat=1, stride=_stride,
                             subblocks=[(3, 512), (3, 512)], index=4)(x)
            elif self.nlayers == 18:
                x = CNNBlock(repeat=2, stride=_stride,
                             subblocks=[(3, 512), (3, 512)], index=4)(x)
            elif self.nlayers == 34:
                x = CNNBlock(repeat=3, stride=_stride,
                             subblocks=[(3, 512), (3, 512)], index=4)(x)
            elif self.nlayers == 50:
                x = CNNBlock(repeat=3, stride=_stride,
                             subblocks=[(1, 512), (3, 512), (1, 2048)], index=4)(x)
            elif self.nlayers == 101:
                x = CNNBlock(repeat=3, stride=_stride,
                             subblocks=[(1, 512), (3, 512), (1, 2048)], index=4)(x)
            elif self.nlayers == 152:
                x = CNNBlock(repeat=3, stride=_stride,
                             subblocks=[(1, 512), (3, 512), (1, 2048)], index=4)(x)
            else:
                raise NotImplementedError('''A resnet with nlayers=%d is
                                           not implemented.''' % self.nlayers)

        x = TimeDistributed(AveragePooling2D((self.roi_pool_size, self.roi_pool_size),
                                             name='avg_pool'))(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        x = TimeDistributed(Flatten(name='classifier_flatten'),
                            name='time_distributed_flatten')(x)
        return x
