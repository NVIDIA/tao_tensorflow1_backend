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

"""FasterRCNN model templates for VGG16/19."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import TimeDistributed

from nvidia_tao_tf1.core.templates.utils import arg_scope, CNNBlock
from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


class IVAVGG(FrcnnModel):
    '''IVA VGG as backbones for FasterRCNN model.

    This is IVA VGG class that use FrcnnModel class as base class and do some customization
    specific to IVA VGG backbone. Methods here will override those functions in FrcnnModel class.
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
        '''Initialize the IVA VGG backbones.'''
        assert nlayers in [16, 19], '''Number of layers for VGG can
         only be 16, 19, got {}'''.format(nlayers)
        super(IVAVGG, self).__init__(nlayers, batch_size_per_gpu,
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
        '''backbones of IVA VGG FasterRCNN.'''
        data_format = 'channels_first'
        first_stride = 1
        stride = 2
        if self.use_pooling:
            # Disable strided convolutions with pooling enabled.
            stride = 1
        freeze1 = 1 in self.freeze_blocks
        freeze2 = 2 in self.freeze_blocks
        freeze3 = 3 in self.freeze_blocks
        freeze4 = 4 in self.freeze_blocks
        freeze5 = 5 in self.freeze_blocks
        # Define a block functor which can create blocks.
        with arg_scope([CNNBlock],
                       use_batch_norm=True,
                       freeze_bn=self.freeze_bn,
                       use_bias=not self.conv_bn_share_bias,
                       use_shortcuts=False,
                       data_format='channels_first',
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       activation_type='relu'):
            # Implementing VGG 16 architecture.
            if self.nlayers == 16:
                # Block - 1.
                x = CNNBlock(repeat=2, stride=first_stride, subblocks=[(3, 64)],
                             index=1, freeze_block=freeze1)(input_images)
                if self.use_pooling:
                    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                     data_format=data_format, name='block1_pool')(x)
                # Block - 2.
                x = CNNBlock(repeat=2, stride=stride, subblocks=[(3, 128)],
                             index=2, freeze_block=freeze2)(x)
                if self.use_pooling:
                    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                     data_format=data_format, name='block2_pool')(x)
                # Block - 3.
                x = CNNBlock(repeat=3, stride=stride, subblocks=[(3, 256)], index=3,
                             freeze_block=freeze3)(x)
                if self.use_pooling:
                    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                     data_format=data_format, name='block3_pool')(x)
                # Block - 4.
                x = CNNBlock(repeat=3, stride=stride, subblocks=[(3, 512)], index=4,
                             freeze_block=freeze4)(x)
                if self.use_pooling:
                    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                     data_format=data_format, name='block4_pool')(x)
                # Block - 5.
                x = CNNBlock(repeat=3, stride=stride, subblocks=[(3, 512)], index=5,
                             freeze_block=freeze5)(x)
            # Implementing VGG 19 architecture.
            elif self.nlayers == 19:
                # Block - 1.
                x = CNNBlock(repeat=2, stride=first_stride, subblocks=[(3, 64)], index=1,
                             freeze_block=freeze1)(input_images)
                if self.use_pooling:
                    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                     data_format=data_format, name='block1_pool')(x)
                # Block - 2.
                x = CNNBlock(repeat=2, stride=stride, subblocks=[(3, 128)], index=2,
                             freeze_block=freeze2)(x)
                if self.use_pooling:
                    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                     data_format=data_format, name='block2_pool')(x)
                # Block - 3.
                x = CNNBlock(repeat=4, stride=stride, subblocks=[(3, 256)], index=3,
                             freeze_block=freeze3)(x)
                if self.use_pooling:
                    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                     data_format=data_format, name='block3_pool')(x)
                # Block - 4.
                x = CNNBlock(repeat=4, stride=stride, subblocks=[(3, 512)], index=4,
                             freeze_block=freeze4)(x)
                if self.use_pooling:
                    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                     data_format=data_format, name='block4_pool')(x)
                # Block - 5.
                x = CNNBlock(repeat=4, stride=stride, subblocks=[(3, 512)], index=5,
                             freeze_block=freeze5)(x)
            else:
                raise NotImplementedError('''A VGG with nlayers=%d is not
                                           implemented.''' % self.nlayers)
        return x

    def rcnn_body(self, x):
        '''RCNN body for IVA VGG backbones.'''
        if self.roi_pool_2x:
            x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='classifier_pool'))(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        out = TimeDistributed(Flatten(name='classifier_flatten'),
                              name='time_distributed_flatten')(x)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1',
                              kernel_regularizer=self.kernel_reg,
                              bias_regularizer=None))(out)
        if self.dropout_rate > 0:
            out = TimeDistributed(Dropout(self.dropout_rate))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2',
                              kernel_regularizer=self.kernel_reg,
                              bias_regularizer=None))(out)
        if self.dropout_rate > 0:
            out = TimeDistributed(Dropout(self.dropout_rate))(out)
        return out
