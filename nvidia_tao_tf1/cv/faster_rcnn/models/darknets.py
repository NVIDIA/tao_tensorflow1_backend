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
"""DarkNet 19/53 as backbone of Faster-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers
from keras.layers import AveragePooling2D, Flatten, TimeDistributed

from nvidia_tao_tf1.core.templates.utils import _leaky_conv, arg_scope
from nvidia_tao_tf1.cv.faster_rcnn.models.model_builder import FrcnnModel


class DarkNet(FrcnnModel):
    '''DarkNet as backbone of FasterRCNN model.

    This is DarkNet class that use FrcnnModel class as base class and do some customization
    specific to DarkNet backbone. Methods here will override those functions in FrcnnModel class.
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
        '''Initialize the DarkNet FasterRCNN model.

        See the docstring in FrcnnModel constructor.
        '''
        assert nlayers in [19, 53], '''Number of layers for DarkNet can
         only be 19, 53, got {}'''.format(nlayers)
        super(DarkNet, self).__init__(nlayers, batch_size_per_gpu,
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
        '''DarkNet backbone implementation.'''
        data_format = 'channels_first'
        with arg_scope([_leaky_conv],
                       use_batch_norm=True,
                       data_format=data_format,
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       alpha=0.1,
                       padding='same',
                       freeze_bn=self.freeze_bn,
                       use_bias=not self.conv_bn_share_bias):
            x = _leaky_conv(input_images, filters=32, kernel=3, strides=1,
                            name='conv1', trainable=not(0 in self.freeze_blocks))
            if self.nlayers == 53:
                x = _leaky_conv(x, filters=64, kernel=3, strides=2, name='conv2',
                                trainable=not(1 in self.freeze_blocks))
                y = _leaky_conv(x, filters=32, kernel=1, strides=1, name='b1_conv1_1',
                                trainable=not(1 in self.freeze_blocks))
                y = _leaky_conv(y, filters=64, kernel=3, strides=1, name='b1_conv1_2',
                                trainable=not(1 in self.freeze_blocks))
                x = layers.Add(name='b1_add1')([x, y])
                x = _leaky_conv(x, filters=128, kernel=3, strides=2, name='conv3',
                                trainable=not(2 in self.freeze_blocks))
                for i in range(2):
                    y = _leaky_conv(x, filters=64, kernel=1, strides=1,
                                    name='b2_conv{}_1'.format(i+1),
                                    trainable=not(2 in self.freeze_blocks))
                    y = _leaky_conv(y, filters=128, kernel=3, strides=1,
                                    name='b2_conv{}_2'.format(i+1),
                                    trainable=not(2 in self.freeze_blocks))
                    x = layers.Add(name='b2_add{}'.format(i+1))([x, y])
                x = _leaky_conv(x, filters=256, kernel=3, strides=2, name='conv4',
                                trainable=not(3 in self.freeze_blocks))
                for i in range(8):
                    y = _leaky_conv(x, filters=128, kernel=1, strides=1,
                                    name='b3_conv{}_1'.format(i+1),
                                    trainable=not(3 in self.freeze_blocks))
                    y = _leaky_conv(y, filters=256, kernel=3, strides=1,
                                    name='b3_conv{}_2'.format(i+1),
                                    trainable=not(3 in self.freeze_blocks))
                    x = layers.Add(name='b3_add{}'.format(i+1))([x, y])
                x = _leaky_conv(x, filters=512, kernel=3, strides=2, name='conv5',
                                trainable=not(4 in self.freeze_blocks))
                for i in range(8):
                    y = _leaky_conv(x, filters=256, kernel=1, strides=1,
                                    name='b4_conv{}_1'.format(i+1),
                                    trainable=not(4 in self.freeze_blocks))
                    y = _leaky_conv(y, filters=512, kernel=3, strides=1,
                                    name='b4_conv{}_2'.format(i+1),
                                    trainable=not(4 in self.freeze_blocks))
                    x = layers.Add(name='b4_add{}'.format(i+1))([x, y])
            elif self.nlayers == 19:
                x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                        padding='same', name='maxpool_1')(x)
                x = _leaky_conv(x, filters=64, kernel=3, strides=1, name='b1_conv1',
                                trainable=not(1 in self.freeze_blocks))
                x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                        padding='same', name='maxpool_2')(x)
                x = _leaky_conv(x, filters=128, kernel=3, strides=1,
                                name='b2_conv1', trainable=not(2 in self.freeze_blocks))
                x = _leaky_conv(x, filters=64, kernel=1, strides=1,
                                name='b2_conv2', trainable=not(2 in self.freeze_blocks))
                x = _leaky_conv(x, filters=128, kernel=3, strides=1,
                                name='b2_conv3', trainable=not(2 in self.freeze_blocks))
                x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                        padding='same', name='maxpool_3')(x)
                x = _leaky_conv(x, filters=256, kernel=3, strides=1,
                                name='b3_conv1', trainable=not(3 in self.freeze_blocks))
                x = _leaky_conv(x, filters=128, kernel=1, strides=1,
                                name='b3_conv2', trainable=not(3 in self.freeze_blocks))
                x = _leaky_conv(x, filters=256, kernel=3, strides=1,
                                name='b3_conv3', trainable=not(3 in self.freeze_blocks))
                x = layers.MaxPooling2D(pool_size=2, strides=2, data_format=data_format,
                                        padding='same', name='maxpool_4')(x)
                x = _leaky_conv(x, filters=512, kernel=3, strides=1,
                                name='b4_conv1', trainable=not(4 in self.freeze_blocks))
                x = _leaky_conv(x, filters=256, kernel=1, strides=1,
                                name='b4_conv2', trainable=not(4 in self.freeze_blocks))
                x = _leaky_conv(x, filters=512, kernel=3, strides=1,
                                name='b4_conv3', trainable=not(4 in self.freeze_blocks))
                x = _leaky_conv(x, filters=256, kernel=1, strides=1,
                                name='b4_conv4', trainable=not(4 in self.freeze_blocks))
                x = _leaky_conv(x, filters=512, kernel=3, strides=1,
                                name='b4_conv5', trainable=not(4 in self.freeze_blocks))
            else:
                raise NotImplementedError('''
                          A DarkNet with nlayers=%d is not implemented.''' % self.nlayers)
        return x

    def rcnn_body(self, x):
        '''DarkNet RCNN body.'''
        if self.roi_pool_2x:
            _stride = 2
        else:
            _stride = 1
        with arg_scope([_leaky_conv],
                       use_batch_norm=True,
                       data_format='channels_first',
                       kernel_regularizer=self.kernel_reg,
                       bias_regularizer=None,
                       alpha=0.1,
                       padding='same',
                       freeze_bn=self.freeze_bn,
                       use_bias=not self.conv_bn_share_bias,
                       use_td=True):
            if self.nlayers == 53:
                x = _leaky_conv(x, filters=1024, kernel=3, strides=_stride, name='conv6',
                                trainable=not(5 in self.freeze_blocks))
                for i in range(4):
                    y = _leaky_conv(x, filters=512, kernel=1, strides=1,
                                    name='b5_conv{}_1'.format(i+1),
                                    trainable=not(5 in self.freeze_blocks))
                    y = _leaky_conv(y, filters=1024, kernel=3, strides=1,
                                    name='b5_conv{}_2'.format(i+1),
                                    trainable=not(5 in self.freeze_blocks))
                    x = layers.Add(name='b5_add{}'.format(i+1))([x, y])
            elif self.nlayers == 19:
                x = _leaky_conv(x, filters=1024, kernel=3, strides=_stride,
                                name='b5_conv1', trainable=not(5 in self.freeze_blocks))
                x = _leaky_conv(x, filters=512, kernel=1, strides=1,
                                name='b5_conv2', trainable=not(5 in self.freeze_blocks))
                x = _leaky_conv(x, filters=1024, kernel=3, strides=1,
                                name='b5_conv3', trainable=not(5 in self.freeze_blocks))
                x = _leaky_conv(x, filters=512, kernel=1, strides=1,
                                name='b5_conv4', trainable=not(5 in self.freeze_blocks))
                x = _leaky_conv(x, filters=1024, kernel=3, strides=1,
                                name='b5_conv5', trainable=not(5 in self.freeze_blocks))
            else:
                raise NotImplementedError('''
                          A DarkNet with nlayers=%d is not implemented.''' % self.nlayers)
        x = TimeDistributed(AveragePooling2D(pool_size=(self.roi_pool_size, self.roi_pool_size),
                            strides=(1, 1), padding='valid',
                            data_format='channels_first', name='avg_pool'))(x)
        # During export, in order to map this node to UFF Flatten Op, we have to
        # make sure this layer name has 'flatten' in it. Otherwise, it cannot be
        # converted to UFF Flatten Op during pb to UFF conversion.
        x = TimeDistributed(Flatten(name='classifier_flatten'), name='time_distributed_flatten')(x)
        return x
