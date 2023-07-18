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
"""ProposalTarget layer for FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer
import numpy as np

from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import proposal_target_wrapper


class ProposalTarget(Layer):
    '''ProposalTarget layer constructed with TF Ops.

    In FasterRCNN, ProposalTarget layer is applied on top of Proposal layer
    to generate the target tensors for RCNN. Apart from Proposal layer, this also need
    input class groundtruth and input bbox groundtruth as inputs. In principle, it
    calculates the IoUs between the groundtruth boxes and the RoIs come from Proposal layer.
    With some specific IoU thresholds, it categorizes the RoIs as two classes: positive RoIs
    and negative RoIs. Then positive RoIs are use to train RCNN classifier and regressor.
    While negative RoIs are only used to train RCNN classifier.
    '''

    def __init__(self,
                 gt_as_roi, iou_high_thres,
                 iou_low_thres, bg_class_id,
                 roi_train_bs, roi_positive_ratio,
                 deltas_scaling, bs_per_gpu, **kwargs):
        '''Initialize the ProposalTarget layer.

        Args:
            gt_as_roi(bool): Whether or not to use groundtruth boxes as RoIs to train RCNN.
                If this is True, wil concatenate the RoIs from Proposal layer with groundtruth
                boxes and forward it to RCNN. Default is False.
            iou_high_thres(float): high IoU threshold above which we regard those RoIs as positiv
                RoIs.
            iou_low_thres(float): low IoU threshold below which we regard those RoIs as negative
                RoIs.
            bg_class_id(int): the class ID(number) for the background class. For training RCNN
                classifier, we always need a background class. But background is not a valid class
                for training RCNN regressor. By convention, valid class ID is 0, 1, 2, ..., (N-2),
                and background class ID is N-1. So background class ID is always the number of
                classes subtracted by 1.
            roi_train_bs(int): the batch size used to train RCNN for each image.
            roi_positive_ratio(float): the ratio for positive RoIs in the roi_train_bs(batch size).
                By convention, this is set to 0.25 in current implementation.
            deltas_scaling(list of float): the scaling factors applied to RCNN regressor,
                one scalar for each coordinate in (x, y, w, h). List length is 4.
            bs_per_gpu(int): the image batch size per GPU for this layer. Due to implementation,
                this layer is not agnostic to batch size.
        '''
        self.gt_as_roi = gt_as_roi
        self.iou_high_thres = iou_high_thres
        self.iou_low_thres = iou_low_thres
        self.roi_train_bs = roi_train_bs
        self.roi_positive_ratio = roi_positive_ratio
        self.deltas_scaling = np.array(deltas_scaling, dtype=np.float32)
        self.bg_class_id = bg_class_id
        self.bs_per_gpu = bs_per_gpu
        super(ProposalTarget, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        '''compute the output shape.'''
        return [(None, self.roi_train_bs, 4),
                (None, self.roi_train_bs, self.bg_class_id+1),
                (None, self.roi_train_bs, self.bg_class_id*8),
                (None, None)]

    def call(self, x, mask=None):
        '''call method.

        Args:
            x(list): the list of input tensors.
                x[0]: RoIs from Proposal layer.
                x[1]: groundtruth class IDs for each objects.
                x[2]: groundtruth bbox coordinates for each objects.
        '''

        rois = x[0]
        gt_class_ids = x[1]
        gt_boxes = x[2]
        result = proposal_target_wrapper(rois, gt_class_ids, gt_boxes, self.iou_high_thres,
                                         self.iou_low_thres, self.roi_train_bs,
                                         self.roi_positive_ratio, self.deltas_scaling,
                                         self.bg_class_id, self.bs_per_gpu, self.gt_as_roi)
        return result

    def get_config(self):
        """Get config for this layer."""
        config = {'gt_as_roi': self.gt_as_roi,
                  'iou_high_thres': self.iou_high_thres,
                  'iou_low_thres': self.iou_low_thres,
                  'roi_train_bs': self.roi_train_bs,
                  'roi_positive_ratio': self.roi_positive_ratio,
                  'deltas_scaling': self.deltas_scaling,
                  'bg_class_id': self.bg_class_id,
                  'bs_per_gpu': self.bs_per_gpu}
        base_config = super(ProposalTarget, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
