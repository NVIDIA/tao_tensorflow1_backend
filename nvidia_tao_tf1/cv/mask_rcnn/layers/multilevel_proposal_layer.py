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

"""MaskRCNN custom layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from nvidia_tao_tf1.cv.mask_rcnn.ops import roi_ops


class MultilevelProposal(keras.layers.Layer):
    '''A custom Keras layer to generate RoIs.'''

    def __init__(self,
                 rpn_pre_nms_topn=2000,
                 rpn_post_nms_topn=1000,
                 rpn_nms_threshold=0.7,
                 rpn_min_size=0,
                 bbox_reg_weights=None,
                 use_batched_nms=True,
                 **kwargs):
        '''Init function.'''
        self.rpn_pre_nms_topn = rpn_pre_nms_topn
        self.rpn_post_nms_topn = rpn_post_nms_topn
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_min_size = rpn_min_size
        self.bbox_reg_weights = bbox_reg_weights
        self.use_batched_nms = use_batched_nms
        super(MultilevelProposal, self).__init__(**kwargs)

    def call(self, inputs):
        """Proposes RoIs given a group of candidates from different FPN levels."""
        scores_outputs = inputs[0:5]
        box_outputs = inputs[5:10]
        anchor_boxes = inputs[10:15]
        image_info = inputs[-1]

        # turn into dict
        k_order = list(range(2, 7))
        scores_outputs = dict(zip(k_order, scores_outputs))
        box_outputs = dict(zip(k_order, box_outputs))
        anchor_boxes = dict(zip(k_order, anchor_boxes))

        if isinstance(self.rpn_pre_nms_topn, tuple):
            self.rpn_pre_nms_topn = self.rpn_pre_nms_topn[0]
        if isinstance(self.rpn_post_nms_topn, tuple):
            self.rpn_post_nms_topn = self.rpn_post_nms_topn[0]
        return roi_ops.multilevel_propose_rois(
            scores_outputs=scores_outputs,
            box_outputs=box_outputs,
            anchor_boxes=anchor_boxes,
            image_info=image_info,
            rpn_pre_nms_topn=self.rpn_pre_nms_topn,
            rpn_post_nms_topn=self.rpn_post_nms_topn,
            rpn_nms_threshold=self.rpn_nms_threshold,
            rpn_min_size=self.rpn_min_size,
            bbox_reg_weights=self.bbox_reg_weights,
            use_batched_nms=self.use_batched_nms
        )

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'rpn_pre_nms_topn': self.rpn_pre_nms_topn,
            'rpn_post_nms_topn': self.rpn_post_nms_topn,
            'rpn_nms_threshold': self.rpn_nms_threshold,
            'rpn_min_size': self.rpn_min_size,
            'bbox_reg_weights': self.bbox_reg_weights,
            'use_batched_nms': self.use_batched_nms,
        }
        base_config = super(MultilevelProposal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
