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

import tensorflow as tf
from tensorflow import keras
from nvidia_tao_tf1.cv.mask_rcnn.ops import training_ops


class ProposalAssignment(keras.layers.Layer):
    '''A custom Keras layer to assign proposals.'''

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=0.25,
                 fg_thresh=0.5,
                 bg_thresh_hi=0.5,
                 bg_thresh_lo=0.,
                 **kwargs):
        '''Init function.'''
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        super(ProposalAssignment, self).__init__(**kwargs)

    def call(self, inputs):
        """Assigns the proposals with ground truth labels and performs subsmpling."""
        boxes, gt_boxes, gt_labels = inputs
        boxes = tf.stop_gradient(boxes)
        return training_ops.proposal_label_op(
            boxes,
            gt_boxes,
            gt_labels,
            batch_size_per_im=self.batch_size_per_im,
            fg_fraction=self.fg_fraction,
            fg_thresh=self.fg_thresh,
            bg_thresh_hi=self.bg_thresh_hi,
            bg_thresh_lo=self.bg_thresh_lo)

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'batch_size_per_im': self.batch_size_per_im,
            'fg_fraction': self.fg_fraction,
            'fg_thresh': self.fg_thresh,
            'bg_thresh_hi': self.bg_thresh_hi,
            'bg_thresh_lo': self.bg_thresh_lo
        }
        base_config = super(ProposalAssignment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
