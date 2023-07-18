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

"""MaskRCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from nvidia_tao_tf1.cv.mask_rcnn.ops import training_ops


class MaskTargetsLayer(keras.layers.Layer):
    '''A custom Keras layer for GT masks.'''

    def __init__(self,
                 mrcnn_resolution=28,
                 **kwargs):
        '''Init function.'''
        self.mrcnn_resolution = mrcnn_resolution
        super(MaskTargetsLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """Generate mask for each ROI."""
        fg_boxes, fg_proposal_to_label_map, fg_box_targets, mask_gt_labels = inputs
        return training_ops.get_mask_targets(
            fg_boxes, fg_proposal_to_label_map,
            fg_box_targets, mask_gt_labels,
            output_size=self.mrcnn_resolution)

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'mrcnn_resolution': self.mrcnn_resolution,
        }
        base_config = super(MaskTargetsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
