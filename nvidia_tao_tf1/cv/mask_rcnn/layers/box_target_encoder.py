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


class BoxTargetEncoder(keras.layers.Layer):
    '''A custom Keras layer to encode box targets.'''

    def __init__(self,
                 bbox_reg_weights=None,
                 **kwargs):
        '''Init function.'''
        self.bbox_reg_weights = bbox_reg_weights
        super(BoxTargetEncoder, self).__init__(**kwargs)

    def call(self, inputs):
        """Generate box target."""
        boxes, gt_boxes, gt_labels = inputs
        return training_ops.encode_box_targets(boxes, gt_boxes, gt_labels, self.bbox_reg_weights)

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'bbox_reg_weights': self.bbox_reg_weights,
        }
        base_config = super(BoxTargetEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
