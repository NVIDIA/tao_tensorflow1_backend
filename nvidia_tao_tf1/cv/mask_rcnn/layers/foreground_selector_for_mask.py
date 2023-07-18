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
from nvidia_tao_tf1.cv.mask_rcnn.ops import training_ops


class ForegroundSelectorForMask(keras.layers.Layer):
    '''A custom Keras layer to select foreground objects for mask training.'''

    def __init__(self,
                 max_num_fg=256,
                 **kwargs):
        '''Init function.'''
        self.max_num_fg = max_num_fg
        super(ForegroundSelectorForMask, self).__init__(**kwargs)

    def call(self, inputs):
        """Selects the fore ground objects for mask branch during training."""
        class_targets, box_targets, boxes, proposal_to_label_map = inputs
        return training_ops.select_fg_for_masks(
            class_targets=class_targets,
            box_targets=box_targets,
            boxes=boxes,
            proposal_to_label_map=proposal_to_label_map,
            max_num_fg=self.max_num_fg)

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'max_num_fg': self.max_num_fg,
        }
        base_config = super(ForegroundSelectorForMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
