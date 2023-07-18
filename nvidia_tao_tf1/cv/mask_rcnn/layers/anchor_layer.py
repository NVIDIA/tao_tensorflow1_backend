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
from nvidia_tao_tf1.cv.mask_rcnn.models import anchors


class AnchorLayer(keras.layers.Layer):
    '''A custom Keras layer to generate anchors.'''

    def __init__(self,
                 min_level,
                 max_level,
                 num_scales,
                 aspect_ratios,
                 anchor_scale,
                 **kwargs):
        '''Init function.'''
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        super(AnchorLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """Get unpacked multiscale Mask-RCNN anchors."""

        _, _, image_height, image_width = inputs.get_shape().as_list()
        all_anchors = anchors.Anchors(
            self.min_level, self.max_level,
            self.num_scales, self.aspect_ratios,
            self.anchor_scale,
            (image_height, image_width))
        return all_anchors.get_unpacked_boxes()

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'min_level': self.min_level,
            'max_level': self.max_level,
            'num_scales': self.num_scales,
            'aspect_ratios': self.aspect_ratios,
            'anchor_scale': self.anchor_scale
        }
        base_config = super(AnchorLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
