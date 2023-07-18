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
from nvidia_tao_tf1.cv.mask_rcnn.ops import spatial_transform_ops


class MultilevelCropResize(keras.layers.Layer):
    '''A Keras custome layer to generate ROI features.'''

    def __init__(self,
                 output_size=7,
                 is_gpu_inference=False,
                 **kwargs):
        '''Init function.'''
        self.output_size = output_size
        self.is_gpu_inference = is_gpu_inference
        super(MultilevelCropResize, self).__init__(**kwargs)

    def call(self, inputs):
        """Crop and resize on multilevel feature pyramid.

        Generate the (output_size, output_size) set of pixels for each input box
        by first locating the box into the correct feature level, and then cropping
        and resizing it using the correspoding feature map of that level.
        """
        # features, boxes = inputs
        features = inputs[0:5]
        boxes = inputs[-1]

        # turn into dict
        k_order = list(range(2, 7))
        features = dict(zip(k_order, features))

        return spatial_transform_ops.multilevel_crop_and_resize(
            features=features,
            boxes=boxes,
            output_size=self.output_size,
            is_gpu_inference=self.is_gpu_inference
        )

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'output_size': self.output_size,
            'is_gpu_inference': self.is_gpu_inference,
        }
        base_config = super(MultilevelCropResize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
