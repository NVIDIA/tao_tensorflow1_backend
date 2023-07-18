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
from nvidia_tao_tf1.cv.mask_rcnn.ops import postprocess_ops


class GPUDetections(keras.layers.Layer):
    '''A Keras layer to generate final prediction output.'''

    def __init__(self,
                 pre_nms_num_detections=1000,
                 post_nms_num_detections=100,
                 nms_threshold=0.5,
                 bbox_reg_weights=(10., 10., 5., 5.),
                 **kwargs):
        '''Init function.'''
        self.pre_nms_num_detections = pre_nms_num_detections
        self.post_nms_num_detections = post_nms_num_detections
        self.nms_threshold = nms_threshold
        self.bbox_reg_weights = bbox_reg_weights
        super(GPUDetections, self).__init__(**kwargs)

    def call(self, inputs):
        """Generate the final detections given the model outputs (GPU version)."""
        class_outputs, box_outputs, anchor_boxes, image_info = inputs
        return postprocess_ops.generate_detections_gpu(
            class_outputs=class_outputs,
            box_outputs=box_outputs,
            anchor_boxes=anchor_boxes,
            image_info=image_info,
            pre_nms_num_detections=self.pre_nms_num_detections,
            post_nms_num_detections=self.post_nms_num_detections,
            nms_threshold=self.nms_threshold,
            bbox_reg_weights=self.bbox_reg_weights
        )

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'pre_nms_num_detections': self.pre_nms_num_detections,
            'post_nms_num_detections': self.post_nms_num_detections,
            'nms_threshold': self.nms_threshold,
            'bbox_reg_weights': self.bbox_reg_weights,
        }
        base_config = super(GPUDetections, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
