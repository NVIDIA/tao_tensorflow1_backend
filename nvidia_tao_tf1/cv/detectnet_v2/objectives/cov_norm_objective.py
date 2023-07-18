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

"""Normalized coverage objective."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.cv.detectnet_v2.objectives.base_objective import BaseObjective


class CovNormObjective(BaseObjective):
    """Normalized coverage objective (not learnable).

    CovNormObjective implements the normalized coverage objective-specific part of the
    following functionality:
    - Rasterization (labels -> tensors)
    """

    def __init__(self, input_layer_name, output_height, output_width):
        """Constructor for normalized coverage objective.

        Args:
            input_layer_name (string): Name of the input layer of the Objective head.
                If None the last layer of the model will be used.
            output_height, output_width: Shape of the DNN output tensor.
        """
        super(CovNormObjective, self).__init__(
            input_layer_name, output_height, output_width)
        self.name = 'cov_norm'
        self.num_channels = 1
        self.learnable = False
        # TODO(pjanis): Check the impact of this one, Rumpy uses passthrough here.
        # Intuitively; should we down-weight objective costs at edges of coverage blobs?
        self.gradient_flag = tao_core.processors.BboxRasterizer.GRADIENT_MODE_MULTIPLY_BY_COVERAGE

    def target_gradient(self, ground_truth_label):
        """Gradient for rasterizing the normalized coverage tensor.

        This will make the rasterizer rasterize a constant value equal to the
        inverse of the bounding box area for each target bounding box. The
        constant value is further multiplied by the coverage value (according
        to self.gradient_flag).

        Args:
            ground_truth_label: dictionary of label attributes. Uses the attribute
                on key 'target/inv_bbox_area'

        Returns:
            The gradients' coefficients.
        """
        inv_bbox_area = ground_truth_label['target/inv_bbox_area']
        num_boxes = tf.size(inv_bbox_area)
        zero = tf.zeros(shape=[num_boxes])
        gradient = tf.transpose([[zero, zero, inv_bbox_area]], (2, 0, 1))

        return gradient
