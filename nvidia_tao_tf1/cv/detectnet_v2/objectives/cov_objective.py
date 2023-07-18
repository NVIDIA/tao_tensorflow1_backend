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

"""Coverage objective."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_functions import (
    weighted_binary_cross_entropy_cost
)
from nvidia_tao_tf1.cv.detectnet_v2.objectives.base_objective import BaseObjective
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer


class CovObjective(BaseObjective):
    """Coverage objective.

    CovObjective implements the coverage objective-specific parts of the
    following functionalities:
    - Rasterization (labels -> tensors)
    - Cost function
    - Objective-specific visualization
    """

    def __init__(self, input_layer_name, output_height, output_width):
        """Constructor for the coverage objective.

        Args:
            input_layer_name (string): Name of the input layer of the Objective head.
                If None the last layer of the model will be used.
            output_height, output_width: Shape of the DNN output tensor.
        """
        super(CovObjective, self).__init__(
            input_layer_name, output_height, output_width)
        self.name = 'cov'
        self.num_channels = 1
        self.activation = 'sigmoid'
        self.gradient_flag = tao_core.processors.BboxRasterizer.GRADIENT_MODE_MULTIPLY_BY_COVERAGE

    def cost(self, y_true, y_pred, target_class, loss_mask=None):
        """Coverage cost function.

        Args:
            y_true: GT tensor dictionary. Contains key 'cov'
            y_pred: Prediction tensor dictionary. Contains key 'cov'
            target_class: (TargetClass) for which to create the cost
            loss_mask: (tf.Tensor) Loss mask to multiply the cost by.

        Returns:
            cost: TF scalar.
        """
        assert 'cov' in y_true
        assert 'cov' in y_pred
        cov_target = y_true['cov']
        cov_pred = y_pred['cov']
        cov_weight = target_class.coverage_foreground_weight
        cov_loss_mask = 1.0 if loss_mask is None else loss_mask
        cov_cost = weighted_binary_cross_entropy_cost(cov_target, cov_pred,
                                                      cov_weight, cov_loss_mask)

        mean_cost = tf.reduce_mean(cov_cost)

        # Visualize cost, target, and prediction.
        if Visualizer.enabled:
            # Visualize mean losses (scalar) always.
            tf.summary.scalar('mean_cost_%s_cov' %
                              target_class.name, mean_cost)

            # Visualize tensors, if it is enabled in the spec. Use absolute
            # scale to avoid Tensorflow automatic scaling. This facilitates
            # comparing images as the network trains.
            value_range = [0.0, 1.0]
            Visualizer.image('%s_cov_cost' % target_class.name, cov_cost,
                             value_range=value_range,
                             collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])
            Visualizer.image('%s_cov_gt' % target_class.name, cov_target,
                             value_range=value_range,
                             collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])
            Visualizer.image('%s_cov_norm' % target_class.name, y_true['cov_norm'],
                             value_range=value_range,
                             collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])
            Visualizer.image('%s_cov_pred' % target_class.name, cov_pred,
                             value_range=value_range,
                             collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])
            if isinstance(loss_mask, tf.Tensor):
                Visualizer.image('%s_cov_loss_mask' % target_class.name, cov_loss_mask,
                                 collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])

        return mean_cost

    def target_gradient(self, ground_truth_label):
        """Coverage gradient.

        This will make the rasterizer rasterize a constant value 1.0 for each
        target bounding box. The constant value is further multiplied by the
        coverage value (according to self.gradient_flag).

        Args:
            ground_truth_label: dictionary of label attributes. Uses the attribute
                on key 'target/inv_bbox_area' for getting the number of boxes.

        Returns:
            gradient (tf.Tensor): The shape is (num_bboxes, 1, 3).
        """
        num_boxes = tf.size(ground_truth_label['target/inv_bbox_area'])
        zero = tf.zeros(shape=[num_boxes])
        one = tf.ones(shape=[num_boxes])
        gradient = tf.transpose([[zero, zero, one]], (2, 0, 1))

        return gradient
