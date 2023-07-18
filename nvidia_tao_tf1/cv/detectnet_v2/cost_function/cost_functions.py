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

"""Cost functions used by gridbox."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

EPSILON = 1e-05
GT_BBOX_AREA_CRITERION = 0.001


def weighted_binary_cross_entropy_cost(target, pred, weight, loss_mask):
    """Elementwise weighted BCE cost."""
    BCE = -(target * tf.log(pred + EPSILON) + (1.0 - target) * tf.log(1.0 - pred + EPSILON))
    weight_vec_for_one = weight * tf.ones_like(target)
    weight_vec_for_zero = (1.0 - weight) * tf.ones_like(target)
    weights_tensor = tf.where(target > 0.5, weight_vec_for_one, weight_vec_for_zero)
    return tf.multiply(loss_mask, weights_tensor * BCE)


def weighted_L1_cost(target, pred, weight, loss_mask):
    """Elementwise weighted L1 cost."""
    weight = tf.ones_like(target) * weight
    dist = tf.abs(pred - target)
    return tf.multiply(loss_mask, tf.multiply(weight, dist))


def weighted_circular_L1_cost(target, pred, weight, loss_mask):
    """Element-wise circular L1 loss.

    <pred> and <target> are expected to produce values in ]-1; 1[ range, as well as represent
    functions with a period of 2.0, for this loss to make any sense.
    Under those two assumptions, the loss l is defined as:
        l = min(2 - |target| - |pred|, |target - pred|)

    Args:
        target (tf.Tensor): Ground truth tensor.
        pred (tf.Tensor): Prediction tensor.
        weight (tf.Tensor): Element-wise weight by which to multiply the cost.
        loss_mask (tf.Tensor): Element-wise loss mask by which to multiply the cost.

    Returns:
        circular_L1_cost (tf.Tensor): Element-wise loss representing l in the above formula.
    """
    weight = tf.ones_like(target) * weight

    abs_pred = tf.abs(pred)
    abs_target = tf.abs(target)

    circular_L1_cost = tf.minimum(2.0 - abs_pred - abs_target, tf.abs(pred - target))

    # Apply weight and loss_mask.
    circular_L1_cost = tf.multiply(loss_mask, tf.multiply(weight, circular_L1_cost))

    return circular_L1_cost


def weighted_GIOU_cost(abs_gt, abs_pred, weight, loss_mask):
    """Element-wise GIOU cost without zero-area bboxes of ground truth.

    Args:
        abs_gt (tf.Tensor): Ground truth tensors of absolute coordinates in input image space.
        abs_pred (tf.Tensor): Prediction tensors of absolute coordinates in input image space.
        weight (tf.Tensor): Element-wise weight by which to multiply the cost.
        loss_mask (tf.Tensor): Element-wise loss mask by which to multiply the cost.

    Returns:
        giou_cost_with_removed_zero_gt (tf.Tensor): Element-wise GIOU cost of shape [B, 4, H, W].
    """
    abs_pred = tf.unstack(abs_pred, axis=1)
    abs_gt = tf.unstack(abs_gt, axis=1)
    coords_left_pred, coords_top_pred, coords_right_pred, coords_bottom_pred = abs_pred
    coords_left_gt, coords_top_gt, coords_right_gt, coords_bottom_gt = abs_gt

    # Calculate element-wise bbox IOU.
    x1 = tf.maximum(coords_left_pred, coords_left_gt)
    y1 = tf.maximum(coords_top_pred, coords_top_gt)
    x2 = tf.minimum(coords_right_pred, coords_right_gt)
    y2 = tf.minimum(coords_bottom_pred, coords_bottom_gt)
    w = tf.maximum(x2 - x1, 0.0)
    h = tf.maximum(y2 - y1, 0.0)

    intersection = tf.multiply(w, h)
    area_pred = tf.multiply(coords_right_pred - coords_left_pred,
                            coords_bottom_pred - coords_top_pred)
    area_gt = tf.multiply(coords_right_gt - coords_left_gt,
                          coords_bottom_gt - coords_top_gt)
    union = area_pred + area_gt - intersection
    iou = tf.divide(intersection, union + EPSILON)

    # Calculate element-wise GIOU-cost.
    x1c = tf.minimum(coords_left_pred, coords_left_gt)
    y1c = tf.minimum(coords_top_pred, coords_top_gt)
    x2c = tf.maximum(coords_right_pred, coords_right_gt)
    y2c = tf.maximum(coords_bottom_pred, coords_bottom_gt)
    area_all = tf.multiply(x2c - x1c, y2c - y1c)

    giou = iou - tf.divide(area_all - union, area_all + EPSILON)
    giou_cost = 1.0 - giou

    # Remove losses related with zero-area ground truth bboxes.
    zero_tmp = tf.zeros_like(area_gt)
    giou_cost_with_removed_zero_gt = \
        tf.where(tf.greater(tf.abs(area_gt), GT_BBOX_AREA_CRITERION),
                 giou_cost, zero_tmp)

    # Expand GIOU_cost to the certain shape [B, 4, H, W].
    giou_cost_with_removed_zero_gt = tf.expand_dims(
        giou_cost_with_removed_zero_gt, 1)
    giou_cost_with_removed_zero_gt = tf.tile(
        giou_cost_with_removed_zero_gt, [1, 4, 1, 1])

    # Multiply weights on GIOU_cost.
    giou_cost_with_removed_zero_gt = tf.multiply(
        giou_cost_with_removed_zero_gt, weight)
    giou_cost_with_removed_zero_gt = tf.multiply(
        giou_cost_with_removed_zero_gt, loss_mask)
    return giou_cost_with_removed_zero_gt
