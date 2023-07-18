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

"""Focal Loss for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def smooth_L1_loss(y_true, y_pred):
    '''
    Compute smooth L1 loss, see references.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
            contains the ground truth bounding box coordinates, where the last dimension
            contains `(d_cx, d_cy, log_w, log_h)`.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box coordinates.

    Returns:
        The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).

    References:
        https://arxiv.org/abs/1504.08083
    '''
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


def bce_focal_loss(y_true, y_pred, alpha, gamma):
    '''
    Compute the bce focal loss.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape (batch_size, #boxes, #classes)
            and contains the ground truth bounding box categories.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box categories.

    Returns:
        The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    '''
    # Compute the log loss
    bce_loss = -(y_true * tf.log(tf.maximum(y_pred, 1e-18)) +
                 (1.0-y_true) * tf.log(tf.maximum(1.0-y_pred, 1e-18)))
    p_ = (y_true * y_pred) + (1.0-y_true) * (1.0-y_pred)
    modulating_factor = tf.pow(1.0 - p_, gamma)
    weight_factor = (y_true * alpha + (1.0 - y_true) * (1.0-alpha))
    focal_loss = modulating_factor * weight_factor * bce_loss

    return tf.reduce_sum(focal_loss, axis=-1)


class FocalLoss:
    '''
    Focal Loss class.

    Focal loss, see https://arxiv.org/abs/1708.02002
    '''

    def __init__(self,
                 loc_loss_weight=1.0,
                 alpha=0.25,
                 gamma=2.0):
        '''Loss init function.'''

        self.loc_loss_weight = loc_loss_weight
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `1 + #classes + 12` and contain
                `[class_weights, classes one-hot encoded, 4 gt box coordinate offsets,
                8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box offsets, 8 arbitrary entries]`.

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32

        # 1: Compute the losses for class and box predictions for every box.

        class_weights = y_true[:, :, 0]

        classification_loss = tf.dtypes.cast(bce_focal_loss(y_true[:, :, 2:-12],
                                             y_pred[:, :, 1:-12],
                                             self.alpha, self.gamma),
                                             tf.float32)

        localization_loss = tf.dtypes.cast(smooth_L1_loss(y_true[:, :, -12:-8],
                                           y_pred[:, :, -12:-8]), tf.float32)
        # 2: Compute the classification losses for the positive and negative targets.

        # Create masks for the positive and negative ground truth classes.
        # Tensor of shape (batch_size, n_boxes)
        positives = tf.dtypes.cast(tf.reduce_max(y_true[:, :, 2:-12], axis=-1), tf.float32)
        non_neutral = tf.dtypes.cast(tf.reduce_max(y_true[:, :, 1:-12], axis=-1), tf.float32)
        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
        n_positive = tf.reduce_sum(positives)

        class_loss = tf.reduce_sum(classification_loss * non_neutral * class_weights, axis=-1)
        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes
        #    (obviously: there are no ground truth boxes they would correspond to).
        # Shape (batch_size,)
        loc_loss = tf.reduce_sum(localization_loss * positives * class_weights, axis=-1)

        # 4: Compute the total loss.

        # In case `n_positive == 0`
        total_loss = (class_loss + self.loc_loss_weight *
                      loc_loss) / tf.maximum(1.0, n_positive)

        # Keras has the bad habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes
        # in the batch (by which we're dividing in the line above), not the batch size. So in
        # order to revert Keras' averaging over the batch size, we'll have to multiply by it.
        total_loss = total_loss * tf.dtypes.cast(batch_size, tf.float32)

        return total_loss
