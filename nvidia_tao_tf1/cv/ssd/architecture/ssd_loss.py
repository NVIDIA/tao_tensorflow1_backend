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

"""SSD Loss for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SSDLoss:
    '''The SSD loss, see https://arxiv.org/abs/1512.02325.'''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Initialization of SSD Loss.

        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
        '''

        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
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
        l1_loss = tf.where(tf.less(absolute_loss, 1.0),
                           square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.

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
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets,
                8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`,
                i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets,
                8 arbitrary entries]`.

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
        # Output dtype: tf.int32, note that `n_boxes` in this context denotes
        # the total number of boxes per image, not the number of boxes per cell.
        n_boxes = tf.shape(y_pred)[1]

        # 1: Compute the losses for class and box predictions for every box.

        # Output shape: (batch_size, n_boxes)
        classification_loss = tf.to_float(self.log_loss(y_true[:, :, :-12],
                                                        y_pred[:, :, :-12]))
        # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:, :, -12:-8],
                                                            y_pred[:, :, -12:-8]))

        # 2: Compute the classification losses for the positive and negative targets.

        # Create masks for the positive and negative ground truth classes.
        # Tensor of shape (batch_size, n_boxes)
        negatives = y_true[:, :, 0]
        # Tensor of shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes).
        # Tensor of shape (batch_size,)
        pos_class_loss = tf.reduce_sum(
            classification_loss * positives, axis=-1)

        # Compute the classification loss for the negative default boxes (if there are any).

        # First, compute the classification loss for all negative boxes.
        # Tensor of shape (batch_size, n_boxes)
        neg_class_loss_all = classification_loss * negatives
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32)

        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive),
                                                self.n_neg_min), n_neg_losses)

        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss.

        def f2():
            # Tensor of shape (batch_size * n_boxes,)
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
            _, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                     k=n_negative_keep,
                                     sorted=False)
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(
                                               indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D))
            negatives_keep = tf.to_float(tf.reshape(
                negatives_keep, [batch_size, n_boxes]))
            neg_class_loss = tf.reduce_sum(
                classification_loss * negatives_keep, axis=-1)
            return neg_class_loss

        neg_class_loss = tf.cond(
            tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        # Tensor of shape (batch_size,)
        class_loss = pos_class_loss + neg_class_loss

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

        # 4: Compute the total loss.

        total_loss = (class_loss + self.alpha * loc_loss) / \
            tf.maximum(1.0, n_positive)
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
