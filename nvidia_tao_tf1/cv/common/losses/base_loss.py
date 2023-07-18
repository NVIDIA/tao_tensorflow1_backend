# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

'''Base loss for IVA models.'''

from abc import ABC, abstractmethod
import tensorflow as tf


class BaseLoss(ABC):
    '''
    IVA Base losses.

    All model losses (if needs customization) should be inherited from this class.

    Child class must implement: compute_loss(self, y_true, y_false). And this should
        be passed into model.compile() as loss.
    '''

    def bce_loss(self, y_true, y_pred, smoothing=0.0):
        '''
        Compute the bce loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.
            smoothing (float): y_true = y_true * (1-smoothing) + smoothing / 2.0. Smoothing=0 is
                same as old bce_loss.

        Returns:
            The bce loss
        '''
        # Compute the log loss
        y_true = y_true * (1.0 - smoothing) + smoothing / 2.0

        y_pred = tf.sigmoid(y_pred)

        bce_loss = -(y_true * tf.log(tf.maximum(y_pred, 1e-18)) +
                     (1.0-y_true) * tf.log(tf.maximum(1.0-y_pred, 1e-18)))

        return tf.reduce_sum(bce_loss, axis=-1)

    def bce_focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0, smoothing=0.0):
        '''
        Compute the bce focal loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.
            alpha: alpha of focal loss
            gamma: gamma of focal loss
            smoothing (float): y_true = y_true * (1-smoothing) + smoothing / 2.0.

        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        y_true = y_true * (1.0 - smoothing) + smoothing / 2.0
        y_pred = tf.sigmoid(y_pred)
        # Compute the log loss
        bce_loss = -(y_true * tf.log(tf.maximum(y_pred, 1e-18)) +
                     (1.0-y_true) * tf.log(tf.maximum(1.0-y_pred, 1e-18)))
        p_ = (y_true * y_pred) + (1.0-y_true) * (1.0-y_pred)
        modulating_factor = tf.pow(1.0 - p_, gamma)
        weight_factor = (y_true * alpha + (1.0 - y_true) * (1.0-alpha))
        focal_loss = modulating_factor * weight_factor * bce_loss

        return tf.reduce_sum(focal_loss, axis=-1)

    def L2_loss(self, y_true, y_pred):
        '''Compute L2 loss.'''
        square_loss = 0.5 * (y_true - y_pred)**2
        return tf.reduce_sum(square_loss, axis=-1)

    @abstractmethod
    def compute_loss(self, y_true, y_pred):
        '''compute_loss to be implemented in child class.'''
        raise NotImplementedError("compute_loss not implemented!")
