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

"""Automatic class weighting callback."""

from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import tensorflow as tf


class AutoClassWeighting(Callback):
    """
    Automatically update the classes weighting for anchor-based detection algorithms.

    Currently, only merged with YOLOV4.
    Detailed demonstration could be found at:
    https://confluence.nvidia.com/display/~tylerz/Auto+class+weighting+-
        -+trainning+anchor-based+object+detection+model+on+unbalanced+dataset
    """

    def __init__(self, train_dataset, loss_ops, alpha=0.9, interval=10):
        """Init function."""
        super(AutoClassWeighting, self).__init__()
        self.train_dataset = train_dataset
        self.classes = train_dataset.classes
        self.loss_ops = loss_ops
        self.alpha = alpha
        self.pred = tf.Variable(0., validate_shape=False,
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])
        self.label = tf.Variable(0., validate_shape=False,
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES])
        self.loss_per_class = np.zeros((len(self.classes)), dtype=np.float32)
        self.anchor_per_class = np.zeros((len(self.classes)), dtype=np.float32)
        self.interval = interval

    def on_batch_end(self, batch, logs=None):
        """Compute per anchor loss in every batch."""
        # compute the per class loc_loss cls_loss (average by num of assigned anchors)
        # input_data
        pred = K.get_value(self.pred)
        encoded_lab = K.get_value(self.label)
        # y_pred_encoded = self.model.predict(batch_x)
        batch_loss = K.get_session().run(self.loss_ops[2],
                                         feed_dict={self.loss_ops[0]: encoded_lab,
                                                    self.loss_ops[1]: pred})
        # loc_loss: [#batch, #anchor]; cls_loss: [#batch, #anchor]
        loc_loss, cls_loss = batch_loss
        # convert the one-hot vector to index
        idx_map = np.tile(np.arange(len(self.classes)),
                          [loc_loss.shape[0], loc_loss.shape[1], 1])
        one_hot_vectors = encoded_lab[:, :, 6:-1]
        neg_map = np.full(one_hot_vectors.shape, -1)
        cls_idx = np.max(np.where(one_hot_vectors == 1, idx_map, neg_map), axis=-1)
        # compute the loss per class
        for idx in range(len(self.classes)):
            cur_loc_loss = float(0)
            cur_cls_loss = float(0)
            cur_loc_loss = loc_loss[cls_idx == idx]
            if len(cur_loc_loss) <= 0:
                continue
            num_anchor = cur_loc_loss.shape[0]
            cur_loc_loss = np.sum(cur_loc_loss)
            cur_cls_loss = np.sum(cls_loss[cls_idx == idx])
            self.loss_per_class[idx] += cur_loc_loss + cur_cls_loss
            self.anchor_per_class[idx] += num_anchor

    def on_epoch_end(self, epoch, logs=None):
        """Compute per anchor per class loss and classes weighting at the end of epoch."""
        # compute the per class weights (reciprocal of each loss * maximum loss)
        old_weights = np.array(self.train_dataset.encode_fn.class_weights)
        # print(f"old_weights: {old_weights}")
        self.loss_per_class = self.loss_per_class / old_weights
        self.loss_per_class = self.loss_per_class / self.anchor_per_class
        # max_loss = np.max(self.loss_per_class)
        min_loss = np.min(self.loss_per_class)
        new_wts = []
        for idx in range(len(self.classes)):
            new_w = float(self.loss_per_class[idx]) / min_loss
            # print(f"{self.classes[idx]}:{self.loss_per_class[idx]}")
            # print(f"{self.classes[idx]}_anchor:{self.anchor_per_class[idx]}")
            # logs[f"{self.classes[idx]}_loss"] = self.loss_per_class[idx]
            # logs[f"{self.classes[idx]}_anchor"] = self.anchor_per_class[idx]
            new_wts.append(new_w)
        # Make the first epoch count !
        if epoch != 0:
            new_wts = old_weights * self.alpha + np.array(new_wts) * (1 - self.alpha)
        # print(f"new_weights: {new_wts}")
        if epoch % self.interval == 0:
            self.train_dataset.encode_fn.update_class_weights(new_wts)
        self.loss_per_class = np.zeros((len(self.classes)), dtype=np.float32)
        self.anchor_per_class = np.zeros((len(self.classes)), dtype=np.float32)
