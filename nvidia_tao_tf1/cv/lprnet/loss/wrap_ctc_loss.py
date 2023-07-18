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

"""TLT LPRNet CTC loss."""

import tensorflow as tf


class WrapCTCLoss:
    """Wrap tf ctc loss into keras loss style."""

    def __init__(self, max_label_length):
        """Initialize CTC loss's parameter."""

        self.max_label_length = max_label_length

    def compute_loss(self, y_true, y_pred):
        """Compute CTC loss."""

        label_input = tf.reshape(y_true[:, 0:self.max_label_length],
                                 (-1, self.max_label_length))
        ctc_input_length = tf.reshape(y_true[:, -2],
                                      (-1, 1))
        label_length = tf.reshape(y_true[:, -1],
                                  (-1, 1))

        ctc_loss = tf.keras.backend.ctc_batch_cost(label_input,
                                                   y_pred,
                                                   ctc_input_length,
                                                   label_length)

        return tf.reduce_mean(ctc_loss)
