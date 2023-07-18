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
"""Binary Crossentropy Error Loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.losses.loss import Loss
from nvidia_tao_tf1.core.coreobject import save_args


class BinaryCrossentropyLoss(Loss):
    """Simple Binary Crossentropy Error Loss."""

    @save_args
    def __init__(self, output_summary=True, reduce_mean=True):
        """__init__ method.

        Args:
            output_summary (bool): Flag to toggle tf summary output (True to write).
            reduce_mean (bool): Flag to apply the mean of tensor elements (True to apply).
        """
        self.__name__ = "binary_crossentropy"
        self._output_summary = output_summary
        self._reduce_mean = reduce_mean

    def __call__(self, labels, predictions):
        """__call__ method.

        Calculate the loss.

        Args:
            labels (tensor): Labels.
            predictions (tensor): Predictions.

        Returns:
            loss (tensor).
        """
        losses = tf.keras.losses.binary_crossentropy(labels, predictions)
        if self._reduce_mean:
            loss = tf.reduce_mean(input_tensor=losses)
        else:
            loss = losses
        if self._output_summary:
            tf.compat.v1.summary.scalar(
                name="binary_crossentropy_loss",
                tensor=tf.reduce_mean(input_tensor=losses),
            )
        return loss
