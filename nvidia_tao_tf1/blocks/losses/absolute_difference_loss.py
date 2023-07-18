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
"""Absolute Difference Loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.losses.loss import Loss


class AbsoluteDifferenceLoss(Loss):
    """Simple Absolute Difference Loss."""

    def __call__(self, labels, predictions, weights=1.0):
        """__call__ method.

        Calculate the loss.

        Args:
            labels (tensor): labels.
            predictions (tensor): predictions.
            weights (tensor): weights. Default: 1.0.

        Returns:
            loss (tensor).
        """
        loss = tf.compat.v1.losses.absolute_difference(labels, predictions, weights)
        tf.compat.v1.summary.scalar(name="absolute_difference_loss", tensor=loss)
        return loss
