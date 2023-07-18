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
"""Mean Squared Error Loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.losses.loss import Loss


class MseLoss(Loss):
    """Simple Mean Squared Error Loss."""

    def __call__(self, labels, predictions):
        """__call__ method.

        Calculate the loss.

        Args:
            labels (tensor): labels.
            predictions (tensor): predictions.

        Returns:
            loss (tensor).
        """
        loss = tf.compat.v1.losses.mean_squared_error(
            labels=labels, predictions=predictions
        )
        tf.compat.v1.summary.scalar(name="mse_loss", tensor=loss)
        return loss
