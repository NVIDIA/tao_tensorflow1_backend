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
"""Exponential Decay Learning Rate Schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.blocks.learning_rate_schedules.learning_rate_schedule import (
    LearningRateSchedule,
)
from nvidia_tao_tf1.core.coreobject import save_args


class ExponentialDecayLearningRateSchedule(LearningRateSchedule):
    """ExponentialDecayLearningRateSchedule class."""

    @save_args
    def __init__(
        self,
        learning_rate,
        decay_steps,
        decay_rate,
        staircase=False,
        min_learning_rate=0.0,
        **kwargs
    ):
        """__init__ method.

        decayed_learning_rate = learning_rate *
                                decay_rate ^ (global_step / decay_steps)

        Args:
            learning_rate (float): initial learning rate to be used.
            decay_steps (int): number of steps before next decay.
            decay_rate (float): the decay rate.
            staircase (bool): whether to apply decay in a discrete staircase as opposed to
             continuous fashion.
            min_learning_rate (float): the minimum learning rate to be used.
        """
        super(ExponentialDecayLearningRateSchedule, self).__init__(**kwargs)
        self._learning_rate = learning_rate
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._staircase = staircase
        self._min_learning_rate = min_learning_rate

    def get_tensor(self):
        """Get the learning rate tensor.

        Returns:
            scalar tensor (tf.float32)
        """
        global_step = tf.compat.v1.train.get_global_step()

        lr = tf.compat.v1.train.exponential_decay(
            self._learning_rate,
            global_step,
            self._decay_steps,
            self._decay_rate,
            self._staircase,
        )

        learning_rate_tensor = tf.maximum(lr, self._min_learning_rate)

        tf.compat.v1.summary.scalar(name="learning_rate", tensor=learning_rate_tensor)

        return learning_rate_tensor
