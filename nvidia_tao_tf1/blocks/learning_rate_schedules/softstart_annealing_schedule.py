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
"""Softstart Annealing Learning Rate Schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.blocks.learning_rate_schedules.learning_rate_schedule import (
    LearningRateSchedule,
)
import nvidia_tao_tf1.core.hooks.utils
from nvidia_tao_tf1.core.coreobject import save_args


class SoftstartAnnealingLearningRateSchedule(LearningRateSchedule):
    r"""SoftstartAnnealingLearningRateSchedule class.

    The learning rate schedule looks something like this:

      learning_rate
      ^
      |      ______________ <-- base_learning_rate
      |     /              |
      |    /               \
      |   /                 \
      |  /                   \
      | /                     \
      |-                       \___
      -------------------------------------------> number of global steps taken
            ^              ^
            soft_start     annealing

    (The actual ramp up and ramp down portions are exponential curves).
    """

    @save_args
    def __init__(
        self,
        base_learning_rate,
        min_learning_rate,
        soft_start,
        annealing,
        last_step=None,
        **kwargs
    ):
        """__init__ method.

        Args:
            base_learning_rate (float): Learning rate.
            min_learning_rate (float): Minimum value the learning rate will be set to.
            soft_start (float): Number between 0. and 1. indicating the fraction of `last_step`
                that will be taken before reaching the base_learning rate.
            annealing (float): Number between 0. and 1. indicating the fraction of `last_step`
                after which the learning rate ramps down from base_learning rate.
            last_step (int): Last step the schedule is made for.
        """
        super(SoftstartAnnealingLearningRateSchedule, self).__init__(**kwargs)
        self._base_learning_rate = base_learning_rate
        self._min_learning_rate = min_learning_rate
        self._soft_start = soft_start
        self._annealing = annealing
        self._last_step = last_step
        self._global_step = (
            tf.compat.v1.train.get_or_create_global_step()
            if not tf.executing_eagerly()
            else tf.Variable(0, dtype=tf.int64)
        )

    @property
    def last_step(self):
        """Gets the last step."""
        return self._last_step

    @property
    def global_step(self):
        """Gets the global step (tensor)."""
        if not tf.executing_eagerly():
            return tf.compat.v1.train.get_global_step()
        return self._global_step

    @last_step.setter
    def last_step(self, last_step):
        """Sets the last step.

        Args:
            last_step (int): Last step the schedule is made for.
        """
        self._last_step = last_step

    @global_step.setter
    def global_step(self, global_step):
        """Sets the global step.

        Args:
            global_step (tensor): Step at which to get the value of the schedule.
        """
        self._global_step = global_step

    def get_tensor(self):
        """Get the learning rate tensor.

        Returns:
            scalar tensor (tf.float32)
        """
        if not self._last_step:
            raise ValueError("last step must be > 0. It is {}".format(self._last_step))

        learning_rate_tensor = nvidia_tao_tf1.core.hooks.utils.get_softstart_annealing_learning_rate(
            progress=tf.cast(self.global_step, dtype=tf.float32) / self._last_step,
            soft_start=self._soft_start,
            annealing=self._annealing,
            base_lr=self._base_learning_rate,
            min_lr=self._min_learning_rate,
        )
        if not tf.executing_eagerly():
            tf.compat.v1.summary.scalar(
                name="learning_rate", tensor=learning_rate_tensor
            )
        return learning_rate_tensor
