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
"""Constant Learning Rate Schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.learning_rate_schedules.learning_rate_schedule import (
    LearningRateSchedule,
)
from nvidia_tao_tf1.core.coreobject import save_args


class ConstantLearningRateSchedule(LearningRateSchedule):
    """ConstantLearningRateSchedule class."""

    @save_args
    def __init__(self, learning_rate, **kwargs):
        """__init__ method.

        Args:
            learning_rate (float): learning_rate value to be used.
        """
        super(ConstantLearningRateSchedule, self).__init__(**kwargs)
        self._learning_rate = learning_rate

    def get_tensor(self):
        """Get the learning rate tensor.

        Returns:
            scalar tensor (tf.float32)
        """
        return tf.constant(self._learning_rate, dtype=tf.float32)
