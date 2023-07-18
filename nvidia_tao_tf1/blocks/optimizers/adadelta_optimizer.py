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
"""Adadelta Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.optimizers.optimizer import Optimizer
from nvidia_tao_tf1.core.coreobject import save_args


class AdadeltaOptimizer(Optimizer):
    """AdadeltaOptimizer class."""

    @save_args
    def __init__(self, learning_rate_schedule, rho=0.95, epsilon=1e-8, **kwargs):
        """__init__ method.

        learning_rate_schedule (LearningRateSchedule): the object from which we obtain the
            learning rate scalar tensor.
        rho (float): A float value or a constant float tensor. The exponential decay rate for
            the 1st moment estimates.
        epsilon (float): A small constant for numerical stability.
        """
        super(AdadeltaOptimizer, self).__init__(
            learning_rate_schedule=learning_rate_schedule, **kwargs
        )
        self._rho = rho
        self._epsilon = epsilon

    def build(self):
        """Build the optimizer.

        Instantiates the underlying optimizer object.
        """
        self._learning_rate_tensor = self.learning_rate_schedule.learning_rate_tensor
        self._optimizer = tf.compat.v1.train.AdadeltaOptimizer(
            learning_rate=self._learning_rate_tensor,
            rho=self._rho,
            epsilon=self._epsilon,
        )
        self._distribute()
