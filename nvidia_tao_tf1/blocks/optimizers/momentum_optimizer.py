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
"""Momentum Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.optimizers.optimizer import Optimizer
from nvidia_tao_tf1.core.coreobject import save_args


class MomentumOptimizer(Optimizer):
    """Momentum class."""

    @save_args
    def __init__(
        self, learning_rate_schedule, momentum=0.9, use_nesterov=False, **kwargs
    ):
        """__init__ method.

        learning_rate_schedule (LearningRateSchedule): The object from which we obtain the
            learning rate scalar tensor.
        momentum (float): A float value or a constant float tensor. The momentum factor. The method
            falls back into gradient descend optimizer when momentum is set to 0.
        use_nesterov (bool): If True, use the Nesterov momentum.
        """
        super(MomentumOptimizer, self).__init__(
            learning_rate_schedule=learning_rate_schedule, **kwargs
        )
        self._momentum = momentum
        self._use_nesterov = use_nesterov

    def build(self):
        """Build the optimizer.

        Instantiates the underlying optimizer object.
        """
        self._learning_rate_tensor = self.learning_rate_schedule.learning_rate_tensor
        self._optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=self._learning_rate_tensor,
            momentum=self._momentum,
            use_nesterov=self._use_nesterov,
        )
        self._distribute()
