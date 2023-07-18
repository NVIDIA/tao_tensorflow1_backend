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
"""Base Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.distribution import get_distributor
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args


class Optimizer(TAOObject):
    """Optimizer class."""

    @save_args
    def __init__(self, learning_rate_schedule, **kwargs):
        """__init__ method.

        learning_rate_schedule (LearningRateSchedule): the object from which we obtain the
            learning rate scalar tensor.
        """
        super(Optimizer, self).__init__(**kwargs)
        self._learning_rate_schedule = learning_rate_schedule
        # This needs to be populated by child implementations of the `build()` method.
        self._optimizer = None
        self._learning_rate_tensor = None

    def build(self):
        """Build the optimizer.

        Raises:
            NotImplementedError: should be subclassed.
        """
        # Note: this is expected to populate the `self._optimizer` field.
        raise NotImplementedError()

    def _distribute(self):
        # This helper can be called by child implementations after everything is
        # setup in their `build` methods.
        self._optimizer = get_distributor().distribute_optimizer(self._optimizer)

    def minimize(self, loss, increment_step=True, **kwargs):
        """Minimize the loss by computing and applying gradients.

        Args:
            loss (tensor): the loss to be minimized.

        Returns:
            A tensor operation to be run that performs the minimization.
        """
        if self._optimizer is None:
            self.build()

        if "global_step" not in kwargs and increment_step:
            kwargs["global_step"] = tf.compat.v1.train.get_or_create_global_step()

        train_op = self._optimizer.minimize(loss=loss, **kwargs)
        return train_op

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients."""
        return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        """Apply gradients."""
        return self._optimizer.apply_gradients(*args, **kwargs)

    @property
    def learning_rate_tensor(self):
        """Handle on the learning rate tensor."""
        return self._learning_rate_tensor

    @property
    def learning_rate_schedule(self):
        """Handle on the learning rate schedule.

        Returns:
            (LearningRateSchedule)
        """
        return self._learning_rate_schedule
