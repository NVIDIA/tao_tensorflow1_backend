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
"""Base Learning Rate Schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import TAOObject


class LearningRateSchedule(TAOObject):
    """LearningRateSchedule class."""

    def __init__(self):
        """__init__ method.

        Initialize common private variables.
        """
        self._steps_per_epoch = None

    @property
    @abstractmethod
    def last_step(self):
        """Gets the last step."""
        raise NotImplementedError("Last step is not defined yet.")

    @property
    @abstractmethod
    def global_step(self):
        """Gets the global step."""
        raise NotImplementedError("Global step is not defined yet.")

    @last_step.setter
    @abstractmethod
    def last_step(self, last_step):
        """Sets the last step.

        Args:
            last_step (int): Last step the schedule is made for.
        """

    @global_step.setter
    @abstractmethod
    def global_step(self, global_step):
        """Sets the global step.

        Args:
            global_step (tensor): Step at which to get the value of the schedule.
        """

    @property
    def learning_rate_tensor(self):
        """Returns a function for the eager mode and a tensor for graph mode."""
        if tf.executing_eagerly():
            return self.get_tensor
        return self.get_tensor()

    @property
    def steps_per_epoch(self):
        """Gets the steps per epoch."""
        return self._steps_per_epoch

    @steps_per_epoch.setter
    def steps_per_epoch(self, steps_per_epoch):
        """Sets the steps per epoch.

        Args:
            steps_per_epoch (int): Number of steps required to consume dataset once.
        """
        self._steps_per_epoch = steps_per_epoch

    def get_tensor(self, *args, **kwargs):
        """Get the learning rate tensor.

        Raises:
            NotImplementedError: method should be subclassed.
        """
        raise NotImplementedError()
