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
"""BpNet Exponential Decay Learning Rate Schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.learning_rate_schedules.exponential_decay_schedule import (
    ExponentialDecayLearningRateSchedule
)
from nvidia_tao_tf1.core.coreobject import save_args


class BpNetExponentialDecayLRSchedule(ExponentialDecayLearningRateSchedule):
    """BpNetExponentialDecayLRSchedule class.

    Derived from ExponentialDecayLearningRateSchedule to accomodate
    option to use decay_epochs and/instead of `decay_steps`. This helps
    to avoid manually calculating `decay_steps` for different settings
    like multi-gpu training, different sizes of datasets, batchsizes etc.
    """

    @save_args
    def __init__(self,
                 decay_epochs,
                 decay_steps=None,
                 **kwargs):
        """__init__ method.

        decayed_learning_rate = learning_rate *
                                decay_rate ^ (global_step / decay_steps)

        Args:
            decay_epochs (int): number of epochs before next decay.
            decay_steps (int): number of steps before next decay.
        """
        super(BpNetExponentialDecayLRSchedule, self).__init__(decay_steps=decay_steps,
                                                              **kwargs)
        self._decay_epochs = decay_epochs

    def update_decay_steps(self, steps_per_epoch):
        """Update the decay steps using decay_epochs and steps_per_epoch."""

        self._decay_steps = self._decay_epochs * steps_per_epoch
        print("Decay Steps: {}".format(self._decay_steps))
