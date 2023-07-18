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

"""Soft start annealing learning rate schedule."""

from math import exp, log
from tensorflow import keras


class SoftStartAnnealingLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler implementation.

    Learning rate scheduler modulates learning rate according to the progress in the
    training experiment. Specifically the training progress is defined as the ratio of
    the current iteration to the maximum iterations. Learning rate scheduler adjusts
    learning rate in the following 3 phases:
        Phase 1: 0.0 <= progress < soft_start:
                 Starting from min_lr exponentially increase the learning rate to base_lr
        Phase 2: soft_start <= progress < annealing_start:
                 Maintain the learning rate at base_lr
        Phase 3: annealing_start <= progress <= 1.0:
                 Starting from base_lr exponentially decay the learning rate to min_lr

    Example:
        ```python
        lrscheduler = SoftStartAnnealingLearningRateScheduler(
            max_iterations=max_iterations)

        model.fit(X_train, Y_train, callbacks=[lrscheduler])
        ```

    Args:
        base_lr: Maximum learning rate
        min_lr_ratio: The ratio between minimum learning rate (min_lr) and base_lr
        soft_start: The progress at which learning rate achieves base_lr when starting from min_lr
        annealing_start: The progress at which learning rate starts to drop from base_lr to min_lr
        max_iterations: Total number of iterations in the experiment
    """

    def __init__(self, max_iterations, base_lr=5e-4, min_lr_ratio=0.01, soft_start=0.1,
                 annealing_start=0.7):
        """__init__ method."""
        super(SoftStartAnnealingLearningRateScheduler, self).__init__()

        if not 0.0 <= soft_start <= 1.0:
            raise ValueError('The soft_start varible should be >= 0.0 or <= 1.0.')
        if not 0.0 <= annealing_start <= 1.0:
            raise ValueError('The annealing_start variable should be >= 0.0 or <= 1.0.')
        if not soft_start < annealing_start:
            raise ValueError('Varialbe soft_start should not be less than annealing_start.')

        self.base_lr = base_lr
        self.min_lr_ratio = min_lr_ratio
        self.soft_start = soft_start  # Increase to lr from min_lr until this point.
        self.annealing_start = annealing_start  # Start annealing to min_lr at this point.
        self.max_iterations = max_iterations
        self.min_lr = min_lr_ratio * base_lr
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global_step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """on_train_begin method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = keras.backend.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max_iterations."""
        if not 0. <= progress <= 1.:
            raise ValueError('SoftStartAnnealingLearningRateScheduler '
                             'does not support a progress value < 0.0 or > 1.0 '
                             'received (%f)' % progress)

        if not self.base_lr:
            return self.base_lr

        if self.soft_start > 0.0:
            soft_start = progress / self.soft_start
        else:  # learning rate starts from base_lr
            soft_start = 1.0

        if self.annealing_start < 1.0:
            annealing = (1.0 - progress) / (1.0 - self.annealing_start)
        else:   # learning rate is never annealed
            annealing = 1.0

        t = soft_start if progress < self.soft_start else 1.0
        t = annealing if progress > self.annealing_start else t

        lr = exp(log(self.min_lr) + t * (log(self.base_lr) - log(self.min_lr)))

        return lr
