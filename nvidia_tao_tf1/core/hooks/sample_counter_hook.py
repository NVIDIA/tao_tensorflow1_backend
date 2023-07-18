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

"""Hook for calculating the training throughput."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import tensorflow.compat.v1 as tf

logger = logging.getLogger(__name__)


class SampleCounterHook(tf.estimator.SessionRunHook):
    """Hook that logs throughput in a tf.Session."""

    def __init__(self, batch_size, every_n_steps=25, name=""):
        """Constructor.

        Args:
            batch_size (int): Number of samples in a minibatch.
            every_n_steps (int): Controls how often the hook actually logs the throughput.
            name (str): Name for session. Optional default "".
        """
        self._batch_size = batch_size
        self._every_n_steps = every_n_steps

        self._start_time = None
        self._step_counter = -1
        self._samples_per_second = 0.0
        self._name = name

    def before_run(self, run_context):
        """Increment internal step counter and reset the timer if necessary.

        Args:
            run_context: A `SessionRunContext` object.

        Returns:
            A `SessionRunArgs` object.
        """
        self._step_counter += 1
        if self._step_counter % self._every_n_steps == 0:
            self._start_time = time.time()

    def after_run(self, run_context, run_values):
        """Calculate the throughput, if necessary.

        Args:
            run_context: A `SessionRunContext` object.
            run_values: A `SessionRunValues` object.
        """
        if (
            self._step_counter + 1
        ) % self._every_n_steps == 0 or self._step_counter == 0:
            time_taken = time.time() - self._start_time
            self._samples_per_second = (
                (self._batch_size * self._every_n_steps) / time_taken
                if self._step_counter
                else self._batch_size / time_taken
            )
            logger.info(
                "{} Samples / sec: {:.3f}".format(self._name, self._samples_per_second)
            )

    def end(self, session):
        """Print samples per sec at the end of the run.

        Args: session: A `Session` object.
        """
        logger.info(
            "{} Samples / sec: {:.3f}".format(self._name, self._samples_per_second)
        )
