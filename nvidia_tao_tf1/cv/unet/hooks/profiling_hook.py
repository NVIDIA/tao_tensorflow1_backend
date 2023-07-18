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

"""Hook to log the thoroughput and latency."""

import time
import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.unet.distribution import distribution
from nvidia_tao_tf1.cv.unet.utils.parse_results import process_performance_stats


class ProfilingHook(tf.estimator.SessionRunHook):
    """Saves thoroughput and latency every N steps."""

    def __init__(self, logger, batch_size, log_every, warmup_steps, mode):
        """Initialize ProfilingHook.

        Args:
            logger (str): Logger object to log the losses.
            batch_size (int): The batch size to compute the performance metrics.
            log_every (int): Save every N steps.
            warmup_steps (int): The warm up steps after which the logging is done.
            mode (str): The mode whether training or evaluation.
        """
        self._log_every = log_every
        self._warmup_steps = warmup_steps
        self._current_step = 0
        self._global_batch_size = batch_size * distribution.get_distributor().size()
        self._t0 = 0
        self._timestamps = []
        self.logger = logger
        self.mode = mode

    def before_run(self, run_context):
        """Training time start recording before a training run."""
        if self._current_step > self._warmup_steps:
            self._t0 = time.time()

    def after_run(self,
                  run_context,
                  run_values):
        """Training time start recording before a training run."""
        if self._current_step > self._warmup_steps:
            self._timestamps.append(time.time() - self._t0)
        self._current_step += 1

    def begin(self):
        """Begin of tensorflow session."""
        pass

    def end(self, session):
        """End of tensorflow session."""
        if distribution.get_distributor().rank() == 0:
            throughput_imgps, latency_ms = process_performance_stats(np.array(self._timestamps),
                                                                     self._global_batch_size)
            self.logger.log(step=(),
                            data={'throughput_{}'.format(self.mode): throughput_imgps,
                                  'latency_{}'.format(self.mode): latency_ms})
