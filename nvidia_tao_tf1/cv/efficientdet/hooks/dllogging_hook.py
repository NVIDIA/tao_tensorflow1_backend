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
# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================
"""dllogger setup script."""
import time
import dllogger
import tensorflow.compat.v1 as tf


def setup_dllogger(rank, enabled=True, filename='log.json'):
    """Set up Dllogger."""
    if enabled and rank == 0:
        backends = [
            dllogger.StdOutBackend(dllogger.Verbosity.DEFAULT),
            dllogger.JSONStreamBackend(
                dllogger.Verbosity.VERBOSE,
                filename,
                ),
            ]
        dllogger.init(backends)
    # else:
    #     dllogger.init([])


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, warmup=0, keep=False):
        """Init."""
        self.reset()
        self.warmup = warmup
        self.keep = keep

    def reset(self):
        """Reset values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.iters = 0
        self.vals = []

    def update(self, val, n=1):
        """Update."""
        self.iters += 1
        self.val = val

        if self.iters > self.warmup:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            if self.keep:
                self.vals.append(val)


class DLLoggerHook(tf.estimator.SessionRunHook):
    """Dllogger hook."""

    def __init__(self, batch_size, num_examples_per_epoch, logging_frequency,
                 checkpoint_period, rank=-1, size=1):
        """Init."""
        self.local_batch_size = batch_size
        self.global_batch_size = self.local_batch_size * size
        self.num_examples_per_epoch = num_examples_per_epoch
        self.logging_frequency = logging_frequency
        self.checkpoint_period = checkpoint_period
        self.rank = rank

    def after_create_session(self, session, coord):
        """After session is created."""
        self.meters = {}
        warmup = 100
        self.meters['train_throughput'] = AverageMeter(warmup=warmup)

    def before_run(self, run_context):
        """Before session run."""
        self.t0 = time.time()
        return tf.estimator.SessionRunArgs(
            fetches=['learning_rate:0', 'total_loss:0', 'global_step:0'])

    def after_run(self, run_context, run_values):
        """After session run."""
        throughput = self.global_batch_size/(time.time() - self.t0)
        learning_rate, loss, current_step = run_values.results
        if current_step % self.logging_frequency == 0:
            summary = {
                'global step': str(current_step + 1),
                'epoch': str(
                    (((current_step + 1) * self.local_batch_size) //
                        self.num_examples_per_epoch) + 1),
                'learning_rate': str(learning_rate),
                'total_loss': str(loss),
            }
            dllogger.log(step=int(current_step), data=summary)
        # if current_step % self.checkpoint_period == 0:
        #   summary = {
        #     'INFO': 'Saved checkpoint at global step: {}'.format(current_step),
        #   }
        #   dllogger.log(step=int(current_step), data=summary)
        self.meters['train_throughput'].update(throughput)

    def end(self, session):
        """Dump log."""
        summary = {
            'train_throughput': self.meters['train_throughput'].avg,
        }
        dllogger.log(step=tuple(), data=summary)
