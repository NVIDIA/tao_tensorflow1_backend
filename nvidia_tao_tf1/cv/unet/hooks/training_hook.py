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

"""Hook to save the loss logs."""

from datetime import timedelta
import gc
import json
import os
import time
import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.unet.distribution import distribution

MONITOR_JSON_FILENAME = "monitor.json"


def write_monitor_json(
    save_path, loss_value, running_avg_loss, current_epoch, max_epoch, time_per_epoch, ETA
):
    """Write the monitor.json file for cluster monitoring purposes.

    Args:
        save_path (str): Path where monitor.json needs to be saved. Basically the
            result directory.
        loss_value (float): Current value of loss to be recorder in the monitor.
        current_epoch (int): Current epoch.
        max_epoch (int): Total number of epochs.
        time_per_epoch (float): Time per epoch in seconds.
        ETA (float): Time per epoch in seconds.

    Returns:
        monitor_data (dict): The monitor data as a dict.
    """
    s_logger = status_logging.get_status_logger()
    monitor_data = {
        "epoch": current_epoch,
        "max_epoch": max_epoch,
        "time_per_epoch": str(timedelta(seconds=time_per_epoch)),
        "ETA": str(timedelta(seconds=ETA)),
        "mini_batch_loss": loss_value,
        "running_average_loss": running_avg_loss,
    }
    # Save the json file.
    try:
        s_logger.graphical = {
            "loss": running_avg_loss,
        }
        s_logger.write(
            data=monitor_data,
            status_level=status_logging.Status.RUNNING)
    except IOError:
        # We let this pass because we do not want the json file writing to crash the whole job.
        pass

    # Save the json file.
    filename = os.path.join(save_path, MONITOR_JSON_FILENAME)
    try:
        with open(filename, "w") as f:
            json.dump(monitor_data, f)
    except IOError:
        # We let this pass because we do not want the json file writing to crash the whole job.
        pass


class TrainingHook(tf.estimator.SessionRunHook):
    """Hook to gather and save the Total loss after every training iteration."""

    def __init__(self, logger, steps_per_epoch, max_epochs, save_path, params, log_every=1):
        """Initialize TrainingHook.

        Args:
            logger (str): Output dir to store the losses log file.
            max_steps (int): The key to decode the model.
            log_every (int): Save every N steps.
        """
        self._log_every = log_every
        self._iter_idx = 0
        self.logger = logger
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        # Initialize variables for epoch time calculation.
        self.time_per_epoch = 0
        self._step_start_time = None
        # Closest estimate of the start time, in case starting from mid-epoch.
        self._epoch_start_time = time.time()
        self.save_path = save_path
        self.params = params
        self.run_average_loss = 0
        self.run_average_loss_sum = 0

    def before_run(self, run_context):
        """Losses are consolidated before a training run."""
        run_args = tf.estimator.SessionRunArgs(
            fetches={
                "total_loss" : 'total_loss_ref:0',
                "step": tf.train.get_or_create_global_step()
            }
        )
        self._step_start_time = time.time()

        return run_args

    def after_run(self,
                  run_context,
                  run_values):
        """Losses are consolidated after a training run."""

        cur_step = run_values.results["step"]
        cur_step_per_epoch = (cur_step + 1) % self.steps_per_epoch
        if (cur_step + 1) % self.steps_per_epoch == 0:
            # Last step of an epoch is completed.
            epoch_end_time = time.time()
            self.time_per_epoch = epoch_end_time - self._epoch_start_time
            # First step of a new epoch is completed. Update the time when step was started.
            self._epoch_start_time = self._step_start_time
        total_loss = run_values.results["total_loss"]
        self.run_average_loss_sum += total_loss
        if cur_step_per_epoch:
            self.run_average_loss = self.run_average_loss_sum/float(cur_step_per_epoch)
        else:
            self.run_average_loss = self.run_average_loss_sum/float(self.steps_per_epoch)
            self.run_average_loss_sum = 0
        if (cur_step % self.steps_per_epoch == 0) and (distribution.get_distributor().rank() == 0):
            current_epoch = int(cur_step / self.steps_per_epoch)
            write_monitor_json(
                save_path=self.save_path,
                loss_value=float(total_loss),
                running_avg_loss=float(self.run_average_loss),
                current_epoch=current_epoch,
                max_epoch=self.max_epochs,
                time_per_epoch=self.time_per_epoch,
                ETA=(self.max_epochs - current_epoch) * self.time_per_epoch,
            )
        if (cur_step % self._log_every == 0) and (distribution.get_distributor().rank() == 0):
            current_epoch = float(cur_step / self.steps_per_epoch)
            self.logger.info(
                "Epoch: %f/%d:, Cur-Step: %d, loss(%s): %0.5f, Running average loss:"
                "%0.5f, Time taken: %s ETA: %s"
                % (
                    current_epoch,
                    self.max_epochs,
                    cur_step,
                    self.params.loss,
                    float(total_loss),
                    float(self.run_average_loss),
                    self.time_per_epoch,
                    (self.max_epochs - current_epoch) * self.time_per_epoch,
                )
            )
        gc.collect()
