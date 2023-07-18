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

"""Hook for job progress monitoring on clusters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import timedelta
import logging
import time
import tensorflow.compat.v1 as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging

logger = logging.getLogger(__name__)

MONITOR_JSON_FILENAME = "monitor.json"


def write_status_json(
    save_path, loss_value, current_epoch, max_epoch, time_per_epoch, ETA, learning_rate
):
    """Write out the data to the status.json file initiated by the experiment for monitoring.

    Args:
        save_path (str): Path where monitor.json needs to be saved. Basically the
            result directory.
        loss_value (float): Current value of loss to be recorder in the monitor.
        current_epoch (int): Current epoch.
        max_epoch (int): Total number of epochs.
        time_per_epoch (float): Time per epoch in seconds.
        ETA (float): Time per epoch in seconds.
        learning_rate (float): Learning rate tensor.

    Returns:
        monitor_data (dict): The monitor data as a dict.
    """

    monitor_data = {
        "epoch": current_epoch,
        "max_epoch": max_epoch,
        "time_per_epoch": str(timedelta(seconds=time_per_epoch)),
        "eta": str(timedelta(seconds=ETA)),
    }
    s_logger = status_logging.get_status_logger()
    # Save the json file.
    try:
        s_logger.graphical = {
            "loss": loss_value,
            "learning_rate": learning_rate
        }
        s_logger.write(
            data=monitor_data,
            status_level=status_logging.Status.RUNNING)
    except IOError:
        # We let this pass because we do not want the json file writing to crash the whole job.
        pass

    # Adding the data back after the graphical data was set to the status logger.
    monitor_data["loss"] = loss_value
    monitor_data["learning_rate"] = learning_rate
    return monitor_data


class TaskProgressMonitorHook(tf.estimator.SessionRunHook):
    """Log loss and epochs for monitoring progress of cluster jobs.

    Writes the current training progress (current loss, current epoch and
    maximum epoch) to a json file.
    """

    def __init__(self, loggable_tensors, save_path, epochs, steps_per_epoch):
        """Initialization.

        Args:
            loss: Loss tensor.
            save_path (str): Absolute save path.
            epochs (int): Number of training epochs.
            steps_per_epoch (int): Number of steps per epoch.
        """
        # Define the tensors to be fetched at every step.
        self._fetches = loggable_tensors
        self.save_path = save_path
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        # Initialize variables for epoch time calculation.
        self.time_per_epoch = 0
        self._step_start_time = None
        # Closest estimate of the start time, in case starting from mid-epoch.
        self._epoch_start_time = time.time()

    def before_run(self, run_context):
        """Request loss and global step from the session.

        Args:
            run_context: A `SessionRunContext` object.
        Returns:
            A `SessionRunArgs` object.
        """
        # Record start time for each step. Use the value later, if this step started an epoch.
        self._step_start_time = time.time()
        # Assign the tensors to be fetched.
        return tf.train.SessionRunArgs(self._fetches)

    def after_run(self, run_context, run_values):
        """Write the progress to json-file after each epoch.

        Args:
            run_context: A `SessionRunContext` object.
            run_values: A `SessionRunValues` object. Contains the loss value
                requested by before_run().
        """
        # Get the global step value.
        step = run_values.results["step"]

        if (step + 1) % self.steps_per_epoch == 0:
            # Last step of an epoch is completed.
            epoch_end_time = time.time()
            self.time_per_epoch = epoch_end_time - self._epoch_start_time

        if step % self.steps_per_epoch == 0:
            # First step of a new epoch is completed. Store the time when step was started.
            self._epoch_start_time = self._step_start_time
            loss_value = run_values.results["loss"]
            learning_rate = str(run_values.results.get("learning_rate", "Not logged"))
            current_epoch = int(step // self.steps_per_epoch)
            monitor_data = write_status_json(
                save_path=self.save_path,
                loss_value=float(loss_value),
                current_epoch=current_epoch,
                max_epoch=self.epochs,
                time_per_epoch=self.time_per_epoch,
                ETA=(self.epochs - current_epoch) * self.time_per_epoch,
                learning_rate=learning_rate
            )
            logger.info(
                "Epoch %d/%d: loss: %0.5f learning rate: %s Time taken: %s ETA: %s"
                % (
                    monitor_data["epoch"],
                    monitor_data["max_epoch"],
                    monitor_data["loss"],
                    monitor_data["learning_rate"],
                    monitor_data["time_per_epoch"],
                    monitor_data["eta"],
                )
            )
