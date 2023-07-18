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
import time
import tensorflow.compat.v1 as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.mask_rcnn.utils.logging_formatter import logging


def write_status_json(loss_value, current_epoch, max_epoch, time_per_epoch, ETA, learning_rate):
    """Write out the data to the status.json file initiated by the experiment for monitoring.

    Args:
        loss_value (float): Current value of loss to be recorder in the monitor.
        current_epoch (int): Current epoch.
        max_epoch (int): Total number of epochs.
        time_per_epoch (float): Time per epoch in seconds.
        ETA (float): Time per epoch in seconds.
        learning_rate (float): Learning rate tensor.

    Returns:
        monitor_data (dict): The monitor data as a dict.
    """
    s_logger = status_logging.get_status_logger()
    monitor_data = {
        "epoch": current_epoch,
        "max_epoch": max_epoch,
        "time_per_epoch": str(timedelta(seconds=time_per_epoch)),
        "eta": str(timedelta(seconds=ETA)),
    }
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

    def __init__(self, batch_size, epochs, steps_per_epoch, logging_frequency=10):
        """Initialization.

        Args:
            batch_size (str): batch_size for training.
            epochs (int): Number of training epochs.
            steps_per_epoch (int): Number of steps per epoch.
            logging_frequency (int): Print training summary every N steps.
        """
        # Define the tensors to be fetched at every step.
        self.local_batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        assert 0 < logging_frequency <= 1000, "Logging frequency must be no greater than 1000."
        self.logging_frequency = logging_frequency
        # Initialize variables for epoch time calculation.
        self.time_per_epoch = 0
        self._step_start_time = None
        # Closest estimate of the start time, in case starting from mid-epoch.
        self._epoch_start_time = time.time()

    def begin(self):
        """Begin."""
        self._global_step_tensor = tf.train.get_global_step()
        self._fetches = {
            'ops': ['learning_rate:0',
                    'log_total_loss:0',
                    'log_rpn_score_loss:0',
                    'log_rpn_box_loss:0',
                    'log_fast_rcnn_class_loss:0',
                    'log_fast_rcnn_box_loss:0',
                    'global_step:0'],
            'epoch': self._global_step_tensor // self.steps_per_epoch}

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
        learning_rate, loss_value, rpn_sl, rpn_bl, frcnn_sl, frcnn_bl, step = \
            run_values.results['ops']
        current_epoch = (step + 1) // self.steps_per_epoch

        if (step + 1) % self.logging_frequency == 0:
            logging.info(
                "Global step %d (epoch %d/%d): total loss: %0.5f "
                "(rpn score loss: %0.5f rpn box loss: %0.5f "
                "fast_rcnn class loss: %0.5f fast_rcnn box loss: %0.5f) learning rate: %0.5f"
                % (
                    int(step + 1),
                    current_epoch + 1,
                    self.epochs,
                    float(loss_value),
                    float(rpn_sl),
                    float(rpn_bl),
                    float(frcnn_sl),
                    float(frcnn_bl),
                    float(learning_rate)
                )
            )
        if (step + 1) % self.steps_per_epoch == 0:
            # Last step of an epoch is completed.
            epoch_end_time = time.time()
            self.time_per_epoch = epoch_end_time - self._epoch_start_time

        if (step + 1) % self.steps_per_epoch == 0:
            # First step of a new epoch is completed. Store the time when step was started.
            self._epoch_start_time = self._step_start_time
            monitor_data = write_status_json(
                loss_value=float(loss_value),
                current_epoch=int(current_epoch),
                max_epoch=int(self.epochs),
                time_per_epoch=self.time_per_epoch,
                ETA=(self.epochs - current_epoch) * self.time_per_epoch,
                learning_rate=float(learning_rate)
            )
            logging.info(
                "Epoch %d/%d: loss: %0.5f learning rate: %0.5f Time taken: %s ETA: %s"
                % (
                    monitor_data["epoch"],
                    monitor_data["max_epoch"],
                    monitor_data["loss"],
                    monitor_data["learning_rate"],
                    monitor_data["time_per_epoch"],
                    monitor_data["eta"],
                )
            )
