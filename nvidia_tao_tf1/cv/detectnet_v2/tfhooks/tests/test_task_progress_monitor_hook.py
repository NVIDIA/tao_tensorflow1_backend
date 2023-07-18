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

"""Tests for the TaskProgressMonitorHook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import mock
import numpy as np
import tensorflow.compat.v1 as tf

from nvidia_tao_tf1.cv.common.logging import logging as status_logging
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.task_progress_monitor_hook import (
    TaskProgressMonitorHook
)

if sys.version_info >= (3, 0):
    _BUILTIN_OPEN = "builtins.open"
else:
    _BUILTIN_OPEN = "__builtin__.open"

status_logging.set_status_logger(status_logging.StatusLogger(filename="/root", is_master=False))


@mock.patch("time.time")
def test_task_progress_monitor_hook(mock_time):
    """Test that monitor.json is correctly written."""
    num_epochs = 2
    steps_per_epoch = 3
    mock_time.side_effect = [1000, 1060, 2000, 2180]
    loggable_tensors = {}
    with tf.device("/cpu:0"):
        x = tf.placeholder(1)
        y = tf.placeholder(1)
        z = tf.placeholder(1)
        loggable_tensors["loss"] = x
        loggable_tensors["learning_rate"] = y
        loggable_tensors["step"] = z
        progress_monitor_hook = TaskProgressMonitorHook(
            loggable_tensors, "", num_epochs, steps_per_epoch
        )

    # Input data is a sequence of numbers.
    data = np.arange(num_epochs * steps_per_epoch)
    learning_rate = np.arange(num_epochs * steps_per_epoch)
    expected_time_per_epoch = {0: "0:00:00", 1: "0:01:00"}
    expected_ETA = {0: "0:00:00", 1: "0:01:00"}

    mock_open = mock.mock_open()
    handle = mock_open()
    with mock.patch(_BUILTIN_OPEN, mock_open, create=True):
        with tf.train.SingularMonitoredSession(hooks=[progress_monitor_hook]) as sess:
            for epoch in range(num_epochs):
                for step in range(steps_per_epoch):
                    sess.run([loggable_tensors], feed_dict={
                             x: data[epoch * steps_per_epoch + step],
                             y: learning_rate[epoch * steps_per_epoch + step],
                             z: epoch * steps_per_epoch + step})
                    expected_write_data = {
                        "cur_epoch": epoch,
                        "loss": steps_per_epoch * epoch,
                        "max_epoch": num_epochs,
                        "ETA": expected_ETA[epoch],
                        "time_per_epoch": expected_time_per_epoch[epoch],
                        "learning_rate": epoch * steps_per_epoch
                    }
                    assert handle.write.called_once_with(expected_write_data)


def test_epoch_time():
    """Test that time taken per epoch is calculated correctly."""
    num_epochs = 2
    steps_per_epoch = 2
    x = tf.placeholder(1)
    progress_monitor_hook = TaskProgressMonitorHook(
        x, "", num_epochs, steps_per_epoch)
    expected_time_per_epoch = {0: "0:00:00", 1: "0:00:02"}
    expected_ETA = {0: "0:00:00", 1: "0:00:02"}
    # Mock run_values argument for after_run()
    progress_monitor_hook.begin()
    mock_open = mock.mock_open()
    handle = mock_open()
    with mock.patch(_BUILTIN_OPEN, mock_open, create=True):
        global_step = 0
        for epoch in range(num_epochs):
            for _ in range(steps_per_epoch):
                mock_run_values = mock.MagicMock(
                    results={"loss": 2, "step": global_step, "learning_rate": 0.1}
                )
                progress_monitor_hook.before_run(None)
                time.sleep(1)
                progress_monitor_hook.after_run(None, mock_run_values)
                expected_write_data = {
                    "cur_epoch": epoch,
                    "loss": 2,
                    "max_epoch": num_epochs,
                    "ETA": expected_ETA[epoch],
                    "time_per_epoch": expected_time_per_epoch[epoch],
                    "learning_rate": 0.1,
                }
                assert handle.write.called_once_with(expected_write_data)
                global_step += 1
