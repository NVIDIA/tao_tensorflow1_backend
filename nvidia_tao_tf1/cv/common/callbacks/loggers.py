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

"""Status Logger callback."""

from collections import Iterable

from datetime import timedelta

import os
import time

from keras.callbacks import Callback

import numpy as np
import six

from nvidia_tao_tf1.cv.common.logging.logging import (
    get_status_logger,
    Status,
    StatusLogger,
    Verbosity
)

# Get default status logger() if it's been previously defined.
logger = get_status_logger()

KEY_MAP = {
    "val_loss": "validation_loss",
    "val_acc": "validation_accuracy",
    "loss": "loss",
    "acc": "training_accuracy",
    "lr": "learning_rate",
    "mAP": "mean average precision"
}


class TAOStatusLogger(Callback):
    """Callback that streams the data training data to a status.json file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    logger = TAOStatusLogger('/path/to/results_dir')
    model.fit(X_train, Y_train, callbacks=[logger])
    ```

    # Arguments
        results_dir (str): The directory where the logs will be saved.
        num_epochs (int): Number of epochs to run the training
        verbosity (status_logger.verbosity.Verbosity()): Verbosity level.
        is_master (bool): Boolean value to check if the gpu rank is 0.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, results_dir, num_epochs=120,
                 verbosity=Verbosity.INFO,
                 append=False,
                 is_master=False):
        """Instantiate the TAOStatusLogger."""
        # Make sure that the status logger obtained is always
        # an instance of nvidia_tao_tf1.cv.common.logging.logging.StatusLogger.
        # Otherwise, this data get's rendered in stdout.
        if isinstance(logger, StatusLogger):
            self.logger = logger
        else:
            self.logger = StatusLogger(
                filename=os.path.join(results_dir, "status.json"),
                is_master=is_master,
                verbosity=verbosity,
                append=append
            )
        self.keys = None
        self.max_epochs = num_epochs
        self._epoch_start_time = None
        super(TAOStatusLogger, self).__init__()

    def on_train_begin(self, logs=None):
        """Write data beginning of the training."""
        self.logger.write(
            status_level=Status.STARTED,
            message="Starting Training Loop."
        )

    @staticmethod
    def _handle_value(k):
        is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
        if isinstance(k, six.string_types):
            return k
        if isinstance(k, Iterable) and not is_zero_dim_ndarray:
            return '"[%s]"' % (', '.join(map(str, k)))
        return k

    def on_epoch_begin(self, epoch, logs=None):
        """Routines to be run at the beginning of the epoch."""
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Collect data at the end of an epoch."""
        logs = logs or {}
        data = {}
        data["epoch"] = epoch + 1
        data["max_epoch"] = self.max_epochs
        epoch_end_time = time.time()
        time_per_epoch = epoch_end_time - self._epoch_start_time
        eta = (self.max_epochs - (epoch + 1)) * time_per_epoch
        data["time_per_epoch"] = str(timedelta(seconds=time_per_epoch))
        data["eta"] = str(timedelta(seconds=eta))
        graphical_data = {}
        kpi_data = {}
        for k, v in logs.items():
            if "loss" in k:
                key = KEY_MAP[k] if k in KEY_MAP.keys() else k
                graphical_data[key] = str(self._handle_value(v))
            if "acc" in k:
                key = KEY_MAP[k] if k in KEY_MAP.keys() else k
                graphical_data[key] = str(self._handle_value(v))
                kpi_data[key] = str(self._handle_value(v))
            if k == "mAP":
                key = KEY_MAP[k] if k in KEY_MAP.keys() else k
                graphical_data[key] = str(self._handle_value(v))
                kpi_data[key] = str(self._handle_value(v))
            if k == "lr":
                graphical_data[KEY_MAP[k]] = str(self._handle_value(v))
        self.logger.graphical = graphical_data
        self.logger.kpi = kpi_data
        self.logger.write(data=data, message="Training loop in progress")

    def on_train_end(self, logs=None):
        """Callback function run at the end of training."""
        self.logger.write(
            status_level=Status.RUNNING,
            message="Training loop complete."
        )
