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

"""License plate accuracy callback."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.lprnet.utils.ctc_decoder import decode_ctc_conf


class LPRAccuracyCallback(tf.keras.callbacks.Callback):
    """License plate accuracy callback."""

    def __init__(self, eval_model, eval_interval, val_dataset, verbose=1):
        """init LPR accuracy callback."""

        self.eval_model = eval_model
        self.eval_interval = max(1, int(eval_interval))
        self.val_dataset = val_dataset
        self.verbose = verbose

    def _get_accuracy(self, logs):
        """compute accuracy."""
        # evaluation
        self.eval_model.set_weights(self.model.get_weights())
        total_cnt = self.val_dataset.n_samples
        correct = 0
        batch_size = self.val_dataset.batch_size
        classes = self.val_dataset.classes
        for idx in range(len(self.val_dataset)):
            # prepare data:
            batch_x, batch_y = self.val_dataset[idx]
            # predict:
            prediction = self.eval_model.predict(x=batch_x, batch_size=batch_size)

            # decode prediction
            decoded_lp, _ = decode_ctc_conf(prediction,
                                            classes=classes,
                                            blank_id=len(classes))

            for idx, lp in enumerate(decoded_lp):
                if lp == batch_y[idx]:
                    correct += 1

        if logs is not None:
            logs['accuracy'] = float(correct)/float(total_cnt)

        print("\n")
        print("*******************************************")
        print("Accuracy: {} / {}  {}".format(correct, total_cnt,
                                             float(correct)/float(total_cnt)))
        print("*******************************************")
        print("\n")

        kpi_data = {
            "validation_accuracy": round(float(correct)/float(total_cnt), 5)
        }
        s_logger = status_logging.get_status_logger()
        if isinstance(s_logger, status_logging.StatusLogger):
            s_logger.kpi = kpi_data
            s_logger.write(
                status_level=status_logging.Status.RUNNING,
                message="Evaluation metrics generated."
            )

    def on_epoch_end(self, epoch, logs):
        """evaluate at the epoch end."""
        self.epoch_cnt = epoch + 1

        if self.epoch_cnt % self.eval_interval != 0:
            logs['accuracy'] = np.nan
        else:
            self._get_accuracy(logs)

    def on_train_end(self, logs=None):
        """compute the accuracy at the end of training."""
        if self.epoch_cnt % self.eval_interval != 0:
            self._get_accuracy(logs)
