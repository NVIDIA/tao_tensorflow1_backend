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
"""Callbacks: utilities called at certain points during model training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from keras.callbacks import ModelCheckpoint
from nvidia_tao_tf1.cv.common.utils import encode_from_keras


class AdvModelCheckpoint(ModelCheckpoint):
    """Save the encrypted model after every epoch.

    Attributes:
        ENC_KEY: API key to encrypt the model.
        epocs_since_last_save: Number of epochs since model was last saved.
        save_best_only: Flag to save model with best accuracy.
        best: saved instance of best model.
        verbose: Enable verbose messages.
    """

    def __init__(self, filepath, key, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        """Initialization with encryption key."""
        super(AdvModelCheckpoint, self).__init__(filepath)
        self._ENC_KEY = key
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available,'
                                  ' skipping.' % (self.monitor),
                                  RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to '
                                  '%0.5f, saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if filepath.endswith(".hdf5"):
                            self.model.save(filepath, overwrite=True)
                        else:
                            encode_from_keras(self.model, filepath, self._ENC_KEY)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f'
                                  % (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s'
                          % (epoch + 1, filepath))
                if str(filepath).endswith(".hdf5"):
                    self.model.save(str(filepath), overwrite=True)
                else:
                    encode_from_keras(self.model, str(filepath), self._ENC_KEY)
