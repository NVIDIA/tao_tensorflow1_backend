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

"""Utility function definitions for file processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import logging
import os
import shutil


CHECKPOINT_FILENAME = 'checkpoint'
VAL_LOG_FILENAME = 'validation.log'


def save_best_model(save_dir, global_step, current_cost, epoch_based_checkpoint=False, extension="tlt"):
    """Only save the model with lowest current_cost in results directory.

    Args:
        save_dir (str): The directory where all the model files are saved.
        global_step (int): Current global step number.
        current_cost (float): Evaluation cost at current step.
    """
    log_filename = os.path.join(save_dir, VAL_LOG_FILENAME)

    # Rename and keep the model in the first round of validation.
    if not os.path.isfile(log_filename):
        with open(log_filename, 'w') as f:
            f.write('best_model@global_step {} : {}\n'.format(global_step, current_cost))
            _shutil_keras_models(
                save_dir, global_step,
                epoch_based_checkpoint,
                extension=extension
            )
    else:
        with open(log_filename, 'r') as f:
            lines = f.readlines()

        # Get current lowest cost from log file.
        lowest_cost = float(lines[0].split()[-1])
        # Save model and discard previous ones if current global step gives lowest loss.
        if current_cost < lowest_cost:
            lines[0] = 'best_model@global_step {} : {}\n'.format(global_step, current_cost)
            with open(log_filename, 'w') as f:
                for line in lines:
                    f.write(line)
            _shutil_keras_models(
                save_dir, global_step,
                epoch_based_checkpoint,
                extension=extension
            )


def _shutil_keras_models(save_dir, global_step, epoch_based_checkpoint=False, extension="tlt"):
    """Shutil copy and move calls to save and delete keras models.

    This will delete old backup models and copy the current keras model to 'model.hdf5'.
    Also moves current keras model to 'model.keras.backup-{global_step}.hdf5".

    Args:
        save_dir (str): The directory where all the model files are saved.
        global_step (int): Current global step number.
    """
    format_string = "step"
    if epoch_based_checkpoint:
        format_string = "epoch"
    old_backup_files = glob.glob(os.path.join(save_dir, f'model.{format_string}.backup-*'))
    for fl in old_backup_files:
        os.remove(fl)
        logging.debug("rm {}".format(fl))

    shutil.copy(os.path.join(save_dir, f'model.{format_string}-') +
                str(global_step) + f'.{extension}', os.path.join(save_dir, f'model.{extension}'))
    shutil.move(os.path.join(save_dir, f'model.{format_string}-') +
                str(global_step) + f'.{extension}',
                os.path.join(save_dir, f'model.{format_string}.backup-') +
                str(global_step) + f'.{extension}')

    logging.debug("cp 'model.keras-{global_step}.{extension}' to 'model.{extension}'".format(
        global_step=global_step,
        extension=extension))
    logging.debug(
        "mv 'model.keras-{global_step}.{extension}' to "
        "'model.keras.backup-{global_step}.{extension}'".format(
            global_step=global_step,
            extension=extension
        )
    )


def clean_checkpoint_dir(save_dir, global_step):
    """Remove extraneous checkpoint files and keras files but keeps last checkpoint files.

    The validation log file VAL_LOG_FILENAME must be initialized.

    Args:
        save_dir (str): The directory where all the model files are saved.

    Raises:
        ValueError: Checkpoint file has not been created.
    """
    _files = glob.glob(os.path.join(save_dir, 'model.epoch-*'))
    # Delete model files other than the lowest cost model files.
    # Keep backup models.
    files_to_delete = [fn for fn in _files if '-' + str(global_step+1) not in fn and
                       'backup' not in fn and '-' + str(global_step) not in fn]

    for fl in files_to_delete:
        os.remove(fl)
        logging.debug("rm {}".format(fl))
