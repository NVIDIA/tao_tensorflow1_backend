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

"""IVA checkpoint hook for tlt files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
from zipfile import ZipFile

from nvidia_tao_tf1.core.decorators import override, subclass
from nvidia_tao_tf1.encoding import encoding

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

INFREQUENT_SUMMARY_KEY = b'infrequent_summary'


@subclass
class IVACheckpointSaverHook(tf.estimator.CheckpointSaverHook):
    """Saves time files only for every N steps or seconds."""

    def __init__(self,
                 checkpoint_dir,
                 key=None,
                 save_secs=None,
                 save_steps=None,
                 saver=None,
                 checkpoint_basename="model.ckpt",
                 steps_per_epoch=None,
                 scaffold=None,
                 listeners=None):
        """Initialize an IVACheckpointSaverHook.

        Args:
            checkpoint_dir (str): Base directory for the checkpoint files.
            key (str): The key to decode the model.
            save_secs (int): Save every N secs.
            save_steps (int): Save every N steps.
            saver (Saver): Object used for saving.
            checkpoint_basename (str): Base name for the checkpoint files.
            scaffold (Scaffold): Use to get saver object.
            listeners (list of CheckpointSaverListener): Subclass instances.
            Used for callbacks that run immediately before or after this hook saves
            the checkpoint.

        Raises:
            ValueError: One of `save_steps` or `save_secs` should be set.
            ValueError: At most one of `saver` or `scaffold` should be set.
        """
        # Initialize the parent class.
        super(IVACheckpointSaverHook, self).__init__(checkpoint_dir,
                                                     save_secs=save_secs,
                                                     save_steps=save_steps,
                                                     saver=saver,
                                                     checkpoint_basename=checkpoint_basename,
                                                     scaffold=scaffold,
                                                     listeners=listeners)
        self.key = key
        self.steps_per_epoch = steps_per_epoch

    @override
    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        logging.info("Saving checkpoints for step-%d.", step)

        # Saving the keras model.
        for l in self._listeners:
            l.before_save(session, step)

        should_stop = False

        # Setting up checkpoint saving.
        self._save_encrypted_checkpoint(session, step)

        for l in self._listeners:
            if l.after_save(session, step):
                logging.info(
                    "A CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l))
                should_stop = True
        return should_stop

    def _save_encrypted_checkpoint(self, session, step):
        """Saves the encrypted checkpoint."""
        # Get checkpoint saver and save to tempfile.
        saver = self._get_saver()
        temp_ckpt_path = tempfile.mkdtemp()

        # Template for zip file.
        epoch = int(step / self.steps_per_epoch)
        ckzip_file = os.path.join(self._checkpoint_dir, 'model.epoch-{}.ckzip'.format(epoch))

        # Saving session to the zip file.
        saver.save(session, os.path.join(temp_ckpt_path, "model.ckpt"), global_step=epoch)

        prev_dir = os.getcwd()
        os.chdir(temp_ckpt_path)

        # Zip the checkpoint files to one file.
        with ZipFile(ckzip_file, 'w') as zip_object:
            for ckpt_file in os.listdir(temp_ckpt_path):
                zip_object.write(ckpt_file)

        # Restore previous execution directory and remove tmp files/directories.
        os.chdir(prev_dir)
        shutil.rmtree(temp_ckpt_path)
