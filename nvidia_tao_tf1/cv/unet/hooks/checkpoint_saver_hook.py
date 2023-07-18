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
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from nvidia_tao_tf1.core.decorators import override, subclass

INFREQUENT_SUMMARY_KEY = b'infrequent_summary'


@subclass
class IVACheckpointSaverHook(tf.estimator.CheckpointSaverHook):
    """Saves time files only for every N steps or seconds."""

    def __init__(self,
                 checkpoint_dir,
                 save_secs=None,
                 save_steps=None,
                 model_json=None,
                 saver=None,
                 checkpoint_basename="model.ckpt",
                 steps_per_epoch=None,
                 scaffold=None,
                 listeners=None,
                 load_graph=False):
        """Initialize an IVACheckpointSaverHook.

        Args:
            checkpoint_dir (str): Base directory for the checkpoint files.
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
        self.model_json = model_json
        self.load_graph = load_graph
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
        self._save_checkpoint(session, step)

        for l in self._listeners:
            if l.after_save(session, step):
                logging.info(
                    "A CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l))
                should_stop = True
        return should_stop

    def _save_checkpoint(self, session, step):
        """Saves the checkpoint."""
        
        saver = self._get_saver()
        epoch = int(step / self.steps_per_epoch)
        ckzip_folder = os.path.join(self._checkpoint_dir, 'model.epoch-{}.tlt'.format(epoch))
        if not os.path.isdir(ckzip_folder):
            os.makedirs(ckzip_folder)
        # Saving session to the zip file.
        saver.save(session, os.path.join(ckzip_folder, "model.ckpt"), global_step=epoch)
        if self.model_json and self.load_graph:
            with open(self.model_json, 'r') as json_file:
                json_savedModel = json_file.read()
            with open(os.path.join(ckzip_folder, "model.json"), 'w') as json_file:
                json_file.write(json_savedModel)
