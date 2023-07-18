"""Checkpointing hook with encryption."""
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

import os
import shutil
from zipfile import ZipFile

import tensorflow as tf

from nvidia_tao_tf1.cv.mask_rcnn.utils.logging_formatter import logging
__all__ = ["EncryptCheckpointSaverHook"]


class EncryptCheckpointSaverHook(tf.estimator.SessionRunHook):
    """Saves encrypted checkpoints every N steps or seconds."""

    def __init__(self, checkpoint_dir, temp_dir, key, checkpoint_basename="model.ckpt",
                 steps_per_epoch=100):
        """Initializes a `EncryptCheckpointSaverHook`.

        Args:
          checkpoint_dir: `str`, base directory for the checkpoint files.
          checkpoint_basename: `str`, base name for the checkpoint files.
        Raises:
          ValueError: One of `save_steps` or `save_secs` should be set.
          ValueError: At most one of `saver` or `scaffold` should be set.
        """
        logging.info("Create EncryptCheckpointSaverHook.")

        self._saver = None
        self._checkpoint_dir = checkpoint_dir
        self._save_path = temp_dir
        self._key = key

        self._steps_per_run = 1
        self._steps_per_epoch = steps_per_epoch
        self._is_initialized = False

        self._global_step_tensor = None
        self._summary_writer = None

    def _set_steps_per_run(self, steps_per_run):
        """Set steps."""
        self._steps_per_run = steps_per_run

    def begin(self):
        """Begin."""
        self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()

        self._saver = tf.compat.v1.train.Saver()

        from tensorflow.python.training import summary_io
        self._summary_writer = summary_io.SummaryWriterCache.get(self._checkpoint_dir)

        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use EncryptCheckpointSaverHook."
            )

    def after_create_session(self, session, coord):
        """After session created."""
        if not self._is_initialized:
            global_step = session.run(self._global_step_tensor)
            from tensorflow.python.keras.backend import get_graph
            default_graph = get_graph()

            # We do write graph and saver_def at the first call of before_run.
            # We cannot do this in begin, since we let other hooks to change graph and
            # add variables in begin. Graph is finalized after all begin calls.
            tf.io.write_graph(
                default_graph.as_graph_def(add_shapes=True),
                self._checkpoint_dir,
                "graph.pbtxt"
            )

            saver_def = self._saver.saver_def

            from tensorflow.python.framework import meta_graph

            meta_graph_def = meta_graph.create_meta_graph_def(
                graph_def=default_graph.as_graph_def(add_shapes=True),
                saver_def=saver_def
            )

            self._summary_writer.add_graph(default_graph)
            self._summary_writer.add_meta_graph(meta_graph_def)

            # The checkpoint saved here is the state at step "global_step".
            self._save(session, global_step)

            self._is_initialized = True

    def end(self, session):
        """Run session."""
        last_step = session.run(self._global_step_tensor)
        self._save(session, last_step)

    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        # reset model saving path
        # to prevent unwanted files after evaluation
        if os.path.exists(self._save_path):
            shutil.rmtree(self._save_path)
            os.mkdir(self._save_path)
        self._saver.save(session, os.path.join(self._save_path, "model.ckpt"), global_step=step)

        # Template for zip file.
        # Step based checkpoint naming < TAO 5.0
        # tlt_file = os.path.join(self._checkpoint_dir, 'model.step-{}.tlt'.format(step))

        # New checkpoint naming for > TAO 5.0
        epoch = int(step / self._steps_per_epoch)
        tlt_file = os.path.join(self._checkpoint_dir, 'model.epoch-{}.tlt'.format(epoch))
        logging.info("Saving checkpoints for epoch %d into %s.", epoch, tlt_file)

        prev_dir = os.getcwd()
        os.chdir(self._save_path)
        # Zip the checkpoint files to one file.
        with ZipFile(tlt_file, 'w') as zip_object:
            for ckpt_file in os.listdir(self._save_path):
                logging.debug(f"Writing out data for {ckpt_file}")
                zip_object.write(ckpt_file)

        # Restore previous execution directory and remove tmp files/directories.
        os.chdir(prev_dir)
        shutil.rmtree(self._save_path)
        os.mkdir(self._save_path)

        # # TODO(@yuw): whether to change self._save_path to self._checkpoint_dir
        self._summary_writer.add_session_log(
            tf.compat.v1.SessionLog(status=tf.compat.v1.SessionLog.CHECKPOINT,
                                    checkpoint_path=self._save_path), step)
