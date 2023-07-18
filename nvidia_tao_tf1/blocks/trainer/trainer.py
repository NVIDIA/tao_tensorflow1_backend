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
"""Base trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import lru_cache
import logging
import tensorflow as tf

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args


logger = logging.getLogger(__name__)


class Trainer(TAOObject):
    """Trainer class."""

    @save_args
    def __init__(self, dataloader, model, optimizer, loss, hooks=None, **kwargs):
        """__init__ method.

        Args:
            dataloader (DataLoader).
            model (Model).
            optimizer (Optimizer).
            loss (Loss).
        """
        super(Trainer, self).__init__(**kwargs)
        self._dataloader = dataloader
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._hooks = [] if hooks is None else hooks
        tf.compat.v1.train.get_or_create_global_step()

    def train(self):
        """Train method.

        Raises:
            NotImplementedError: method should be subclassed.
        """
        raise NotImplementedError()

    @property
    @lru_cache()
    def local_init_op(self):
        """Initialize the local variables. Used with the scaffold.

        Returns:
            A tf.Operation that will perform the initialization when evaluated.
        """
        return tf.group(
            tf.compat.v1.local_variables_initializer(),
            tf.compat.v1.tables_initializer(),
            *tf.compat.v1.get_collection("iterator_init")
        )

    @property
    @lru_cache()
    def scaffold(self):
        """Create a Scaffold, used to create and gather parts used for our training loop.

        Returns:
            A tf.Scaffold object.
        """
        return tf.compat.v1.train.Scaffold(local_init_op=self.local_init_op)

    def run_training_loop(
        self, train_op, hooks, checkpoint_dir=None, checkpoint_filename_with_path=None
    ):
        """Run the training loop in a tensorflow session.

        Args:
            train_op (tensor): Tensorflow op to be evaluated to take a training step.
            hooks (list of Hooks): List of Tensorflow Hooks to be used as callbacks while running
                training.
            checkpoint_dir (str): for resuming from a checkpoint. If this value is `None` it will
                not restore variables. If it points to a directory, it will find the latest variable
                snapshot and resume from there. Default None.
            checkpoint_filename_with_path (str): For resuming from a checkpoint file. If this value
                is `None` it will not restore variables. Default None.
        """
        # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
        # The SingularMonitoredSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        # Notice we are not using the `MonitoredTrainingSession` variant because that automatically
        # adds unwanted hooks if a `checkpoint_dir` is provided: and if we do not provide it,
        # we cannot resume out checkpoint.
        config = tao_core.distribution.get_distributor().get_config()

        ignore_keras_values = checkpoint_dir is not None \
            or checkpoint_filename_with_path is not None

        if self._model.keras_model is not None:
            # KerasModelHook takes care of initializing model variables.
            hooks.insert(
                0,
                tao_core.hooks.hooks.KerasModelHook(
                    self._model.keras_model,
                    ignore_keras_values
                )
            )

        with tf.compat.v1.train.SingularMonitoredSession(
            hooks=hooks,
            scaffold=self.scaffold,
            checkpoint_dir=checkpoint_dir,
            config=config,
            checkpoint_filename_with_path=checkpoint_filename_with_path,
        ) as sess:
            try:
                while not sess.should_stop():
                    # Run training ops with the wrapped session.
                    sess.run(train_op)

            except (KeyboardInterrupt, SystemExit):
                logger.info("Training interrupted.")
