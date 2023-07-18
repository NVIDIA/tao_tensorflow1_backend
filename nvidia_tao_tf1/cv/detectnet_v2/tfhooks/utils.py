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

"""DetectNet_v2 setting up hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from nvidia_tao_tf1.core import distribution
import nvidia_tao_tf1.core.hooks
from nvidia_tao_tf1.core.hooks.validation_hook import ValidationHook
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.checkpoint_saver_hook import IVACheckpointSaverHook

INFREQUENT_SUMMARY_KEY = 'infrequent_summary'


def get_common_training_hooks(log_tensors, log_every_n_secs, checkpoint_n_steps, model, last_step,
                              checkpoint_dir, scaffold, summary_every_n_steps,
                              infrequent_summary_every_n_steps,
                              steps_per_epoch=None, validation_every_n_steps=None,
                              evaluator=None, model_store_config=None, listeners=None,
                              max_ckpt_to_keep=5, key=None):
    """Set up commonly used hooks for tensorflow training sessions.

    Args:
        log_tensors (dict): A dictionary of tensors to print to stdout. The keys of the dict should
            be strings, and the values should be tensors.
        log_every_n_secs (int): Log the ``log_tensors`` argument every ``n`` seconds.
        checkpoint_n_steps (int, list): Perform a tensorflow and Keras checkpoint every ``n`` steps.
        model: An instance of ``keras.models.Model`` to be saved with each snapshot.
        last_step (int): The step after which the associated session's `should_stop` method should
            evaluate to ``True``.
        checkpoint_dir: The directory used for saving the graph, summaries and checkpoints. In case
            it's ``None``, no checkpoints and model files will be saved and no tensorboard summaries
            will be produced.
        scaffold: An instance of the same ``tf.train.Scaffold`` that will be passed to the
            training session.
        summary_every_n_steps: Save sumaries every ``n`` steps. The steps per second will also
            be printed to console.
        infrequent_summary_every_n_steps: Save infrequent summaries every ``n`` steps. This is for
            summaries that should be rarely evaluated, like images or histograms. This relates
            to summaries marked with the ``INFREQUENT_SUMMARY_KEY`` key.
        steps_per_epoch (int): Number of steps per epoch.
        validation_every_n_steps (int): Validate every ``n`` steps. Should be specified if evaluator
            object is not None.
        evaluator: An instance of Evaluator class that performs evaluation (default=None).
        model_store_config (dict): a dictionary consisting of the following key/values:
            client (:any:`modelstore.Client`): client to use to push model checkpoints.
            model_id (str): ID of model to assign model checkpoints to.
            param_set_id (str): ID of param set to assign model checkpoint to.
            fold (int): fold to assign model checkpoints to.
        listeners: A list of CheckpointSaverListener objects (or child classes). Can be None.
            If provided, will leave out the default listeners provided otherwise.
        max_ckpt_to_keep: Maximum number of model checkpoints to keep.
    Returns:
        A list of hooks, all inheriting from ``tf.SessionRunHook``.
    """
    hooks = [tf.estimator.LoggingTensorHook(tensors=log_tensors, every_n_secs=log_every_n_secs),
             tf.estimator.StopAtStepHook(last_step=last_step),
             # Setup hook that cleanly stops the session if SIGUSR1 is received.
             nvidia_tao_tf1.core.hooks.SignalHandlerHook(), ]

    if model is not None:
        hooks.append(nvidia_tao_tf1.core.hooks.KerasModelHook(model))

    # If we are running in a distributed setting, we need to broadcast the initial variables.
    if distribution.get_distributor().is_distributed():
        hooks.append(distribution.get_distributor().broadcast_global_variables_hook())

    # Save checkpoints only on master to prevent other workers from corrupting them.
    if distribution.get_distributor().is_master():
        step_counter_hook = tf.estimator.StepCounterHook(
            every_n_steps=summary_every_n_steps,
            output_dir=checkpoint_dir
        )
        hooks.append(step_counter_hook)

        if checkpoint_dir is not None:
            if listeners is None:
                listeners = []
            if model is not None:
                keras_checkpoint_listener = nvidia_tao_tf1.core.hooks.KerasCheckpointListener(
                    model=model, checkpoint_dir=checkpoint_dir,
                    max_to_keep=max_ckpt_to_keep)
                listeners.insert(0, keras_checkpoint_listener)

            if not isinstance(checkpoint_n_steps, list):
                checkpoint_n_steps = [checkpoint_n_steps]

            for n_steps in checkpoint_n_steps:
                checkpoint_hook = IVACheckpointSaverHook(checkpoint_dir=checkpoint_dir,
                                                         key=key,
                                                         save_steps=n_steps,
                                                         listeners=listeners,
                                                         steps_per_epoch=steps_per_epoch,
                                                         scaffold=scaffold)
                hooks.append(checkpoint_hook)

            # Set up the frequent and infrequent summary savers.
            summary_saver_directory = os.path.join(checkpoint_dir, "events")
            if not os.path.exists(summary_saver_directory):
                os.makedirs(summary_saver_directory)
            if summary_every_n_steps > 0:
                summary_saver = tf.estimator.SummarySaverHook(
                    save_steps=summary_every_n_steps,
                    scaffold=scaffold,
                    output_dir=summary_saver_directory
                )
                hooks.append(summary_saver)

            if infrequent_summary_every_n_steps > 0:
                infrequent_summary_op = tf.compat.v1.summary.merge_all(key=INFREQUENT_SUMMARY_KEY)

                if infrequent_summary_op is None:
                    raise ValueError('Infrequent summaries requested, but None found.')

                infrequent_summary_saver = tf.estimator.SummarySaverHook(
                    save_steps=infrequent_summary_every_n_steps,
                    output_dir=summary_saver_directory,
                    summary_op=infrequent_summary_op)
                hooks.append(infrequent_summary_saver)

        # Set up evaluator hook after checkpoint saver hook, so that evaluation is performed
        # on the latest saved model.
        if evaluator is not None:
            if validation_every_n_steps is not None:
                hooks.append(ValidationHook(evaluator, validation_every_n_steps))
            else:
                raise ValueError('Specify ``validation_every_n_steps`` if Evaluator is not None')

    return hooks
