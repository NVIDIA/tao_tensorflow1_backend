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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import backend as K
import tensorflow as tf

from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.core import hooks as tao_hooks
from nvidia_tao_tf1.core.utils import set_random_seed
from nvidia_tao_tf1.cv.detectnet_v2.common.graph import get_init_ops


def initialize(random_seed, training_precision=None):
    """Initialization.

    Args:
        random_seed: Random_seed in experiment spec.
        training_precision: (TrainingPrecision or None) Proto object with FP16/FP32 parameters or
            None. None leaves K.floatx() in its previous setting.
    """
    setup_keras_backend(training_precision, is_training=True)

    # Set Maglev random seed. Take care to give different seed to each process.
    seed = distribution.get_distributor().distributed_seed(random_seed)
    set_random_seed(seed)


def setup_keras_backend(training_precision, is_training):
    """Setup Keras-specific backend settings for training or inference.

    Args:
        training_precision: (TrainingPrecision or None) Proto object with FP16/FP32 parameters or
            None. None leaves K.floatx() in its previous setting.
        is_training: (bool) If enabled, Keras is set in training mode.
    """
    # Learning phase of '1' indicates training mode -- important for operations
    # that behave differently at training/test times (e.g. batch normalization)
    if is_training:
        K.set_learning_phase(1)
    else:
        K.set_learning_phase(0)

    # Set training precision, if given. Otherwise leave K.floatx() in its previous setting.
    # K.floatx() determines how Keras creates weights and casts them (Keras default: 'float32').
    if training_precision is not None:
        if training_precision.backend_floatx == training_precision.FLOAT32:
            K.set_floatx('float32')
        elif training_precision.backend_floatx == training_precision.FLOAT16:
            K.set_floatx('float16')
        else:
            raise RuntimeError('Invalid training precision selected')


def get_weights_dir(results_dir):
    """Return weights directory.

    Args:
        results_dir: Base results directory.
    Returns:
        A directory for saved model and weights.
    """
    save_weights_dir = os.path.join(results_dir, 'weights')
    if distribution.get_distributor().is_master() and not os.path.exists(save_weights_dir):
        os.makedirs(save_weights_dir)
    return save_weights_dir


def compute_steps_per_epoch(num_samples, batch_size_per_gpu, logger):
    """Compute steps per epoch based on data set size, minibatch size, and number of GPUs.

    Args:
        num_samples (int): Number of samples in a data set.
        batch_size_per_gpu (int): Minibatch size for a single GPU.
        logger: logger instance.
    Returns:
        Number of steps needed to iterate through the data set once.
    """
    steps_per_epoch, remainder = divmod(num_samples, batch_size_per_gpu)
    if remainder != 0:
        logger.info("Cannot iterate over exactly {} samples with a batch size of {}; "
                    "each epoch will therefore take one extra step.".format(
                        num_samples, batch_size_per_gpu))
        steps_per_epoch = steps_per_epoch + 1

    number_of_processors = distribution.get_distributor().size()
    steps_per_epoch, remainder = divmod(steps_per_epoch, number_of_processors)
    if remainder != 0:
        logger.info("Cannot iterate over exactly {} steps per epoch with {} processors; "
                    "each processor will therefore take one extra step per epoch.".format(
                        steps_per_epoch, batch_size_per_gpu))
        steps_per_epoch = steps_per_epoch + 1
    return steps_per_epoch


def compute_summary_logging_frequency(steps_per_epoch_per_gpu, num_logging_points=10):
    """Compute summary logging point frequency.

    Args:
        steps_per_epoch_per_gpu (int): Number of steps per epoch for single GPU.
        num_logging_points (int): Number of logging points per epoch.
    Returns:
        Summary logging frequency (int).
    """
    if num_logging_points > steps_per_epoch_per_gpu:
        return 1  # Log every step in epoch.

    return steps_per_epoch_per_gpu // num_logging_points


def get_singular_monitored_session(keras_models, session_config=None,
                                   hooks=None, scaffold=None,
                                   checkpoint_filename=None):
    """Create a SingularMonitoredSession with KerasModelHook.

    Args:
        keras_models: A single Keras model or list of Keras models.
        session_config (tf.ConfigProto): Specifies the session configuration options. Optional.
        hooks (list): List of tf.SessionRunHook (or child class) objects. Can be None, in which case
            a KerasModelHook is added, which takes care of properly initializing the variables in
            a keras model.
        scaffold (tf.train.Scaffold): Scaffold object that may contain various pieces needed to
            train a model. Can be None, in which case only local variable initializer ops are added.

    Returns:
        A SingularMonitoredSession that initializes the given Keras model.
    """
    ignore_keras_values = checkpoint_filename is not None
    if hooks is None:
        hooks = []
    if keras_models is not None:
        # KerasModelHook takes care of initializing model variables.
        hooks.insert(0, tao_hooks.KerasModelHook(keras_models, ignore_keras_values))
    if scaffold is None:
        scaffold = tf.train.Scaffold(local_init_op=get_init_ops())
    return tf.train.SingularMonitoredSession(hooks=hooks,
                                             scaffold=scaffold,
                                             config=session_config,
                                             checkpoint_filename_with_path=checkpoint_filename)
