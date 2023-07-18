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

"""Utility functions for parsing training configs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf

from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.core.hooks.utils import get_softstart_annealing_learning_rate
from nvidia_tao_tf1.cv.detectnet_v2.proto.regularizer_config_pb2 import RegularizerConfig
from nvidia_tao_tf1.cv.detectnet_v2.training.train_op_generator import TrainOpGenerator


def build_optimizer(optimizer_config, learning_rate):
    """Build an Optimizer.

    Arguments:
        optimizer_config (optimizer_config_pb2.OptimizerConfig): Configuration for the Optimizer
            being built.
        learning_rate: Constant or variable learning rate.
    """
    # Check the config and create object.
    distributor = distribution.get_distributor()
    if optimizer_config.HasField("adam"):
        adam = optimizer_config.adam
        if adam.epsilon <= 0.0:
            raise ValueError("AdamOptimizerConfig.epsilon must be > 0")
        if adam.beta1 < 0.0 or adam.beta1 >= 1.0:
            raise ValueError("AdamOptimizerConfig.beta1 must be >= 0 and < 1")
        if adam.beta2 < 0.0 or adam.beta2 >= 1.0:
            raise ValueError("AdamOptimizerConfig.beta2 must be >= 0 and < 1")

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=adam.beta1,
                                           beta2=adam.beta2,
                                           epsilon=adam.epsilon)
    else:
        raise NotImplementedError("The selected optimizer is not supported.")

    # Wrap the optimizer to the Horovod optimizer to ensure synchronous training in the multi-GPU
    # case.
    optimizer = distributor.distribute_optimizer(optimizer)

    return optimizer


def build_regularizer(regularizer_config):
    """Build kernel and bias regularizers.

    Arguments:
        regularizer_config (regularizer_config_pb2.RegularizerConfig): Config for
          regularization.
    Returns:
        kernel_regularizer, bias_regularizer: Keras regularizers created.
    """
    # Check the config and create objects.
    if regularizer_config.weight < 0.0:
        raise ValueError("TrainingConfig.regularization_weight must be >= 0")

    if regularizer_config.type == RegularizerConfig.NO_REG:
        kernel_regularizer = None
        bias_regularizer = None
    elif regularizer_config.type == RegularizerConfig.L1:
        kernel_regularizer = keras.regularizers.l1(regularizer_config.weight)
        bias_regularizer = keras.regularizers.l1(regularizer_config.weight)
    elif regularizer_config.type == RegularizerConfig.L2:
        kernel_regularizer = keras.regularizers.l2(regularizer_config.weight)
        bias_regularizer = keras.regularizers.l2(regularizer_config.weight)
    else:
        raise NotImplementedError("The selected regularizer is not supported.")

    return kernel_regularizer, bias_regularizer


def build_learning_rate_schedule(learning_rate_config, max_steps):
    """Build learning rate schedule.

    Args:
        learning_rate_config (learning_rate_config_pb2.LearningRateConfig): Configuration for
            learning rate.
        max_steps (int): Total number of training steps.
    Returns:
        learning_rate: Learning rate schedule created.
    """
    # Check the config and create objects.
    global_step = tf.train.get_or_create_global_step()
    if learning_rate_config.HasField("soft_start_annealing_schedule"):
        params = learning_rate_config.soft_start_annealing_schedule

        if params.min_learning_rate <= 0.0:
            raise ValueError("SoftStartAnnealingScheduleConfig.min_learning_rate must be > 0")
        if params.max_learning_rate <= 0.0:
            raise ValueError("SoftStartAnnealingScheduleConfig.max_learning_rate must be > 0")
        if params.soft_start < 0.0 or params.soft_start > 1.0 or\
           params.soft_start > params.annealing:
            raise ValueError("SoftStartAnnealingScheduleConfig.soft_start must be between 0 and 1 \
                              and less than SoftStartAnnealingScheduleConfig.annealing")
        if params.annealing < 0.0 or params.annealing > 1.0:
            raise ValueError("SoftStartAnnealingScheduleConfig.annealing must be between 0 and 1")

        learning_rate = get_softstart_annealing_learning_rate(
            progress=tf.cast(global_step, dtype=tf.float32) / max_steps,
            soft_start=params.soft_start,
            annealing=params.annealing,
            base_lr=params.max_learning_rate,
            min_lr=params.min_learning_rate)
    else:
        raise NotImplementedError("The selected learning rate schedule is not supported.")

    return learning_rate


def build_train_op_generator(cost_scaling_config):
    """Build a class that returns train op with or without cost scaling.

    Arguments:
        cost_scaling_config (cost_scaling_config_pb2.CostScalingConfig): Configuration for
            cost scaling.
    """
    if cost_scaling_config.increment < 0.0:
        raise ValueError("CostScalingConfig.increment must be >= 0")
    if cost_scaling_config.decrement < 0.0:
        raise ValueError("CostScalingConfig.decrement must be >= 0")

    return TrainOpGenerator(
        cost_scaling_config.enabled,
        cost_scaling_config.initial_exponent,
        cost_scaling_config.increment,
        cost_scaling_config.decrement
    )
