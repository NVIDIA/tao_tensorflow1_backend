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
"""Tests for TrainingConfig parsing functions."""
from __future__ import absolute_import

import keras
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.proto.learning_rate_config_pb2 import LearningRateConfig
from nvidia_tao_tf1.cv.detectnet_v2.proto.optimizer_config_pb2 import OptimizerConfig
from nvidia_tao_tf1.cv.detectnet_v2.proto.regularizer_config_pb2 import RegularizerConfig
from nvidia_tao_tf1.cv.detectnet_v2.training.training_proto_utilities import (
    build_learning_rate_schedule,
    build_optimizer,
    build_regularizer
)


def test_build_optimizer():
    """Test optimizer parsing."""
    optimizer_config = OptimizerConfig()
    learning_rate = 0.5

    # Default values shouldn't pass.
    with pytest.raises(NotImplementedError):
        build_optimizer(optimizer_config, learning_rate)

    # Valid config should work.
    optimizer_config.adam.epsilon = 0.01
    optimizer_config.adam.beta1 = 0.9
    optimizer_config.adam.beta2 = 0.999
    ret = build_optimizer(optimizer_config, learning_rate)
    assert isinstance(ret, tf.train.AdamOptimizer)

    # Test various invalid values.
    with pytest.raises(ValueError):
        optimizer_config.adam.beta1 = 1.1
        build_optimizer(optimizer_config, learning_rate)

    with pytest.raises(ValueError):
        optimizer_config.adam.beta1 = 0.9
        optimizer_config.adam.beta2 = -1.0
        build_optimizer(optimizer_config, learning_rate)

    with pytest.raises(ValueError):
        optimizer_config.adam.beta2 = 0.99
        optimizer_config.adam.epsilon = 0.0
        build_optimizer(optimizer_config, learning_rate)


def test_build_regularizer():
    """Test regularizer parsing."""
    regularizer_config = RegularizerConfig()
    weight = 0.001

    # Default values should pass (defaults to NO_REG).
    ret = build_regularizer(regularizer_config)
    assert ret == (None, None)

    # Test the other regularization types.
    regularizer_config.weight = weight

    regularizer_config.type = RegularizerConfig.L1
    ret = build_regularizer(regularizer_config)
    assert isinstance(ret[0], keras.regularizers.L1L2)
    assert isinstance(ret[1], keras.regularizers.L1L2)
    assert pytest.approx(ret[0].get_config()['l1']) == weight
    assert pytest.approx(ret[0].get_config()['l2']) == 0.0
    assert pytest.approx(ret[1].get_config()['l1']) == weight
    assert pytest.approx(ret[1].get_config()['l2']) == 0.0

    regularizer_config.type = RegularizerConfig.L2
    ret = build_regularizer(regularizer_config)
    assert isinstance(ret[0], keras.regularizers.L1L2)
    assert isinstance(ret[1], keras.regularizers.L1L2)
    assert pytest.approx(ret[0].get_config()['l1']) == 0.0
    assert pytest.approx(ret[0].get_config()['l2']) == weight
    assert pytest.approx(ret[1].get_config()['l1']) == 0.0
    assert pytest.approx(ret[1].get_config()['l2']) == weight

    # Test invalid values.
    with pytest.raises(ValueError):
        regularizer_config.weight = -1.0
        build_regularizer(regularizer_config)


def test_build_learning_rate_schedule():
    """Test learning rate schedule parsing."""
    learning_rate_config = LearningRateConfig()

    # Default values should not pass, forcing user to set the config.
    with pytest.raises(NotImplementedError):
        build_learning_rate_schedule(learning_rate_config, 10)

    # Default values should not pass.
    params = learning_rate_config.soft_start_annealing_schedule
    params.min_learning_rate = 0.1
    with pytest.raises(ValueError):
        build_learning_rate_schedule(learning_rate_config, 10)

    # Setting proper values should pass.
    params.max_learning_rate = 1.0
    params.soft_start = 0.1
    params.annealing = 0.7
    ret = build_learning_rate_schedule(learning_rate_config, 10)
    assert isinstance(ret, tf.Tensor)

    # Test various invalid values.
    with pytest.raises(ValueError):
        params.min_learning_rate = 0.0
        build_learning_rate_schedule(learning_rate_config, 10)

    with pytest.raises(ValueError):
        params.min_learning_rate = 0.1
        params.max_learning_rate = 0.0
        build_learning_rate_schedule(learning_rate_config, 10)

    with pytest.raises(ValueError):
        params.soft_start = 1.0
        params.max_learning_rate = 1.0
        build_learning_rate_schedule(learning_rate_config, 10)

    with pytest.raises(ValueError):
        params.soft_start = 0.4
        params.annealing = 0.3
        build_learning_rate_schedule(learning_rate_config, 10)

    with pytest.raises(ValueError):
        params.soft_start = 0.4
        params.annealing = 1.1
        build_learning_rate_schedule(learning_rate_config, 10)
