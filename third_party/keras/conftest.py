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
"""Test configuration."""

from __future__ import absolute_import

import logging
import logging.config

import pytest

"""Root logger for tests."""
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


@pytest.fixture(scope="function", autouse=True)
def clear_session():
    """Clear the Keras session at the end of a test."""
    import keras
    import tensorflow as tf
    import random
    import numpy as np
    import third_party.keras.tensorflow_backend

    third_party.keras.tensorflow_backend.limit_tensorflow_GPU_mem(gpu_fraction=0.9)
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    tf.compat.v1.set_random_seed(DEFAULT_SEED)
    # Yield and let test run to completion.
    yield
    # Clear session.
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


def pytest_addoption(parser):
    """
    Verbosity options.

    This adds two command-line flags:
    --vv for INFO verbosity,
    --vvv for DEBUG verbosity.

    Example:
    pytest -s -v --vv modulus

    Default logging verbosity is WARNING.
    """
    parser.addoption(
        "--vv", action="store_true", default=False, help="log INFO messages."
    )
    parser.addoption(
        "--vvv", action="store_true", default=False, help="log DEBUG messages."
    )


def pytest_configure(config):
    """
    Pytest configuration.

    This is executed after parsing the command line.
    """
    if config.getoption("--vvv"):
        verbosity = "DEBUG"
    elif config.getoption("--vv"):
        verbosity = "INFO"
    else:
        verbosity = "WARNING"

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=verbosity
    )
