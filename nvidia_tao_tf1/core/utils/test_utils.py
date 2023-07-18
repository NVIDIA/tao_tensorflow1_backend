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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import keras
from keras import backend as K
import numpy as np
import pytest
import tensorflow as tf

import nvidia_tao_tf1.core.utils


@pytest.mark.parametrize("model_prefix", ["model", "model_1", "super_model"])
def test_find_latest_keras_model_in_directory(tmpdir, model_prefix):
    """Test find latest Keras model in directory."""
    fake_model_dir = tmpdir.mkdir("model_dir")
    mocked_files = [
        "{}.keras-1.hdf5".format(model_prefix),
        "{}.keras-4.hdf5".format(model_prefix),
        "{}.keras-100.hdf5".format(model_prefix),
    ]
    for mocked_file in mocked_files:
        fake_model_dir.join(mocked_file).write("content")
    latest_model_file_path = nvidia_tao_tf1.core.utils.find_latest_keras_model_in_directory(
        fake_model_dir.strpath, model_prefix=model_prefix
    )
    assert latest_model_file_path == os.path.join(
        fake_model_dir.strpath, "{}.keras-100.hdf5".format(model_prefix)
    )

    latest_model_file_path = nvidia_tao_tf1.core.utils.find_latest_keras_model_in_directory(
        tmpdir.mkdir("empty_dir").strpath, model_prefix=model_prefix
    )
    assert latest_model_file_path is None


def test_get_uid():
    # make a random name
    name = "".join(["%c" % (chr(ord("A") + random.randint(0, 25))) for _ in range(10)])
    assert nvidia_tao_tf1.core.utils.get_uid_name(name) == "%s" % name
    assert nvidia_tao_tf1.core.utils.get_uid_name(name) == "%s_1" % name
    assert nvidia_tao_tf1.core.utils.get_uid_name(name) == "%s_2" % name


def test_random_seed_determinism(seed=0):
    """Test random seeds results in deterministic results."""

    def _get_random_numbers():
        """Get random numbers from a few backends."""
        # Keras weight RNG
        tf_rng_op = tf.random.normal([1])
        layer = keras.layers.Dense(1)
        layer.build([10, 1])
        keras_weight_val = layer.get_weights()[0]
        # Keras direct RNG
        keras_val = K.eval(K.random_normal([1]))
        # Tensorflow RNG
        tf_val = K.eval(tf_rng_op)
        # Numpy RNG
        np_val = np.random.rand()
        # Python RNG
        py_val = random.random()
        # Clear the backend state of the framework
        K.clear_session()
        return np.array([keras_weight_val, keras_val, tf_val, np_val, py_val])

    a = _get_random_numbers()
    nvidia_tao_tf1.core.utils.set_random_seed(seed)
    b = _get_random_numbers()
    nvidia_tao_tf1.core.utils.set_random_seed(seed)
    c = _get_random_numbers()
    d = _get_random_numbers()

    np.testing.assert_equal((a == b).any(), False)
    np.testing.assert_array_equal(b, c)
    np.testing.assert_equal((c == d).any(), False)


@pytest.mark.parametrize(
    "string_input, expected_output",
    [("test", "Test"), ("test_name", "TestName"), ("long_test_name", "LongTestName")],
)
def test_to_camel_case(string_input, expected_output):
    assert nvidia_tao_tf1.core.utils.to_camel_case(string_input) == expected_output


@pytest.mark.parametrize(
    "string_input, expected_output",
    [("Test", "test"), ("TestName", "test_name"), ("LongTestName", "long_test_name")],
)
def test_to_snake_case(string_input, expected_output):
    assert nvidia_tao_tf1.core.utils.to_snake_case(string_input) == expected_output


@pytest.mark.parametrize(
    "tag, simple_value", [("loss", 0.0003), ("lr", 0.0001), ("acc", 88.12)]
)
def test_simple_values_from_event_file(tmpdir, tag, simple_value):
    """Test function to get simple values from event file."""
    events_path = str(tmpdir)

    def summary_creator(tag, simple_value):
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=str(tag), simple_value=simple_value)]
        )
        return summary

    summary_writer = tf.compat.v1.summary.FileWriter(events_path)
    summary = summary_creator(tag, simple_value)
    summary_writer.add_summary(summary)
    summary_writer.flush()
    summary_writer.close()

    calculated_dict = nvidia_tao_tf1.core.utils.get_all_simple_values_from_event_file(events_path)
    calculated_values = [calculated_dict[key][0] for key in calculated_dict.keys()]

    assert list(calculated_dict.keys()) == [tag]
    assert np.allclose(calculated_values, [simple_value])
