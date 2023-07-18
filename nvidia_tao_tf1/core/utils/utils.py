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
"""Modulus utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import errno
import glob
import os
import random
import re
import threading

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2


def find_latest_keras_model_in_directory(model_directory, model_prefix="model"):
    """Method to find the latest model in a given model directory.

    Args:
        model_directory (str): absolute path to the model directory.
        model_prefix (str): File prefix used to look up hdf5 files. Defaults to "model".
    """

    def extract_model_number(files):
        s = re.findall(r"{}.keras-(\d+).hdf5".format(model_prefix), files)
        if not s:
            raise ValueError("No Keras model found in {}".format(model_directory))
        return int(s[0]) if s else -1, files

    model_files = glob.glob(os.path.join(model_directory, "*.hdf5"))
    if not model_files:
        return None
    latest_model = max(model_files, key=extract_model_number)

    return latest_model


def get_uid(base_name):
    """Return a unique ID."""
    get_uid.lock.acquire()
    if base_name not in get_uid.seqn:
        get_uid.seqn[base_name] = 0
    uid = get_uid.seqn[base_name]
    get_uid.seqn[base_name] += 1
    get_uid.lock.release()
    return uid


def get_uid_name(base_name):
    """
    Get unique name.

    Get a unique name for an object, structured as the specified
    base name appended by an integer.

    Args:
        base_name (str): Base name.
    Returns:
        (str): Unique name for the given base name.
    """
    uid = get_uid(base_name)
    return "%s%s" % (base_name, "_%s" % uid if uid > 0 else "")


get_uid.seqn = {}
get_uid.lock = threading.Lock()


def set_random_seed(seed):
    """
    Set random seeds.

    This sets the random seeds of Python, Numpy and TensorFlow.

    Args:
        seed (int): seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def to_camel_case(snake_str):
    """Convert a name to camel case.

    For example ``test_name`` becomes ``TestName``.

    Args:
        name (str): name fo convert.
    """
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


def to_snake_case(name):
    """Convert a name to snake case.

    For example ``TestName`` becomes ``test_name``.

    Args:
        name (str): name fo convert.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def test_session(allow_soft_placement=False, log_device_placement=False):
    """Get a tensorflow session with all graph optimizations turned off for deterministic testing.

    Using a regular session does not nessecarily guarantee explicit device placement to be placed
    on the requisted device. This test session makes sure placement is put on requested devices.

    Args:
        soft_device_placement (bool): Whether soft placement is allowed.
        log_device_placement (bool): Whether log out the device placement for each tensor.
    """
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.opt_level = -1
    config.graph_options.rewrite_options.constant_folding = (
        rewriter_config_pb2.RewriterConfig.OFF
    )
    config.graph_options.rewrite_options.arithmetic_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF
    )
    config.allow_soft_placement = allow_soft_placement
    config.log_device_placement = log_device_placement
    sess = tf.compat.v1.Session(config=config)
    return sess


def summary_from_value(tag, value, scope=None):
    """Generate a manual simple summary object with a tag and a value."""
    summary = tf.compat.v1.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    if scope:
        summary_value.tag = "{}/{}".format(scope, tag)
    else:
        summary_value.tag = tag
    return summary


def get_all_simple_values_from_event_file(event_path):
    """Retrieve all 'simple values' from event file into a nested dict.

    Args:
        event_path (str): path to a directory holding a `events.out.*` file.
    Returns:
        A nested dictionary containing all the simple values found in the events file, nesting
            first the tag (str) and then the step (int) as keys.
    """
    event_files = glob.glob("%s/events.out.*" % event_path)
    assert len(event_files) == 1
    values_dict = defaultdict(dict)
    for e in tf.compat.v1.train.summary_iterator(path=event_files[0]):
        for v in e.summary.value:
            if v.HasField("simple_value"):
                values_dict[v.tag][e.step] = v.simple_value
    return values_dict


def recursive_map_dict(d, fn, exclude_fields=None):
    """Applies a function recursively on a dictionary's non-dict values.

    Note: no deep-copies take place: values are changed in-place, but the dict is returned
    regardless.

    Args:
        d (``dict`` or value): input dictionary or value. If it is not a ``dict``, ``fn``
            will be applied to it.
        fn (function): function to be applied  to ``d`` if ``d`` is not a ``dict``.
        exclude_fields (list): List of fields to exclude when mapping sparify operation to
                tensors i.e. not sparsify example.labels['field'] for field in exclude_fields.
    """
    if exclude_fields is None:
        exclude_fields = []
    if not isinstance(exclude_fields, list):
        raise ValueError("exclude_fields arg must be a list!")
    if isinstance(d, dict):
        for key in d:
            if key in exclude_fields:
                continue
            d[key] = recursive_map_dict(d[key], fn=fn)
        return d
    return fn(d)


def mkdir_p(new_path):
    """Makedir, making also non-existing parent dirs."""
    try:
        os.makedirs(new_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(new_path):
            pass
        else:
            raise
