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
"""BpNet Model utils."""

import os
import keras


def get_step_from_filename(path):
    """Gets the step number from a checkpoint filename.

    Args:
        path (str): path to the checkpoint.
    Returns:
        int: the step number.
    """
    return int(os.path.basename(path).split('.')[1].split('-')[1])


def get_latest_keras_model(results_dir):
    """Get the latest checkpoint path from a given results directory.

    Parses through the directory to look for the latest keras file
    and returns the path to this file.

    Args:
        results_dir (str): Path to the results directory.

    Returns:
        latest_checkpoint (str): Path to the latest checkpoint.
    """
    trainable_ckpts = []
    for item in os.listdir(results_dir):
        if item.endswith(".hdf5"):
            try:
                step_num = get_step_from_filename(item)
                trainable_ckpts.append(step_num)
            except IndexError:
                continue
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = os.path.join(results_dir, "model.keras-{}.hdf5".format(latest_step))
    return latest_checkpoint


def _print_model_summary_recurse(model):
    """Print model summary recursively.

    Helper function for printing nested models (ie. models that have models as layers).

    Args:
        model: Keras model to print.
    """
    model.summary()
    for l in model.layers:
        if isinstance(l, keras.engine.training.Model):
            print('where %s is' % l.name)
            _print_model_summary_recurse(l)


def print_model_summary(keras_model):
    """Print model summary.

    Args:
        model: Keras model to print.
    """
    _print_model_summary_recurse(keras_model)
