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

"""Test script for training a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import random

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.core.training.features import enable_deterministic_training
from nvidia_tao_tf1.core.utils.path_utils import expand_path


def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def main(model_path, deterministic, seed):
    """Train a model."""
    if deterministic:
        enable_deterministic_training()

    _set_seeds(seed)

    results_dir = os.path.dirname(model_path)
    if not os.path.exists(expand_path(results_dir)):
        os.mkdir(expand_path(results_dir))
    if os.path.exists(expand_path(model_path)):
        raise ValueError("Model already exists at path `{}`".format(model_path))

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=5)
    model.save(model_path)

    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool for models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        action="store_true",
        help="Use deterministic CuDNN kernels to produce the same results every time.",
    )
    parser.add_argument(
        "-o", "--output_model", type=str, help="Model will be stored in the given path."
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    main(model_path=args.output_model, deterministic=args.deterministic, seed=args.seed)
