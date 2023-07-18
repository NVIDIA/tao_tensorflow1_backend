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

"""Test script for training a model in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from nvidia_tao_tf1.core.training.features import enable_deterministic_training  # noqa: E402
from nvidia_tao_tf1.core.utils.path_utils import expand_path


def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def _make_model():
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def _make_data_loaders():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return train_ds, test_ds


def main(model_path, deterministic, seed):
    """Train a model in eager mode."""
    if deterministic:
        enable_deterministic_training()

    _set_seeds(seed)

    results_dir = os.path.dirname(model_path)
    if not os.path.exists(expand_path(results_dir)):
        os.mkdir(expand_path(results_dir))
    if os.path.exists(expand_path(model_path)):
        raise ValueError("Model already exists at path `{}`".format(model_path))

    calculate_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    model = _make_model()

    train_ds, test_ds = _make_data_loaders()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = calculate_loss(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = calculate_loss(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 2
    for epoch in range(EPOCHS):
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )

    model.save(model_path)


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
