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
"""Mixup augmentation."""
import numpy as np


class MixupImageDataGenerator():
    """Mixup image generator."""

    def __init__(
        self, generator, directory, batch_size,
        img_height, img_width, color_mode="rgb",
        interpolation="bilinear", alpha=0.2,
        classes=None
    ):
        """Constructor for mixup image data generator.

        Arguments:
            generator (object): An instance of Keras ImageDataGenerator.
            directory (str): Image directory.
            batch_size (int): Batch size.
            img_height (int): Image height in pixels.
            img_width (int): Image width in pixels.
            color_mode (string): Color mode of images.
            interpolation (string): Interpolation method for resize.
            alpha (float): Mixup beta distribution alpha parameter. (default: {0.2})
            `generator` (ImageDataGenerator).(default: {None})
            classes (list): List of input classes
        """
        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha
        # First iterator yielding tuples of (x, y)
        self.generator = generator.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            color_mode=color_mode,
            batch_size=self.batch_size,
            interpolation=interpolation,
            shuffle=True,
            class_mode='categorical',
            classes=classes
        )
        # Number of images across all classes in image directory.
        self.num_samples = self.generator.samples
        self.class_list = classes

    def reset_index(self):
        """Reset the generator indexes array."""
        self.generator._set_index_array()

    def on_epoch_end(self):
        """reset index on epoch end."""
        self.reset_index()

    def reset(self):
        """reset."""
        self.batch_index = 0

    def __len__(self):
        """length."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch."""
        return self.num_samples // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.

        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.num_samples
        if self.num_samples > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # random sample the lambda value from beta distribution.
        if self.alpha > 0:
            # Get a pair of inputs and outputs from the batch and its reversed batch.
            X1, y1 = self.generator.next()
            # in case the dataset has some garbage, the real batch size
            # might be smaller than self.batch_size
            _l = np.random.beta(self.alpha, self.alpha, X1.shape[0])
            _l = np.maximum(_l, 1.0 - _l)
            X_l = _l.reshape(X1.shape[0], 1, 1, 1)
            y_l = _l.reshape(X1.shape[0], 1)
            X2, y2 = np.flip(X1, 0), np.flip(y1, 0)
            # Perform the mixup.
            X = X1 * X_l + X2 * (1 - X_l)
            y = y1 * y_l + y2 * (1 - y_l)
        else:
            # alpha == 0 essentially disable mixup
            X, y = self.generator.next()
        return X, y

    def __iter__(self):
        """iterator."""
        while True:
            return next(self)

    @property
    def num_classes(self):
        """number of classes."""
        return self.generator.num_classes

    @property
    def class_indices(self):
        """class indices."""
        return self.generator.class_indices
