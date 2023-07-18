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

"""TLT base data sequence."""

from abc import ABC, abstractmethod
from keras.utils import Sequence
import numpy as np
from PIL import Image


class BaseDataSequence(ABC, Sequence):
    """Abstract class for TLT network data sequence.

    There should be another level of abstraction for specific tasks like detection etc.

    To use dataloader:
        1. call __init__(configs)
        2. call add_source(image_folder, label_folder) to add sources
        3. Use data generator in keras model.fit_generator()

    Functions below must be implemented in derived classes:
        1. __init__
        2. _add_source
        3. _preprocessing
        4. _load_gt_label
        5. __getitem__
        6. __len__
    """

    @abstractmethod
    def __init__(self, dataset_config, augmentation_config=None, batch_size=10, is_training=True):
        """init function."""
        self.n_samples = 0
        pass

    @abstractmethod
    def _add_source(self, image_folder, label_folder):
        """add_source."""
        pass

    @abstractmethod
    def _preprocessing(self, image, label, output_img_size):
        """Perform augmentation."""
        pass

    @abstractmethod
    def _load_gt_label(self, label_path):
        """Load GT label from file."""
        pass

    def _load_gt_image(self, image_path):
        """Load GT image from file."""
        img = Image.open(image_path)
        if img.mode != "I":
            img = img.convert('RGB')
            return np.array(img).astype(np.float32)
        # If it is a 16-bit image
        # PIL convert to RGB will clip to 8-bit
        # Hence we achieve this in numpy
        img = np.array(img).astype(np.float32)
        target_shape = img.shape + (3,)
        img = img.reshape(img.shape + (1,))
        img = np.broadcast_to(img, target_shape)
        return img
