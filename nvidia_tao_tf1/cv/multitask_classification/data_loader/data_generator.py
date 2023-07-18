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

"""IVA MultiTask model data generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator, Iterator

import numpy as np
import pandas as pd


class SingleDirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory (no subdir) on disk.

    # Arguments
        directory: Path to the directory to read images from.
            All images should be under this directory.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        class_table: A pandas table containing the true label of the images
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, directory, image_data_generator,
                 class_table, target_size=(256, 256),
                 color_mode='rgb', batch_size=32, shuffle=True,
                 seed=None, data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False, subset=None, interpolation='nearest'):
        """init function for the Iterator.

        # Code largely from Keras DirectoryIterator
        # https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L1507
        """
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.subset = subset

        self._generate_class_mapping(class_table)
        self.num_tasks = len(self.tasks_header)
        self.data_df = class_table

        # print total # of images with K tasks
        print('Found %d images with %d tasks (%s)' %
              (self.samples, self.num_tasks, self.class_dict))

        super(SingleDirectoryIterator, self).__init__(self.samples,
                                                      batch_size,
                                                      shuffle,
                                                      seed)

    def _generate_class_mapping(self, class_table):
        """Prepare task dictionary and class mapping."""
        self.filenames = class_table.iloc[:, 0].values
        self.samples = len(self.filenames)
        self.tasks_header = sorted(class_table.columns.tolist()[1:])
        self.class_dict = {}
        self.class_mapping = {}
        for task in self.tasks_header:
            unique_vals = sorted(class_table.loc[:, task].unique())
            self.class_dict[task] = len(unique_vals)
            self.class_mapping[task] = dict(zip(unique_vals, range(len(unique_vals))))
        # convert class dictionary to a sorted tolist
        self.class_dict_list_sorted = sorted(self.class_dict.items(), key=lambda x: x[0])
        self.class_values_list_sorted = list(zip(*list(self.class_dict_list_sorted)))[1]

    def _get_batches_of_transformed_samples(self, index_array):
        """Prepare input and the groundtruth for a batch of data."""
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels

        # one-hot encoding
        batch_y = []
        for _, cls_cnt in self.class_dict_list_sorted:
            batch_y.append(np.zeros((len(index_array), cls_cnt), dtype=K.floatx()))

        index = 0
        for _, row in self.data_df.iloc[index_array, :].iterrows():
            for i, (c, _) in enumerate(self.class_dict_list_sorted):
                batch_y[i][index, self.class_mapping[c][row[c]]] = 1.
            index += 1

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class MultiClassDataGenerator(ImageDataGenerator):
    """Generate batches of tensor image data with real-time data augmentation.

    Code based on ImageDataGenerator in Keras.
    """

    def flow_from_singledirectory(self, directory, label_csv,
                                  target_size=(256, 256), color_mode='rgb',
                                  batch_size=32, shuffle=True, seed=None,
                                  save_to_dir=None,
                                  save_prefix='',
                                  save_format='png',
                                  follow_links=False,
                                  subset=None,
                                  interpolation='nearest'):
        """Get flow from a single directory with all labels in a separate CSV."""
        df = pd.read_csv(label_csv)
        return SingleDirectoryIterator(
            directory, self, df,
            target_size=target_size, color_mode=color_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)
