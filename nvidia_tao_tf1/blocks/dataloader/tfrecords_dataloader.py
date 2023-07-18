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

"""TFRecords Dataloader Example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from nvidia_tao_tf1.blocks.dataloader.dataloader import DataLoader


class TFRecordsDataLoader(DataLoader):
    """Dataloader for loading TFRecords based datasets."""

    TRAIN = "train"
    VAL = "val"
    SUPPORTED_MODES = [TRAIN, VAL]

    def __init__(
        self,
        train_records_folder,
        val_records_folder,
        input_shape,
        batch_size,
        buffer_size=None,
        repeats=1,
    ):
        """__init__ method.

        Args:
            train_records_folder (string): The path where the training TFRecords are.
            val_records_folder (string): The path where the validation TFRecords are.
            input_shape ([int, int]): The shape size expected to resize the images to.
            batch_size (int): Size of each batch to be fed for training.
            buffer_size (int): Size of the shuffle buffer for feeding in data.
                Default is 8 * batch_size.
            repeats (int): Number of times to repeat the dataset. Default 1.
        """
        super(TFRecordsDataLoader, self).__init__()

        self._train_records_folder = train_records_folder
        self._val_records_folder = val_records_folder
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._repeats = repeats

        if not self._buffer_size:
            self._buffer_size = 8 * self._batch_size

    def __call__(self, mode):
        """__call__ method.

        Args:
            mode (string): Specifies whether to get the training or validation dataset.
                Must be one of 'train' or 'val'
        Returns:
            (function) The function associated with the dataset and split.
        """
        if mode not in self.SUPPORTED_MODES:
            raise "Mode must be one of {}.".format(self.SUPPORTED_MODES)

        if mode == self.TRAIN:
            return self._get_train_content

        return self._get_val_content

    def _get_train_content(self):
        """Gets the training contents inside a TFDataset."""
        dataset = self._create_dataset(self._train_records_folder)
        if self._repeats:
            dataset = dataset.repeat(self._repeats)
        return dataset

    def _get_val_content(self):
        """Gets the validation contents inside a TFDataset."""
        return self._create_dataset(self._val_records_folder)

    def _create_dataset(self, folder):
        """Method to process the files and return a dataset object."""
        files = [os.path.join(folder, x) for x in os.listdir(folder)]
        dataset = tf.data.TFRecordDataset(files)
        dataset = self._process_tfrecord_dataset(dataset)
        dataset = dataset.map(self._process_record)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.shuffle(self._buffer_size)
        return dataset

    def _process_tfrecord_dataset(self, dataset):
        """Method placeholder for dataset processing."""
        return dataset

    def _process_record(self, record_example):
        """Method placeholder for single record processing."""
        raise NotImplementedError("Implement me!")
