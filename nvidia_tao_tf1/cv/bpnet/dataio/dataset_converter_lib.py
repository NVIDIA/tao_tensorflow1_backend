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

"""Base Class Implementation to convert a dataset to TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import logging
import random

import six
import tensorflow as tf
import tqdm

from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _shard
from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _shuffle


logger = logging.getLogger(__name__)


@six.add_metaclass(ABCMeta)
class DatasetConverter(object):
    """Base Class Implementation to convert a dataset to TFRecords.

    This class needs to be subclassed, and the convert() and
    create_example_proto() methods overridden to do the dataset
    conversion. Splitting of partitions to shards, shuffling and
    writing TFRecords are implemented here, as well as counting
    of written targets.
    """

    @abstractmethod
    def __init__(self, root_data_directory_path, num_partitions, num_shards,
                 output_filename):
        """Initialize the converter.

        Args:
            root_data_directory_path (string): Dataset root directory path.
            num_partitions (int): Number of partitions (folds).
            num_shards (int): Number of shards.
            output_filename (str): Path for the output file.
        """
        self.root_dir = root_data_directory_path
        self.output_partitions = num_partitions
        self.output_shards = num_shards
        self.output_filename = output_filename

        # Set a fixed seed to get a reproducible sequence.
        random.seed(42)

    def convert(self):
        """Do the dataset conversion."""
        # Divide dataset into partitions and shuffle them.
        partitions = self._partition()
        _shuffle(partitions)

        # Shard and write the partitions to tfrecords.
        object_count = self._write_partitions(partitions)

        # Log how many objects got written in total.
        log_str = 'Wrote the following numbers of objects: {}\n'.format(object_count)
        logger.info(log_str)

    def _write_partitions(self, partitions):
        """Shard and write partitions into tfrecords.

        Args:
            partitions (list): A list of list of frame IDs.

        Returns:
            object_count (Counter): The total number of objects per target class.
        """
        # Divide partitions into shards.
        sharded_partitions = _shard(partitions, self.output_shards)

        # Write .tfrecords to disk for each partition and shard.
        # Also count the target objects per partition and over the whole dataset.
        object_count = 0
        for p, partition in enumerate(sharded_partitions):
            partition_object_count = 0
            for s, shard in enumerate(partition):
                shard_object_count = self._write_shard(shard, p, s)
                partition_object_count += shard_object_count

            # Log the count in this partition and increase total
            # object count.
            object_count += partition_object_count

        return object_count

    def _write_shard(self, shard, partition_number, shard_number):
        """Write a single shard into the tfrecords file.

        Note that the dataset-specific part is captured in function
        create_example_proto() which needs to be overridden for each
        specific dataset.

        Args:
            shard (list): A list of frame IDs for this shard.
            partition_number (int): Current partition (fold) index.
            shard_number (int): Current shard index.

        Returns:
            object_count (int): The number of written objects.
        """
        logger.info('Writing partition {}, shard {}'.format(partition_number, shard_number))
        output = self.output_filename

        if self.output_partitions != 0:
            output = '{}-fold-{:03d}-of-{:03d}'.format(output, partition_number,
                                                       self.output_partitions)
        if self.output_shards != 0:
            output = '{}-shard-{:05d}-of-{:05d}'.format(output, shard_number, self.output_shards)

        object_count = 0

        # Store all the data for the shard.
        writer = tf.io.TFRecordWriter(output)
        for frame_id in tqdm.tqdm(shard):

            # Create the Example with all labels for this frame_id.
            example = self._create_example_proto(frame_id)

            # The example might be skipped e.g. due to missing labels.
            if example is not None:
                # TODO: add option to sample/filter data
                # Serialize the example.
                writer.write(example.SerializeToString())
                object_count += 1

        writer.close()
        log_str = 'Wrote the following numbers of objects: {}\n'.format(object_count)
        logger.info(log_str)

        return object_count

    @abstractmethod
    def _partition(self):
        """Return dataset partitions."""
        pass

    @abstractmethod
    def _create_example_proto(self, frame_id):
        """Generate the example for this frame."""
        pass
