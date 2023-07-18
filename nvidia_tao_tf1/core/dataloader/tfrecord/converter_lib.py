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

"""Helper functions for converting datasets to .tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import random
import sys

import tensorflow as tf


def _convert_unicode_to_str(item):
    if sys.version_info >= (3, 0):
        # Python 3 strings are unicode, need to convert to bytes.
        if isinstance(item, str):
            return item.encode("ascii", "ignore")
        return item

    if isinstance(item, unicode):  # pylint: disable=undefined-variable # noqa: F821
        return item.encode("ascii", "ignore")
    return item


def _bytes_feature(*values):
    # Convert unicode data to string for saving to TFRecords.
    values = [_convert_unicode_to_str(value) for value in values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _float_feature(*values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(*values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _partition(sequences, num_partitions, divisor, uneven=False):
    """Partition a list of sequences to approximately equal lengths."""
    num_partitions = max(num_partitions, 1)  # 0 means 1 partition.
    # The sequence with longest frames sits at the top.
    sequences_by_length = sorted(sequences, key=len)
    partitions = [[] for _ in range(num_partitions)]

    while sequences_by_length:
        longest_sequence = sequences_by_length.pop()
        # Add the longest_sequence to the shortest partition.
        smallest_partition = min(partitions, key=len)
        smallest_partition.extend(longest_sequence)

    for partition in partitions:
        for _ in range(len(partition) % divisor):
            partition.pop()

    if num_partitions > 1 and uneven:
        if len(partitions) != num_partitions:
            raise RuntimeError("Check the number of partitions.")

        # Flatten the first num_partitions - 1 into one list.
        flat_list = [item for l in partitions[0 : num_partitions - 1] for item in l]
        # Allocate the first k-1th as the 0th partition and the kth as the 1st partition.
        partitions = [flat_list, partitions[-1]]
        validation_sequence_stats = dict()

        for frame in partitions[-1]:
            if "sequence" in frame.keys():
                sequence_name = frame["sequence"]["name"]
            else:
                sequence_name = frame["sequence_name"]
            if sequence_name is None:
                raise RuntimeError("Sequence name is None.")
            if sequence_name in validation_sequence_stats.keys():
                validation_sequence_stats[sequence_name] += 1
            else:
                validation_sequence_stats[sequence_name] = 1

        pp = pprint.PrettyPrinter(indent=4)
        print("%d training frames " % (len(partitions[0])))
        print("%d validation frames" % (len(partitions[-1])))
        print("Validation sequence stats:")
        print("Sequence name: #frame")
        pp.pprint(validation_sequence_stats)

    return partitions


def _shard(partitions, num_shards):
    """Shard each partition."""
    num_shards = max(num_shards, 1)  # 0 means 1 shard.
    shards = []
    for partition in partitions:
        result = []
        shard_size = len(partition) // num_shards
        for i in range(num_shards):
            begin = i * shard_size
            end = (i + 1) * shard_size if i + 1 < num_shards else len(partition)
            result.append(partition[begin:end])
        shards.append(result)
    return shards


def _shuffle(partitions):
    """Shuffle each partition independently."""
    for partition in partitions:
        random.shuffle(partition, random.random)
