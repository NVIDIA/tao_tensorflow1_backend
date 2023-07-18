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
"""Data source for tfrecords based data files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import lru_cache
import os

import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.sources.data_source import (
    DataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.lane_label_parser import (
    LaneLabelParser,
)
from nvidia_tao_tf1.core.coreobject import save_args


class TFRecordsDataSource(DataSource):
    """DataSource for reading examples from TFRecords files."""

    @save_args
    def __init__(
        self,
        tfrecord_path,
        image_dir,
        height,
        width,
        channels,
        subset_size,
        should_normalize_labels=True,
        **kwargs
    ):
        """
        Construct a TFRecordsDataSource.

        Args:
            tfrecord_path (str): Path to a tfrecords file or a list of paths to tfrecords files.
            image_dir (str): Path to directory where images referenced by examples are stored.
            height (int): Height of images and labels stored in this dataset.
            width (int): Width of images and labels stored in this dataset.
            channels (int): Number of channels for images stored in this dataset.
            subset_size (int): Number of images from tfrecord_path to use.
            should_normalize_labels(bool): Whether or not the datasource should normalize the label
                coordinates.
        """
        super(TFRecordsDataSource, self).__init__(**kwargs)
        if not isinstance(tfrecord_path, list):
            tfrecord_path = [tfrecord_path]

        for path in tfrecord_path:
            if not os.path.isfile(path):
                raise ValueError(
                    "No dataset tfrecords file found at path: '{}'".format(path)
                )

        if not os.path.isdir(image_dir):
            raise ValueError(
                "No dataset image directory found at path: '{}'".format(image_dir)
            )

        self.tfrecord_path = tfrecord_path
        self.image_dir = image_dir
        # TODO(vkallioniemi): Remove channels - not used at the moment.
        self.channels = channels
        self.height = height
        self.width = width
        self.subset_size = subset_size
        self._num_shards = 1
        self._shard_id = 0
        self._pseudo_sharding = False
        self._shuffle = False
        self._shuffle_buffer_size = 10000
        self._should_normalize_labels = should_normalize_labels
        self._max_height = height
        self._max_width = width

    @lru_cache()
    def __len__(self):
        """Return the number of examples in the underlying tfrecords file."""
        count = 0
        for path in self.tfrecord_path:
            for _ in tf.compat.v1.python_io.tf_record_iterator(path):
                count += 1

        if self.subset_size is None:
            self.subset_size = count

        if self.subset_size > 0:
            return min(count, self.subset_size)
        return count

    def set_image_properties(self, max_image_width, max_image_height):
        """Overrides the maximum image width and height of this source."""
        self._max_height = max_image_height
        self._max_width = max_image_width

    def get_image_properties(self):
        """Returns the maximum width and height of images for this source."""
        return self._max_width, self._max_height

    def supports_shuffling(self):
        """Whether this source can do its own shuffling."""
        return True

    def set_shuffle(self, buffer_size):
        """Enables shuffling on this data source."""
        self._shuffle = True
        self._shuffle_buffer_size = buffer_size

    def supports_sharding(self):
        """Whether this source can do its own sharding."""
        return True

    def set_shard(self, num_shards, shard_id, pseudo_sharding=False):
        """
        Sets the sharding configuration of this source.

        Args:
           num_shards (int):  The number of shards.
           shard_id  (int):   Shard id from 0 to num_shards - 1.
           pseudo_sharding (bool) if True, then data is not actually sharded, but different shuffle
               seeds are used to differentiate shard batches.
        """
        self._num_shards = num_shards
        self._shard_id = shard_id
        self._pseudo_sharding = pseudo_sharding

    @property
    def parse_example(self):
        """Load features off disk."""
        parser = LaneLabelParser(
            image_dir=self.image_dir,
            extension=self.extension,
            height=self.height,
            width=self.width,
            max_height=self._max_height,
            max_width=self._max_width,
            should_normalize_labels=self._should_normalize_labels,
        )

        return lambda dataset: dataset.map(parser)

    def call(self):
        """Return a tf.data.Dataset for this data source."""
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        if self.subset_size > 0:
            dataset = dataset.take(self.subset_size)

        if not self._pseudo_sharding and self._num_shards > 1:
            dataset = dataset.shard(self._num_shards, self._shard_id)

        # Note: we do shuffling here, so that the shuffle buffer only needs to
        # store raw tfrecords data (instead of fully parsed examples)
        if self._shuffle:
            dataset = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(
                    buffer_size=min(len(self), self._shuffle_buffer_size), count=None
                )
            )
        return dataset
