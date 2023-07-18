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
"""DataSource interface for accessing datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import io
import sys

import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader import processors
from nvidia_tao_tf1.core.coreobject import AbstractTAOObject


class DataSource(AbstractTAOObject):
    """
    Interface for adding new types of datasets.

    Datasets are stored in different on-disk formats (e.g. tfrecords, sqlite). The DataSource
    interface is meant to normalize/standardize datasets to output Example namedtuples so that the
    rest of the training pipeline does not need to know about the on-disk differences.

    The interface implementor is expected to provide 3 properties. These 3 properties are accessed
    in this order to ensure memory efficient loading of the data:

    1. dataset property is used to load metadata data in source specific format. The DataLoader
       makes no assumptions about the structure of the data yielded by this dataset.
    2. dataset.apply is called on the processor returned by the  parse_example property.
       The DataLoader assumes that the processor returns a dataset composed of individual
       examples that can still be in a source specific format (e.g. tfrecord.)

    * Example is currently the namedtuple found in types.py, but will be transitioned to a more
      flexible/generic format documented in this design doc:
      https://docs.google.com/document/d/1qXBUvRt-umAfkHB3KOiUDDXzHD986HRhdtGAZ29vQpQ

    TODO(vkallioniemi): Update ^^ docs when new Example format is adopted.
    """

    def __init__(self, preprocessing=None, sample_ratio=1.0, extension=None):
        """
        Constructs a data source.

        Args:
            preprocessing (Pipeline or list[Processor]): Optional preprocessing processors specific
                to this dataset. Defaults to no preprocessing.
            sample_ratio (float): Optional frequency at which a sample from this data source is
                picked for inclusion in a batch. Defaults to 1.0.
            extension (str): Extension of the data files. E.g., '.fp16'.
        """
        super(DataSource, self).__init__()
        if preprocessing is None:
            preprocessing = processors.Pipeline([])

        if sample_ratio < 0:
            raise ValueError("Sample ratio {} cannot be < 0.".format(sample_ratio))

        self.preprocessing = preprocessing
        self.sample_ratio = sample_ratio
        self.extension = extension

    @abstractmethod
    def call(self):
        """Build a dataset.

        Returns:
            (tf.data.Dataset): Dataset that produces source specific pieces of data.
        """
        raise NotImplementedError("DataSource.call not implemented.")

    def __call__(self):
        """Build a dataset.

        Returns:
            (tf.data.Dataset): Dataset that produces source specific pieces of data.
        """
        return self.call()

    @abstractmethod
    def __len__(self):
        """Returns the number of examples in this dataset."""
        raise NotImplementedError("DataSource.__len__ not implemented.")

    def __str__(self):
        """Return a string representation of this data source."""
        if sys.version_info >= (3, 0):
            out = io.StringIO()
        else:
            out = io.BytesIO()
        self.summary(print_fn=lambda string: print(string, file=out))
        return out.getvalue()

    def summary(self, print_fn=None):
        """
        Print a summary of the contents of this data source.

        Args:
            data_loader (DataLoader): Data loader to summarize.
        """
        if print_fn is None:
            print_fn = print

        print_fn("  - samples: {}".format(len(self)))
        print_fn("  - sample ratio: {}".format(self.sample_ratio))
        print_fn("  - extension: {}".format(self.extension))
        if self.preprocessing:
            print_fn("  - preprocessing:")
            for processor in self.preprocessing:
                print_fn("    - {}".format(processor))

    @property
    def parse_example(self):
        """
        Return processor/function that can be applied to a dataset to parse it.

        The function returned must have this signature:
            `def parser(dataset: tf.data.Dataset[R]) -> tf.data.Dataset[T]`,
            where types R and T are DataSource implementation specific.

        After this processor is applied, dataset can still be in source specific format, but
        each item yielded by the dataset is expected to be an indivdual example.
        """
        # Deprecated and will be removed.  DataSource.dataset will be expected to return parsed
        # examples.
        return None

    def supports_sharding(self):
        """Whether this source can do its own sharding."""
        return False

    def set_shard(self, num_shards, shard_id, pseudo_sharding=False):
        """
        Sets the sharding configuration of this source.

        Args:
           num_shards (int):  The number of shards.
           shard_id  (int):   Shard id from 0 to num_shards - 1.
           pseudo_sharding (bool) if True, then data is not actually sharded, but different shuffle
               seeds are used to differentiate shard batches.
        """
        raise NotImplementedError()

    def supports_shuffling(self):
        """Whether this source can do its own shuffling."""
        return False

    def set_shuffle(self, buffer_size):
        """Enables shuffling on this data source."""
        raise NotImplementedError()

    def set_sequence_length(self, sequence_length):
        """Sets the sequence length (number of frames in sequence)."""
        pass

    def supports_temporal_batching(self):
        """Whether this source does its own temporal batching."""
        return False

    def initialize(self):
        """Called by data loaders after all configuration is done."""
        pass

    def get_image_properties(self):
        """Returns the maximum width and height image for this data source."""
        return 0, 0

    def set_image_properties(self, max_image_width, max_image_height):
        """Overrides the max image width and height of this data source for padding purposes."""
        pass

    @property
    def image_extension(self):
        """Returns the image file extension."""
        return self.extension

    @property
    def image_dtype(self):
        """The default dtype of images for this data source.

        Returns the dtype of images for this data source.

        Return:
            (tf.dtypes.Dtype) Returned dtype of images for this data source.
        """
        if self.image_extension in [".jpeg", ".jpg", ".png"]:
            return tf.uint8

        if self.image_extension == ".fp16":
            return tf.float16

        return tf.float32
