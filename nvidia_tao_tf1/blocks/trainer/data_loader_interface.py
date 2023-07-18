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
"""Data loader interface for use with Trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod, abstractproperty
import io
import sys

from nvidia_tao_tf1.core.coreobject import AbstractTAOObject


class DataLoaderInterface(AbstractTAOObject):
    """Interface that has to be implemented by data loaders to be used with Trainer."""

    @abstractproperty
    def steps(self):
        """Return the number of steps."""

    @abstractproperty
    def batch_size_per_gpu(self):
        """Return the number of examples each batch contains per gpu."""

    @abstractmethod
    def __len__(self):
        """Return the total number of examples that will be produced."""

    def set_shard(self, shard_count=1, shard_index=0, **kwargs):
        """
        Configure the sharding for the current job.

        Args:
            shard_count (int): Number of shards that each dataset will be split into.
            shard_index (int): Index of shard to use [0, shard_count-1].
        """
        pass

    @abstractmethod
    def summary(self, print_fn=None):
        """
        Print a summary of the contents of this data loader.

        Args:
            print_fn (function): Optional function that each line of the summary will be passed to.
                Prints to stdout if not specified.
        """

    @abstractmethod
    def call(self):
        """Produce examples with input features (such as images) and labels.

        Returns:
            (Example / SequenceExample)
        """

    @abstractproperty
    def label_names(self):
        """Get list of label names that this dataloader can produce.

        This set must be unique to this dataloader, it cannot overlap with any other dataloader
        that is being used for training. The purpose of this is to avoid multiple dataloaders
        accidentally feeding into the same task - if this is your goal, use a multi-source
        dataloader.

        Returns:
            (set<str>)
        """

    def __call__(self):
        """Produce examples with input features (such as images) and labels.

        Returns:
            (Example / SequenceExample)
        """
        return self.call()

    def __str__(self):
        """Returns a string summary of this dataloader."""
        if sys.version_info >= (3, 0):
            out = io.StringIO()
        else:
            out = io.BytesIO()
        self.summary(print_fn=lambda string: print(string, file=out))
        return out.getvalue()
