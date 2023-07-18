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

"""Test helper functions for data conversion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _partition, _shard


def test_single_partition():
    """Partition a list of sequences into a single partition."""
    sequences = [[0, 1, 2], [3, 4]]
    assert _partition(sequences, 0, 1) == [[0, 1, 2, 3, 4]]
    assert _partition(sequences, 1, 1) == [[0, 1, 2, 3, 4]]


def test_two_partitions():
    """Partition a list of sequences into 2 partitions."""
    sequences = [[0, 1], [2, 3, 4], [5]]
    assert _partition(sequences, 2, 1) == [[2, 3, 4], [0, 1, 5]]


def test_three_partitions():
    """Partition a list of sequences into 3 partitions."""
    sequences = [[0, 1], [2, 3, 4], [5], [6, 7, 8, 9]]
    assert _partition(sequences, 3, 1) == [[6, 7, 8, 9], [2, 3, 4], [0, 1, 5]]


def test_partitions_divisor():
    """Partition a list of sequences into 2 partitions."""
    sequences = [[0, 1], [2, 3, 4], [5]]
    assert _partition(sequences, 2, 2) == [[2, 3], [0, 1]]


def test_sharding():
    """Shard a list of partitions."""
    partitions = [[0, 1, 2], [3, 4]]
    assert _shard(partitions, 2) == [[[0], [1, 2]], [[3], [4]]]
