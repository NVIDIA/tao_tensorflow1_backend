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
"""Test for sparse tensor generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from nvidia_tao_tf1.blocks.multi_source_loader.processors import sparse_generators

tensor0_indices = np.array(
    [
        [2, 0, 0, 0],
        [2, 0, 0, 1],
        [2, 0, 1, 0],
        [2, 0, 1, 1],
        [2, 0, 2, 0],
        [2, 0, 2, 1],
        [2, 1, 0, 0],
        [2, 1, 0, 1],
        [2, 1, 1, 0],
        [2, 1, 1, 1],
        [2, 1, 2, 0],
        [2, 1, 2, 1],
        [3, 1, 0, 0],
        [3, 1, 0, 1],
        [3, 1, 1, 0],
        [3, 1, 1, 1],
        [3, 1, 2, 0],
        [3, 1, 2, 1],
    ]
)
tensor0_values = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
)

tensor1_indices = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0]]
)
tensor1_values = np.array([0, 1, 2, 3, 4, 5, 6])


def test_sparse_iterator():
    accumulated_sub_indices = []
    accumulated_values = []
    accumulated_indices = []

    expected_values = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
    ]

    expected_indices = [
        [
            [2, 0, 0, 0],
            [2, 0, 0, 1],
            [2, 0, 1, 0],
            [2, 0, 1, 1],
            [2, 0, 2, 0],
            [2, 0, 2, 1],
        ],
        [
            [2, 1, 0, 0],
            [2, 1, 0, 1],
            [2, 1, 1, 0],
            [2, 1, 1, 1],
            [2, 1, 2, 0],
            [2, 1, 2, 1],
        ],
        [
            [3, 1, 0, 0],
            [3, 1, 0, 1],
            [3, 1, 1, 0],
            [3, 1, 1, 1],
            [3, 1, 2, 0],
            [3, 1, 2, 1],
        ],
    ]
    expected_sub_indices = [[2, 0], [2, 1], [3, 1]]

    for sub_index, values, indices in sparse_generators.sparse_generator(
        2, tensor0_values, tensor0_indices
    ):
        accumulated_sub_indices.append(sub_index)
        accumulated_values.append(values)
        accumulated_indices.append(indices)
    assert accumulated_sub_indices == expected_sub_indices
    assert accumulated_values == expected_values
    assert accumulated_indices == expected_indices


def test_matching_indices():
    t0_values_array = []
    t1_values_array = []
    t0_indices_array = []
    t1_indices_array = []

    expected_t0_values = [
        [],
        [],
        [],
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [],
        [12, 13, 14, 15, 16, 17],
    ]

    expected_t1_values = [[0], [1], [2], [3], [4], [5], [6]]

    expected_t0_indices = [
        [],
        [],
        [],
        [
            [2, 0, 0, 0],
            [2, 0, 0, 1],
            [2, 0, 1, 0],
            [2, 0, 1, 1],
            [2, 0, 2, 0],
            [2, 0, 2, 1],
        ],
        [
            [2, 1, 0, 0],
            [2, 1, 0, 1],
            [2, 1, 1, 0],
            [2, 1, 1, 1],
            [2, 1, 2, 0],
            [2, 1, 2, 1],
        ],
        [],
        [
            [3, 1, 0, 0],
            [3, 1, 0, 1],
            [3, 1, 1, 0],
            [3, 1, 1, 1],
            [3, 1, 2, 0],
            [3, 1, 2, 1],
        ],
    ]

    expected_t1_indices = [
        [[0, 0, 0]],
        [[1, 0, 0]],
        [[1, 1, 0]],
        [[2, 0, 0]],
        [[2, 1, 0]],
        [[3, 0, 0]],
        [[3, 1, 0]],
    ]
    for (
        _,
        t0_values,
        t0_indices,
        t1_values,
        t1_indices,
    ) in sparse_generators.matching_indices_generator(
        2, tensor0_values, tensor0_indices, tensor1_values, tensor1_indices
    ):
        t0_values_array.append(t0_values)
        t1_values_array.append(t1_values)

        t0_indices_array.append(t0_indices)
        t1_indices_array.append(t1_indices)

    assert t0_values_array == expected_t0_values
    assert t1_values_array == expected_t1_values
    assert t0_indices_array == expected_t0_indices
    assert t1_indices_array == expected_t1_indices


def test_non_matching_indices():

    tensor0 = [
        ([0, 0, 1, 0], "Car"),
        ([0, 0, 7, 0], "Boat"),
        ([0, 0, 7, 1], "RV"),
        ([0, 0, 9, 0], "Car"),
        ([0, 0, 10, 0], "Car"),
        ([0, 0, 16, 0], "Car"),
        ([0, 0, 17, 0], "Car"),
        ([1, 0, 0, 0], "Car"),
        ([1, 0, 1, 0], "Car"),
        ([1, 0, 34, 0], "Car"),
    ]
    tensor0_indices = np.array([t[0] for t in tensor0])
    tensor0_values = np.array([t[1] for t in tensor0])

    tensor1 = [
        ([0, 0, 0, 0], "object_5"),
        ([0, 0, 1, 0], "object_2"),
        ([0, 0, 9, 0], "object_6"),
        ([0, 0, 10, 0], "object_4"),
        ([1, 0, 34, 0], "object_2"),
        ([1, 0, 35, 0], "object_2"),
        ([1, 0, 35, 1], "object_3"),
    ]
    tensor1_indices = np.array([t[0] for t in tensor1])
    tensor1_values = np.array([t[1] for t in tensor1])

    expected_accumulation = [
        # Only the second tensor has values for first sub index.
        ([0, 0, 0], [], [], [[0, 0, 0, 0]], ["object_5"]),
        # Both tensors have one value for this sub index
        ([0, 0, 1], [[0, 0, 1, 0]], ["Car"], [[0, 0, 1, 0]], ["object_2"]),
        # First tensor has two values, second tensor no values for this sub index
        ([0, 0, 7], [[0, 0, 7, 0], [0, 0, 7, 1]], ["Boat", "RV"], [], []),
        ([0, 0, 9], [[0, 0, 9, 0]], ["Car"], [[0, 0, 9, 0]], ["object_6"]),
        ([0, 0, 10], [[0, 0, 10, 0]], ["Car"], [[0, 0, 10, 0]], ["object_4"]),
        # First tensor has one value, second tensor has no values for this sub index
        ([0, 0, 16], [[0, 0, 16, 0]], ["Car"], [], []),
        ([0, 0, 17], [[0, 0, 17, 0]], ["Car"], [], []),
        ([1, 0, 0], [[1, 0, 0, 0]], ["Car"], [], []),
        ([1, 0, 1], [[1, 0, 1, 0]], ["Car"], [], []),
        ([1, 0, 34], [[1, 0, 34, 0]], ["Car"], [[1, 0, 34, 0]], ["object_2"]),
        # First tensor has no values, second tensor has two values for this sub-index
        ([1, 0, 35], [], [], [[1, 0, 35, 0], [1, 0, 35, 1]], ["object_2", "object_3"]),
    ]
    accumulation = []
    for (
        sub_index,
        t0_values,
        t0_indices,
        t1_values,
        t1_indices,
    ) in sparse_generators.matching_indices_generator(
        3, tensor0_values, tensor0_indices, tensor1_values, tensor1_indices
    ):
        accumulation.append((sub_index, t0_indices, t0_values, t1_indices, t1_values))
    assert accumulation == expected_accumulation
