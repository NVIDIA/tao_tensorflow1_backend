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
"""Generator function for iterating over sparse tensors with corresponding indices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def sparse_generator(index_prefix_size, values_array, indices_array):
    """Function which returns a generator for iterating through elements of a sparse tensor.

    Iterates through elements with matching sub indices.

    Args:
        index_prefix_size (int): The number of indices, starting with the 0th to include in the sub
            indices.
        values_array(np.array): Values array for the sparse tensor.
        indices_array(np.array): Indices array for the sparse tensor.

    Returns:
        Generator which returns the current sub-index, indices and corresponding values for that
        sub index.

    Example:
        Iterate through the vertices for each polygon in a 4D tensor of polygons with the
        following indices B, S, V, C where B=Batch, S=Shape, V=Vertex and C=Coordinate

        indices_array =
            [[2, 0, 0, 0],
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
             [3, 1, 2, 1]])

        values_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        generator = sparse_generator(2, values_array, indices_array)

        sub_index, values, indices = next(generator)
        # sub_index [2, 0]
        # values [0, 1, 2, 3, 4, 5]
        # indices [[2, 0, 0, 0], [2, 0, 0, 1],
        #          [2, 0, 1, 0], [2, 0, 1, 1],
        #          [2, 0, 2, 0], [2, 0, 2, 1]]
        #
        sub_index, values, indices = next(generator)
        # sub index  [2, 1]
        # values  [6, 7, 8, 9, 10, 11]
        # indices [[2, 1, 0, 0], [2, 1, 0, 1],
        #          [2, 1, 1, 0], [2, 1, 1, 1],
        #          [2, 1, 2, 0], [2, 1, 2, 1]]
        #
        sub_index, values, indices = next(generator)
        # sub index  [3, 1]
        # values  [12, 13, 14, 15, 16, 17]
        # indices [[3, 1, 0, 0], [3, 1, 0, 1],
        #          [3, 1, 1, 0], [3, 1, 1, 1],
        #          [3, 1, 2, 0], [3, 1, 2, 1]]
    """
    index = 0
    while index < values_array.size:
        curr_sub_index = indices_array[index][0:index_prefix_size]
        values = []
        indices = []
        while index < values_array.size:
            row_sub_index = indices_array[index][0:index_prefix_size]
            if np.array_equal(curr_sub_index, row_sub_index):
                values.append(values_array[index])
                indices.append(list(indices_array[index]))
                index += 1
            else:
                break
        yield list(curr_sub_index), values, indices


def matching_indices_generator(
    index_prefix_size, sparse0_values, sparse0_indices, sparse1_values, sparse1_indices
):
    """Function which returns a generator for iterating through elements in two sparse tensors.

    Iterates through elements with common subindices across two tensors.

    Args:
        index_prefix_size (int): The number of indices, starting with the 0th, to include in the sub
            indices.
        sparse0_values(np.array): Values array for the first sparse tensor.
        sparse0_indices(np.array): Indices array for the first sparse tensor.
        sparse1_values(np.array): Values array for the second sparse tensor.
        sparse1_indices(np.array): Indices array for the second sparse tensor.

    Returns:
        Generator which returns the current sub-index, indices and corresponding values for that
        sub index for two sparse tensor.

    Example usage:
        Iterate through a sparse tensor of polygon vertices and get corresponding class ids for
        the same polygons.

        for sub_index, polygon_vertices, polygon_indices, class_values, class_indices in
            matching_indices_generator(3, polygon_vertices_values, polygon_vertices_indices,
                                       class_values, class_indices):
            # sub_index: holds the common sub-index shared by the polygon vertices and class values
            # polygon_vertices: contains the vertices for a polygon
            # polygon_indices: contains the indices for those vertices
            # class_values: contains the corresponding class(es) for the polygon
            # class_indices: contains the indices for those classes.
    """
    end = (None, None, None)
    sparse0_iterator = sparse_generator(
        index_prefix_size, sparse0_values, sparse0_indices
    )
    sparse1_iterator = sparse_generator(
        index_prefix_size, sparse1_values, sparse1_indices
    )
    sparse0_index, sparse0_values, sparse0_indices = next(sparse0_iterator, end)
    sparse1_index, sparse1_values, sparse1_indices = next(sparse1_iterator, end)

    def _index_less_than(index_0, index_1):
        for i in range(len(index_0)):
            if index_0[i] == index_1[i]:
                continue
            else:
                return index_0[i] < index_1[i]
        return False

    while sparse0_index is not None or sparse1_index is not None:
        if sparse0_index is None:
            yield sparse1_index, [], [], sparse1_values, sparse1_indices
            sparse1_index, sparse1_values, sparse1_indices = next(sparse1_iterator, end)
        elif sparse1_index is None:
            yield sparse0_index, sparse0_values, sparse0_indices, [], []
            sparse0_index, sparse0_values, sparse0_indices = next(sparse0_iterator, end)
        elif np.array_equal(sparse0_index, sparse1_index):
            yield sparse0_index, sparse0_values, sparse0_indices, sparse1_values, sparse1_indices
            sparse0_index, sparse0_values, sparse0_indices = next(sparse0_iterator, end)
            sparse1_index, sparse1_values, sparse1_indices = next(sparse1_iterator, end)
        elif _index_less_than(sparse0_index, sparse1_index):
            yield sparse0_index, sparse0_values, sparse0_indices, [], []
            sparse0_index, sparse0_values, sparse0_indices = next(sparse0_iterator, end)
        else:
            yield sparse1_index, [], [], sparse1_values, sparse1_indices
            sparse1_index, sparse1_values, sparse1_indices = next(sparse1_iterator, end)
