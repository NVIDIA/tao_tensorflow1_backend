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

"""Convenience tool for building sparse tensors from python nested lists."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SparseTensorBuilder(object):
    """Convenience object for building sparse tensors.

    This class enables arbitrary nesting, and the ability to convert to a tf.SparseTensor.
    In this test harness, it's used to build out coordinate tensors. Nesting is achieved
    by passing instances of `SparseTensorBuilder` as the `things` argument. This recursion
    is terminated when non-SparseTensorBuilder instances are passed in as things. Mixing
    and matching is *not* allowed, although it is valid to pass nothing in, which allows
    representing an item with no values.

    Example:
    # Used for brevity
    class Ex(SparseTensorBuilder):
        pass

    > builder = Ex(
        Ex(
            Ex(0, 1),
            Ex(2, 3)
        ),
        Ex(
            Ex(4, 5),
            Ex(6, 7),
            Ex(8, 9)
        )
    )

    > sparse_ex = builder.build()

    > sparse_ex.indices
    [
        [0, 0, 0]
        [0, 0, 1]
        [0, 1, 0]
        [0, 1, 1]
        [1, 0, 0]
        [1, 0, 1]
        [1, 1, 0]
        [1, 1, 1]
        [1, 2, 0]
        [1, 2, 1]
    ]

    > sparse_ex.values
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    > sparse_ex.dense_shape
    [2, 3, 2]
    """

    def __init__(self, *things):
        """Constructor.

        Args:
            things (object): A tuple of either SparseTensorBuilders, or scalars.
        """
        self.things = things

    def build(self, val_type=None):
        """Converts this outwrap object into a tf.SparseTensor.

        Args:
            val_type (tf.dtype): The desired dtype of the values tensor.
        """
        indices = []
        values = []
        shape = []
        self._inner_populate(indices, values, shape, 0)

        return tf.SparseTensor(
            indices=tf.constant(indices, dtype=tf.int64),
            values=tf.constant(values, dtype=val_type),
            dense_shape=tf.constant(shape, dtype=tf.int64),
        )

    def _inner_populate(self, indices, values, shape, depth, *prev_idx):
        while len(shape) <= depth:
            shape.append(0)
        if self.things and not isinstance(self.things[0], SparseTensorBuilder):
            for i, val in enumerate(self.things):
                indices.append(prev_idx + (i,))
                values.append(val)
        else:
            for i, val in enumerate(self.things):
                index = prev_idx + (i,)
                val._inner_populate(indices, values, shape, depth + 1, *index)
        shape[depth] = max(shape[depth], len(self.things))
