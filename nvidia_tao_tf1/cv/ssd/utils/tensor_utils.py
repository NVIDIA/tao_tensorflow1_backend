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

"""Tensor utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _get_non_empty_rows_2d_sparse_non_empty(input_tensor):
    """
    Helper function to retrieve non-empty rows of a 2d sparse tensor.

    Args:
        input_tensor (tf.sparse.SparseTensor): must be 2-D and non-empty
    Returns:
        output_tensor (tf.sparse.SparseTensor): output tensor with all rows non-empty
    """
    old_inds = input_tensor.indices

    _, new_rows = tf.unique(old_inds[:, 0], out_idx=tf.int64)
    num_new_rows = tf.reduce_max(new_rows) + 1
    cols = old_inds[:, 1]

    out_tensor = tf.sparse.SparseTensor(indices=tf.stack([new_rows, cols], axis=1),
                                        values=input_tensor.values,
                                        dense_shape=[num_new_rows, input_tensor.dense_shape[1]])
    return out_tensor


def get_non_empty_rows_2d_sparse(input_tensor):
    """
    Helper function to retrieve non-empty rows of a 2d sparse tensor.

    Args:
        input_tensor (tf.sparse.SparseTensor): must be 2-D
    Returns:
        output_tensor (tf.sparse.SparseTensor): output tensor with all rows non-empty
    """
    cols = input_tensor.dense_shape[1]
    empty_tensor = tf.sparse.SparseTensor(
        indices=tf.zeros(dtype=tf.int64, shape=[0, 2]),
        values=tf.zeros(dtype=input_tensor.dtype, shape=[0]),
        dense_shape=[0, cols])
    return tf.cond(tf.equal(tf.size(input_tensor.indices), 0), true_fn=lambda: empty_tensor,
                   false_fn=lambda: _get_non_empty_rows_2d_sparse_non_empty(input_tensor))


def tensor_slice_replace(a, b, a_idx, b_idx, scope=None):
    '''
    Returns a new tensor same as `a` but with `a[a_idx] = b[b_idx]`.

    Args:
        a, b (tensor): `a` and `b` must have same shape except for
            the first dimension.
        a_idx, b_idx (tensor): 1D tensors. `a_idx` and `b_idx` must
            have the same shape and all elements in `a_idx` should
            be smaller than `a.shape[0]`. Similar for `b_idx`

    Returns:
        c (tensor): A tensor same as `a` but with `a_idx` repalced
            by `b[b_idx]`.
    '''
    with tf.name_scope(scope, 'SliceReplace'):
        a_all_idx = tf.range(tf.shape(a)[0])
        _, a_remaining_idx = tf.setdiff1d(a_all_idx, a_idx)
        return tf.dynamic_stitch([a_remaining_idx, a_idx],
                                 [tf.gather(a, a_remaining_idx),
                                  tf.gather(b, b_idx)])


def tensor_strided_replace(a, a_range, b, axis=0, scope=None):
    '''
    Tensor strided replace.

    Return a new tensor same as `a` but with `a[...,a_range,...] = b`
    `b` shape on axis can be different from `a_range`.

    Args:
        a, b (tensor): `a` and `b` must have same shape except for the
            `axis` dimension. Moreover, `b` can be a list of tensors
            with no `axis` dimension, in which case tensors in `b` will
            be stacked.
        a_range (tuple): a tuple with 2 integers. `a[tuple[0]:tuple[1]]`
            will be replaced by `b`
        axis (0 or -1): along which axis to replace

    Returns:
        c (tensor): the replaced tensor.
    '''
    with tf.name_scope(scope, 'StridedReplace'):
        if axis not in [0, -1]:
            raise NotImplementedError("This function only supports axis 0 or 1")
        if type(b) == tuple or type(b) == list:
            b = tf.stack(list(b), axis=axis)
        concat_list = [None, b, None]
        if a_range[0] < 0:
            end = a.get_shape().as_list()[-1]
        else:
            end = 0
        """
        axis == 0:
        concat_list[0] = a[:a_range[0]]
        concat_list[-1] = a[a_range[1]:]
        axis == -1:
        concat_list[0] = a[..., :a_range[0]]
        concat_list[-1] = a[..., a_range[1]:]
        """
        a0 = tf.gather(a, tf.range(0, end + a_range[0]), axis=axis)
        a1 = tf.gather(a, tf.range(end + a_range[1], a.get_shape().as_list()[axis]), axis=axis)
        concat_list[0] = a0
        concat_list[-1] = a1
        return tf.concat(concat_list, axis=axis)


def get_init_ops():
    """Return all ops required for initialization."""
    """copied from dlav.common.graph.initializer"""
    return tf.group(tf.local_variables_initializer(),
                    tf.tables_initializer(),
                    *tf.get_collection('iterator_init'))
