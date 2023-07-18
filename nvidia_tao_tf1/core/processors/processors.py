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
"""Modulus processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
from functools import lru_cache
import os
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.framework.sparse_tensor import is_sparse
from tensorflow.python.ops import string_ops

from nvidia_tao_tf1.core.coreobject import AbstractTAOObject
from nvidia_tao_tf1.core.utils import get_uid_name


def is_tensor(x):
    """Determines if input is a TensorFlow tensor."""
    return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)


def dense_to_sparse(dense_tensor):
    """Convert a dense tensor to a sparse tensor."""
    if is_sparse(dense_tensor):
        return dense_tensor
    indices = tf.compat.v1.where(condition=tf.ones_like(dense_tensor, dtype=tf.bool))
    values = tf.gather_nd(params=dense_tensor, indices=indices)
    shape = tf.shape(input=dense_tensor, out_type=tf.int64)
    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


def boolean_mask_sparse_tensor(st, mask, axis=0):
    """Update a sparse tensor with a a mask.

    NOTICE: This function only can only mask out values that are already absent in the sparse
    tensor. That means it can only ever update sparse tensor ``indices`` and ``dense_shape``.
    TODO(xiangbok): Add support to fully mask sparse tensors, including actual values.

    The tensor will be masked with a boolean tensor, but assumes the content is already removed
    and so is not present in the values or indices. However, the sparse tensor's indices and
    shape will still reflect the empty rows. We want to mask out these indices and the shape.

    A sparse tensor consists of three dense tensors named ``indices``, ``values`` and
    ``dense_shape``. This operation will not change the shape of these three tensors, and will never
    change the content of ``values``.

    The ``dense_shape`` is subtracted by the amount of masking that is applied on the
    given ``axis``.
    The ``indices`` on dimension of input ``axis`` are masked out. This means practically that the
    ``indices`` on the given ``axis`` are subtracted by the amount of masking that has been applied
    to previous indices.

    For example, a sparse tensor with ``indices = [0, 2, 2, 4]`` and ``dense_shape = [5]`` and
    ``mask = [True, False, True, True, True]``, then the output ``dense_shape = [4]`` and
    the ouptut ``indices = [0, 1, 1, 3]``.

    Args:
        st (``tf.SparseTensor``): The input tensor to be masked.
        mask (``tf.Tensor``): The dense tensor to be used as a mask over the dimension indicated by
            ``axis``. When mask values are ``True``, it indicates the values to keep.
        axis (int): The axis over which the mask should be applied.

    Returns:
        A masked ``tf.SparseTensor``.

    Raises:
        ValueError: if ``axis`` value is not supported (non-zero).
    """
    if axis != 0:
        raise ValueError("Only axis=0 supported, got `{}`.".format(axis))
    assert_op = tf.compat.v1.assert_equal(
        tf.cast(st.dense_shape[axis], tf.int32), tf.shape(input=mask)[0]
    )
    # Compute the dense shape. The dense shape is simply subtracted over the correct axis
    # by counting the amount elements where the mask is ``False``.
    with tf.control_dependencies([assert_op]):
        count_zero = tf.reduce_sum(
            input_tensor=tf.cast(tf.equal(mask, False), tf.int64)
        )
        dense_shape = tf.concat(
            [[st.dense_shape[axis] - count_zero], st.dense_shape[1:]], axis=axis
        )
    # Split out the axis row from the other rows. We will use the other rows to concat back later.
    dim0, dims = tf.split(st.indices, num_or_size_splits=[1, -1], axis=1)
    # Calculate how much we need to subtract:
    # Example: ``mask = [True, False, True, True, True]`` -> ``subtraction = [0, -1, -1, -1, -1]``.
    subtraction = tf.scan(
        lambda a, x: a - tf.cast(tf.equal(x, 1), tf.int64),
        tf.cast(~mask, tf.int64),
        initializer=tf.constant(0, tf.int64),
        parallel_iterations=1,
        back_prop=False,
    )
    # These previous subtractions relate to an continuous index range. Our sparse tensor might
    # not be continuous, so we have to gather the updates to the correct indices.
    # Example: ``dim0 = [0, 3]`` and ``subtraction = [0, -1, -1, -1, -1]`` then
    # ``subtraction_sparse = [0, -1]``.
    subtraction_sparse = tf.gather(subtraction, dim0)
    # We apply the subtraction here and concatenate the new axis indices together with the other
    # unaffected indices dimensions.
    indices_dim0 = dim0 + subtraction_sparse
    indices = tf.concat([indices_dim0, dims], axis=1)
    return tf.SparseTensor(indices=indices, values=st.values, dense_shape=dense_shape)


def remove_empty_rows_from_sparse_tensor(st, axis=0):
    """Remove empty rows from a sparse tensor over a given axis.

    Removes empty elements over one axis. For example, if the sparse tensor contains indices
    only over rows ``[2, 4, 5]``; it means ``[0, 1, 3]`` are empty. This function will reduce
    the indices so that ``[2, 4, 5]`` now cleanly map to ``[0, 1, 2]``. The sparse tensor's
    ``dense_shape`` will also be changed accordingly. This function never changes actual values.

    Args:
        st (`tf.SparseTensor`): the tensor to be reduced.
        axis (int): the axis over which the reduction will take place.

    Returns:
        tf.SparseTensor with empty rows removed from the input ``st`` tensor.

    Raises:
        ValueError: if ``axis`` value is not supported (non-zero).
    """
    if axis != 0:
        raise NotImplementedError("Only pruning of sparse tensor axis 0 implemented")
    # Indices.
    indices = st.indices[:, 0]
    dim0, dims = tf.split(st.indices, num_or_size_splits=[1, -1], axis=1)
    uniques, indices = tf.unique(dim0[:, 0])
    indices = tf.expand_dims(indices, 1)
    indices = tf.concat([tf.cast(indices, tf.int64), dims], 1)
    # Compute the new dense shape.
    dim_count = tf.cast(tf.shape(input=uniques)[0], tf.int64)
    dense_shape = tf.concat([[dim_count], st.dense_shape[1:]], axis=0)
    return tf.SparseTensor(indices=indices, values=st.values, dense_shape=dense_shape)


def to_dense_if_sparse_tensor_is_fully_dense(st, axis=0):
    """Convert a tf.SparseTensor to a dense Tensor if it's not sparse over given axis.

    Args:
        st (``tf.SparseTensor``): input tensor.
        axis (int): the dimension over which the density check is performed.

    Returns:
        The input sparse tensor converted to a dense type as ``tf.Tensor``.

    Raises:
        NotImplementedError: if ``axis`` value is not supported (non-zero).
        ValueError: if input tensor ``st`` is not a sparse tensor.
    """
    if axis != 0:
        raise NotImplementedError("Axis {} not supported.".format(axis))
    if not is_sparse(st):
        raise ValueError("Input tensor ({}) should be a tf.SparseTensor.".format(st))
    assert_op = tf.compat.v1.assert_equal(
        tf.shape(input=st.values)[0],
        tf.cast(tf.reduce_prod(input_tensor=st.dense_shape), tf.int32),
    )
    with tf.control_dependencies([assert_op]):
        return tf.reshape(st.values, st.dense_shape)


@lru_cache()
def load_custom_tf_op(filename, python_module_path=__file__):
    """Load a custom tf op library from a file.

    Loads a custom tf op library given the specified filename and a path to the caller. The path
    of the caller (`python_module_path`) should usually be obtained by the `__file__` global
    variable. Given `python_module_path`, this function will search a sibling `lib` directory
    for the specified `filename`.

    Example:
        > print(__file__)
        /foo/bar/baz.py

        > load_custom_tf_op('my_custom_baz_op.so', __file__)
        # Library Path: /foo/lib/my_custom_baz_op.so
        # Calls tf.load_op_library('/foo/lib/my_custom_baz_op.so')

    Args:
        filename (str): The name of the library file (e.g. foo_bar.so).
        python_module_path (str): The path of the python module calling this function.
    """
    abs_path = os.path.join(os.path.dirname(python_module_path), "..", "lib", filename)
    return tf.load_op_library(abs_path)


class Processor(AbstractTAOObject):
    """Processor (non-differentiable Layer) base class.

    This object is very similar to a `keras.Layer`, with some minor differences. The methods
    that a subclass should override are the same, but their inputs and outputs allow for more
    flexibility. The inputs and outputs allow arbitrary (keyword) arguments and dictionaries,
    or no input arguments at all.

    Args:
        kwargs (dict): keyword arguments.
    """

    def __init__(self, **kwargs):
        """__init__ method."""
        name = kwargs.get("name", get_uid_name(self.__class__.__name__))
        self.name = name
        self._built = False

    def _build(self, *args, **kwargs):
        """Anything that needs to be created once for the op is done here.

        For example: weight creation, op creation, or anything that needs intializing, etc.
        """
        pass

    def build(self, *args, **kwargs):
        """Passthrough for the build method and set the member 'built' to True."""
        self._build(*args, **kwargs)
        self._built = True

    @abstractmethod
    def call(self, *args, **kwargs):
        """The layers logic should be implemented in this method."""
        raise NotImplementedError("Override me.")

    def __call__(self, *args, **kwargs):
        """The entrypoint for calling the logic of a layer, after its creation.

        If the layer has not been built using `build()`, it will do so.
        """
        with tf.compat.v1.name_scope(self.name):
            if not self._built:
                self.build(*args, **kwargs)
            return self.call(*args, **kwargs)


def json_arrays_to_tensor(value, dtype, add_brackets=False):
    """Convert a json-encoded values (that may be nested in lists) to a sparse tensor.

    Arguments:
        vertices (tf.string): A valid json-encoded string. These values may be nested inside
            lists. It contains values of strictly compatible datatypes (corresponding to ``dtype``
            argument). F.e. floats and integers are compatible (potential data loss due to casting).
        dtype (tf.dtype): Supported datatype (tf.int32, tf.int64, tf.float32, tf.string), the
            output values will be in this ``dtype``.
        add_brackets (bool) False: If we want to add brackets around our input. For example:
            `[..],[..]` -> `[[..],[..]]`. This is a utility function added for ease of use and
            performance to fuse the string concatenation in this operation.

    Returns:
        A ``tf.SparseTensor`` containing (by definition) the ``indices`` (``tf.int64``),
            ``values`` (``dtype``) and ``dense_shape`` (``tf.int64``) of the decoded json.

    Raises:
        ``tf.errors.InvalidArgumentError`` If the input is not a valid json, or there is mixing of
            incompatible datatypes within the json.
        ``TypeError`` If the input is not a scalar tf.string.
    """
    if add_brackets:
        value = tf.strings.join(["[", value, "]"])
    op = load_custom_tf_op("op_json_arrays_to_tensor.so")
    indices, values, dense_shape = op.json_arrays_to_tensor(value=value, dtype=dtype)
    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)


def sparse_coordinate_feature_to_vertices_and_counts(st, vertex_dims=2):
    """Convert a sparse tensor containing coordinate vertices to a vertex and count tensor.

    Args:
        st (``tf.SparseTensor``): A sparse tensor containing features in the outer dimension,
            then vertices, and in the innermost dimension a list of floats.
            The inner dimension has the shape of ``vertex_dims``.
        vertex_dims (int): The dimension of the vertices used.

    Returns:
        vertices (tensor): A tensor of shape (vertex count, ``vertex_dims``), containing all
            concatented vertices. The feature to which each vertex belongs can be distinguished by
            using the ``vertex_count_per_feature`` output.
        vertex_count_per_feature (tensor): The tensor of shape (feature count,) containing vertex
            count per feature. The sum of this tensor is equal to the total vertex count.
    """
    # Calculate the vertex count per feature.
    feature_count = tf.cast(st.dense_shape[0], tf.int32)
    vertex_count_per_feature = (
        tf.math.bincount(tf.cast(st.indices[:, 0], tf.int32), minlength=feature_count)
        // vertex_dims
    )
    # Reshape the vertices simply to (n, vertex_dims).
    vertices = tf.reshape(st.values, [-1, vertex_dims])
    return vertices, vertex_count_per_feature


def values_and_count_to_sparse_tensor(values, counts, counts_of_counts=None):
    """Converts values and its counts into a sparse tensor.

    Args:
        values: tensor containing a long list of values of all the counts.
            The length of this list is therefore equal to the total number of counts
            that will be put into the sparse tensor.
        counts: a 1D int32 tensor. The elements of the list are the value counts that will be put
            into the sparse tensor. The sum of all the values in this list should equal the length
            of the ``values`` list above.
        counts_per_image: an optional 1D int32 tensor. The elements of the list are the counts for
            each image that that will be put into the sparse tensor. The sum of all the values in
            this list should equal the length of the ``counts`` list above. If this parameter is
            not specified, then the output will be of a rank that is lower by 1

    Returns:
        A ``tf.SparseTensor`` containing (by definition) the ``indices`` (``tf.int64``),
            ``values`` (``dtype``) and ``dense_shape`` (``tf.int64``) of the values.
    """
    if counts_of_counts is None:
        counts_of_counts = tf.zeros([], dtype=tf.int32)

    op = load_custom_tf_op("op_values_and_count_to_sparse_tensor.so")
    indices, output_values, dense_shape = op.values_and_count_to_sparse_tensor(
        values=values, counts=counts, counts_of_counts=counts_of_counts
    )

    return tf.SparseTensor(
        indices=indices, values=output_values, dense_shape=dense_shape
    )


def string_lower(x):
    """Convert any tensor of type tf.string to lowercase."""
    return string_ops.string_lower(x)


def string_upper(x):
    """Convert any tensor of type tf.string to uppercase."""
    return string_ops.string_upper(x)
