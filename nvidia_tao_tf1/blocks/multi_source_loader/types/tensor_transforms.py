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

"""Functions for transforming tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.core.processors import values_and_count_to_sparse_tensor


def map_and_stack(fn, elements, dtype=tf.int64, name=None):
    """
    Stack tensors returned by passed in function along the 0th/row dimension.

    This function works similar to tf.map_fn with the difference that the passed in function may
    return tensors that are of different height.

    Args:
        fn (function): Function that should accept single tf.Tensor as argument and return
            a tf.Tensor.
        elements (tf.Tensor): Tensor to iterate over from left to right.
        dtype (tf.DType): Data type of the tensors returned by function.
        name (string): Optional name for the operations.

    Returns:
        (tf.Tensor): A tensor of type `dtype` with as many rows as is the size of passed in
            elements. The dimensionality of each row matches the dimensionality of tensors
            returned by the passed in function.
    """
    with tf.compat.v1.name_scope(name, "map_and_stack", values=[elements]):
        elements_total = tf.size(input=elements)
        result = tf.TensorArray(
            dtype=dtype, size=elements_total, infer_shape=False, name="result"
        )

        def _body(element_index, result):
            value = tf.cast(fn(element_index), dtype)
            result = result.write(element_index, value)
            return (element_index + 1, result)

        def _condition(element_index, result):
            return element_index < elements_total

        _, final_result = tf.while_loop(
            cond=_condition,
            body=_body,
            loop_vars=(0, result),
            back_prop=False,
            name="loop",
        )

        return tf.cond(
            pred=tf.equal(elements_total, 0),
            true_fn=lambda: tf.constant([], dtype=dtype, name="empty_result"),
            false_fn=lambda: final_result.concat(name="result"),
        )


def vector_and_counts_to_sparse_tensor(vector, counts):
    """
    Create a tf.SparseTensor representation of two dense vectors: vector and counts.

    Dense vectors like this are typically used to represent variable length data as dense tensors.
    E.g. you could have the following situation where an image has two polygon labels that each
    have a different number of classes:

        polygon 0 was labeled with 2 classes: "speed_limit_sign" and "60_MPH"
        polygon 1 was labeled with 1 class: "person"

    The dense vector representation of this would look like:

        vector: tf.Tensor(["speed_limit_sign", "60_MPH", "person"])
        counts: tf.Tensor([2, 1])

    The first tensor ("vector") contains all class names and the second tensor ("counts") tells
    you that the first two belong to the polygon at index 0 and the third one belongs to the
    polygon at index 1.

    This function encodes those two dense tensors as a single sparse tensor. The equivalent sparse
    tensor representation would look like:

        tf.SparseTensor(
            values = ["speed_limit_sign", "60_MPH", "person"], # all values from `vector`
            indices =  [
                [0, 0, 0],   # 0th frame, 0th shape, 0th class (speed_limit_sign)
                [0, 0, 1],   # 0th frame, 0th shape, 1st class (60_MPH)
                [0, 1, 0]    # 0th frame, 1st shape, 0th class (person)
            ],
            dense_shape = [1, 2, 2] # where values indicate 1: number of frames, 2: max number of
                                    # shapes, 2: max number of classes.
        )

    Args:
        vector (tf.Tensor): A dense tensor of shape [V].
        counts (tf.Tensor): A dense tensor of shape [C].

    Returns:
        (tf.SparseTensor): A 3D sparse tensor representation of the 2 passed in dense tensors. The
            sparse tensor consists of these 3 dense tensors that encode the inputs like:
                values: 1D tensor of shape (V) - this tensor contains the values from the passed in
                    vector.
                indices: 2D tensor of shape (T, 3), where T is the sum of all values in the passed
                    in counts vector. Each row in this tensor encodes what shape (polygon/line) a
                    value belongs to. The 0th column is always set to 0 as a performance
                    optimizaiton: this function is called from context where we are reading labels
                    for a single frame.
                The dense_shape of the tensor is [V, E, C] where,
                    V: Total number of frames in the passed in vector - always 1.
                    E: Total number of "groups" that values belong to. (== len(counts))
                    C: Max number of values across all groups. (== max(counts))
    """
    empty_sparse_tensor = tf.SparseTensor(
        indices=tf.reshape(tf.constant([], tf.int64), [0, 2]),
        values=tf.constant([], vector.dtype),
        dense_shape=tf.constant([0, 0], tf.int64),
    )

    # Vectors are split into polygons (counts).
    counts = tf.cast(counts, dtype=tf.int32)
    regular_sparse_tensor = values_and_count_to_sparse_tensor(
        values=vector, counts=counts
    )

    return tf.cond(
        pred=tf.equal(tf.size(input=counts), tf.constant(0)),
        true_fn=lambda: empty_sparse_tensor,
        false_fn=lambda: regular_sparse_tensor,
    )


def sparsify_dense_coordinates(dense_coordinates, vertex_counts_per_polygon):
    """
    Convert dense coordinates to sparse coordinates.

    Args:
        dense_coordinates (tf.Tensor): Tensor of shape [N, 2] and type tf.float32.
            This tensor encodes a list of x, y coordinate pairs (vertices).
        vertex_counts_per_polygon (tf.Tensor): Tensor of shape [P] where each element
            indicates thet number of vertices/coordinate-pairs that belong to each shape/polygon.

    Returns:
        (tf.SparseTensor): A 3D sparse tensor encoding the shape and coordinate information stored
            in the dense tensors passed into this function. The shape of the tensor is
            [S, V, C], where:
                S: Shape - e.g. polygon, polyline or bouding box
                V: Vertex (point) - e.g. a triangle has 3 vertices
                C: Coordinate - x, y coordinate of each vertex.
    """
    empty_sparse_tensor = tf.SparseTensor(
        indices=tf.reshape(tf.constant([], tf.int64), [0, 3]),
        values=tf.constant([], dense_coordinates.dtype),
        dense_shape=tf.constant([0, 0, 0], tf.int64),
    )

    # Dense_coordinates are splitted into polygons (counts).
    counts = tf.cast(vertex_counts_per_polygon, dtype=tf.int32)
    regular_sparse_tensor = values_and_count_to_sparse_tensor(
        values=dense_coordinates, counts=counts
    )

    return tf.cond(
        pred=tf.equal(tf.size(input=counts), tf.constant(0)),
        true_fn=lambda: empty_sparse_tensor,
        false_fn=lambda: regular_sparse_tensor,
    )
