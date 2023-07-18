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
"""Polygon label."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2D,
    Coordinates2DWithCounts,
)

_Polygon2DLabel = namedtuple("Polygon2DLabel", ["vertices", "classes", "attributes"])


class Polygon2DLabel(_Polygon2DLabel):
    """
    Polygon label.

    vertices (Coordinates2D): Coordinates for vertices that make up this polygon.
    classes (tf.SparseTensor): Classes associated with each polygon.
    attributes (tf.SparseTensor): Attributes associated with each polygon.
    """

    def compress_frame_dimension(self):
        """
        Combines frame and batch dimensions in the vertices, classes and attributes.

        Compresses vertices from 5D sparse (B, F, S, V, C) tensor to a 4D (B', S, V, C) where
        B=Batch, F=Frame, S=Shape, V=Vertex, C=Coordinate(always dim 2) and B'=new Batach(BxF)

        Compresses classes from 4D tensor of shape [B, F, S, C] to 3D tensor of shape [B', S, C]
        where B=Batch, F=Frame, S=Shape, C=Class(currently always dim 1) and B'=new Batch(BxF)

        Compresses attributes from 4D tensor of shape [B, F, S, A] to 3D tensor of shape [B', S, A]
        where B=Batch, F=Frame, S=Shape, A=Attribute and B'=new Batch(BxF)

        Allows label to be used in processors that don't handle sequence data like the rasterizer.

        Returns
            New Polygon2DLabel with the vertices, classes and attributes tensors reshaped
            to combine batch and sequence dimensions.
        """
        vertices = self.vertices
        classes = self.classes
        attributes = self.attributes

        coordinates = vertices.coordinates

        reshaped_coordinates = tf.sparse.reshape(
            coordinates,
            [
                -1,
                coordinates.dense_shape[-3],
                coordinates.dense_shape[-2],
                coordinates.dense_shape[-1],
            ],
        )

        reshaped_classes = tf.sparse.reshape(
            classes,
            tf.convert_to_tensor(
                value=[-1, classes.dense_shape[-2], classes.dense_shape[-1]],
                dtype=tf.int64,
            ),
        )

        # Check for empty attributes and handle appropriately.
        new_attribute_shape = tf.cond(
            pred=tf.equal(tf.reduce_sum(input_tensor=attributes.dense_shape), 0),
            true_fn=lambda: tf.convert_to_tensor(value=[0, 0, 0], dtype=tf.int64),
            false_fn=lambda: tf.convert_to_tensor(
                value=[-1, attributes.dense_shape[-2], attributes.dense_shape[-1]],
                dtype=tf.int64,
            ),
        )

        reshaped_attributes = tf.sparse.reshape(attributes, new_attribute_shape)

        return Polygon2DLabel(
            vertices=Coordinates2D(
                coordinates=reshaped_coordinates, canvas_shape=vertices.canvas_shape
            ),
            classes=reshaped_classes,
            attributes=reshaped_attributes,
        )

    def slice_to_last_frame(self):
        """
        Slices out all of the frames except for the last.

        The input tensor is expected to have shape (B, F, S, V, C), and the output
        tensor will be (B, F, S, V, C) with F=1.

        Returns:
            New Polygon2DLabel with the vertices, classes, and attributes tensors sliced
            to only contain the final frame.
        """
        vertices = self.vertices
        classes = self.classes
        attributes = self.attributes

        coordinates = vertices.coordinates
        vert_counts = vertices.vertices_count

        slice_coords = _sparse_slice(coordinates, 1, -1)
        slice_counts = _sparse_slice(vert_counts, 1, -1)

        slice_classes = _sparse_slice(classes, 1, -1)

        slice_attributes = _sparse_slice(attributes, 1, -1)

        return Polygon2DLabel(
            vertices=Coordinates2DWithCounts(
                coordinates=slice_coords,
                canvas_shape=vertices.canvas_shape,
                vertices_count=slice_counts,
            ),
            classes=slice_classes,
            attributes=slice_attributes,
        )


def _sparse_slice(tensor, axis, index):
    shape = tensor.dense_shape

    def _get_slice_prms(shape):
        slice_start = np.zeros_like(shape)
        slice_size = np.copy(shape)

        idx = index
        if idx < 0:
            idx = shape[axis] + idx

        slice_start[axis] = idx  # noqa pylint: disable = E1137
        slice_size[axis] = 1

        return slice_start, slice_size

    slice_start, slice_size = tf.compat.v1.py_func(
        _get_slice_prms, [shape], (shape.dtype, shape.dtype), stateful=False
    )

    sliced_tensor = tf.sparse.slice(tensor, slice_start, slice_size)

    return sliced_tensor
