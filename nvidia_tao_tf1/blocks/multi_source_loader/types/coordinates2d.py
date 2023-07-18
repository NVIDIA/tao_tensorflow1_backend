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
"""Coordinates2D are used to represent geometric shapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from nvidia_tao_tf1.core import processors


class Coordinates2D(
    collections.namedtuple("Coordinates2D", ["coordinates", "canvas_shape"])
):
    """
    Geometric primitive for representing coordinates.

    coordinates (tf.SparseTensor): A 5D SparseTensor of shape [B, T, S, V, C] and type tf.float32.
        B = Batch - index of an example within a batch.
        T = Time - timestep/sequence index within an example - matches the index of a frame/image.
        S = Shape - shape index within a single frame/timestep. Shapes can be e.g. polygons,
            bounding boxes or polylines depending on the dataset/task.
        V = Vertex - index of a vertex (point) within a shape. E.g. triangle has 3 vertices.
        C = Coordinate - index of the coordinate within a vertex.  0: x coordinate, 1: y coordinate.
    canvas_shape (Canvas2D): Shape of the canvas on which coordinates reside.

    Coordinates are encoded as sparse tensors using the following scheme:
    # Values contain (x,y) coordinates stored in a 1D tensor
    vertices = tf.constant([
        # 0th shape: triangle
        1, 1, - top left vertex (x=1, y=1)
        3, 1, - top right vertex
        2, 2, - bottom middle vertex
        # 1st shape: line
        5, 5, - left vertex
        10, 10,  - right vertex
    ])

    # sparse indices are used to encode which (x, y) coordinates belong to which shape, image,
    # sequence and batch.
    indices = tf.constant([
        # 0th shape (triangle) coordinates
        [0, 0, 0, 0, 0], # 0th example, 0th shape, 0th vertex, x coordinate
        [0, 0, 0, 0, 1], # 0th example, 0th shape, 0th vertex, y coordinate
        [0, 0, 0, 1, 0], # 0th example, 0th shape, 1st vertex, x coordinate
        [0, 0, 0, 1, 1], # 0th example, 0th shape, 1st vertex, y coordinate
        [0, 0, 0, 2, 0], # 0th example, 0th shape, 2nd vertex, x coordinate
        [0, 0, 0, 2, 1], # 0th example, 0th shape, 2nd vertex, y coordinate
        # 1th shape (line) coordinates
        [0, 0, 1, 0, 0], # 1st shape, 0th vertex, x coordinate
        [0, 0, 1, 0, 1], # 1st shape, 0th vertex, y coordinate
        [0, 0, 1, 1, 0], # 1st shape, 1st vertex, x coordinate
        [0, 0, 1, 1, 1], # 1st shape, 1st vertex, y coordinate
    ], dtype=tf.int64)

    first_frame_shapes = tf.SparseTensor(
        indices = tf.reshape(indices, (-1, 5)),
        values = vertices,
        # dense_shape encodes:
        #   0th dim: number of examples within a batch (e.g. 32 if batch size is 32)
        #   1st dim: max number or timesteps within an example. For image data, this corresponds to
        #       the max number of frames.
        #   2nd dim: max number of shapes per frame.
        #   3rd dim: max number of vertices per shape.
        #   4th dim: number of coordinates per vertex (always 2 for 2D shapes.)
        dense_shape = tf.constant((1, 1, 2, 3, 2), dtype=tf.int64))
    )
    """

    def apply(self, transform, **kwargs):
        """
        Applies transformation to coordinates and canvas shape.

        Args:
            transform (Transform): Transform to apply.

        Returns:
            (Coordinates2D): Transformed coordinates.
        """
        polygon_transform = processors.PolygonTransform()

        transformed_coordinates = polygon_transform(
            self.coordinates, transform.spatial_transform_matrix
        )

        return Coordinates2D(
            coordinates=transformed_coordinates, canvas_shape=transform.canvas_shape
        )

    def replace_coordinates(self, new_coords, canvas_shape=None):
        """
        Create new object with new coordinates tensor.

        Note that we expect shape and countsto remain same.

        Args:
            new_coords (SparseTensor): Replacement tensor for coordinates.
            canvas_shape (Tensor): (optional) canvas_shape.
        """
        return Coordinates2D(
            coordinates=new_coords, canvas_shape=canvas_shape or self.canvas_shape
        )


class Coordinates2DWithCounts(
    collections.namedtuple(
        "Coordinates2DWithCounts", ["coordinates", "canvas_shape", "vertices_count"]
    )
):
    """
    Geometric primitive for representing coordinates, with a vector for counts.

    coordinates (tf.SparseTensor): A 5D SparseTensor of shape [B, T, S, V, C] and type tf.float32.
        B = Batch - index of an example within a batch.
        T = Time - timestep/sequence index within an example - matches the index of a frame/image.
        S = Shape - shape index within a single frame/timestep. Shapes can be e.g. polygons,
            bounding boxes or polylines depending on the dataset/task.
        V = Vertex - index of a vertex (point) within a shape. E.g. triangle has 3 vertices.
        C = Coordinate - index of the coordinate within a vertex.  0: x coordinate, 1: y coordinate.
    canvas_shape (Canvas2D): Shape of the canvas on which coordinates reside.
    vertices_count (tf.SparseTensor): number of vertices per polygon [B, T, S]


    Coordinates are encoded as sparse tensors using the following scheme:
    # Values contain (x,y) coordinates stored in a 1D tensor
    vertices = tf.constant([
        # 0th shape: triangle
        1, 1, - top left vertex (x=1, y=1)
        3, 1, - top right vertex
        2, 2, - bottom middle vertex
        # 1st shape: line
        5, 5, - left vertex
        10, 10,  - right vertex
    ])

    # sparse indices are used to encode which (x, y) coordinates belong to which shape, image,
    # sequence and batch.
    indices = tf.constant([
        # 0th shape (triangle) coordinates
        [0, 0, 0, 0, 0], # 0th example, 0th shape, 0th vertex, x coordinate
        [0, 0, 0, 0, 1], # 0th example, 0th shape, 0th vertex, y coordinate
        [0, 0, 0, 1, 0], # 0th example, 0th shape, 1st vertex, x coordinate
        [0, 0, 0, 1, 1], # 0th example, 0th shape, 1st vertex, y coordinate
        [0, 0, 0, 2, 0], # 0th example, 0th shape, 2nd vertex, x coordinate
        [0, 0, 0, 2, 1], # 0th example, 0th shape, 2nd vertex, y coordinate
        # 1th shape (line) coordinates
        [0, 0, 1, 0, 0], # 1st shape, 0th vertex, x coordinate
        [0, 0, 1, 0, 1], # 1st shape, 0th vertex, y coordinate
        [0, 0, 1, 1, 0], # 1st shape, 1st vertex, x coordinate
        [0, 0, 1, 1, 1], # 1st shape, 1st vertex, y coordinate
    ], dtype=tf.int64)

    first_frame_shapes = tf.SparseTensor(
        indices = tf.reshape(indices, (-1, 5)),
        values = vertices,
        # dense_shape encodes:
        #   0th dim: number of examples within a batch (e.g. 32 if batch size is 32)
        #   1st dim: max number or timesteps within an example. For image data, this corresponds to
        #       the max number of frames.
        #   2nd dim: max number of shapes per frame.
        #   3rd dim: max number of vertices per shape.
        #   4th dim: number of coordinates per vertex (always 2 for 2D shapes.)
        dense_shape = tf.constant((1, 1, 2, 3, 2), dtype=tf.int64))
    )
    """

    def apply(self, transform, **kwargs):
        """
        Applies transformation to coordinates and canvas shape.

        Args:
            transform (Transform): Transform to apply.

        Returns:
            (Coordinates2D): Transformed coordinates.
        """
        polygon_transform = processors.PolygonTransform()

        transformed_coordinates = polygon_transform(
            self.coordinates, transform.spatial_transform_matrix
        )

        return Coordinates2DWithCounts(
            coordinates=transformed_coordinates,
            canvas_shape=transform.canvas_shape,
            vertices_count=self.vertices_count,
        )

    def replace_coordinates(self, new_coords, canvas_shape=None):
        """
        Create new object with a new coordinates tensor.

        Note that we expect shape and counts to remain same.

        Args:
            new_coords (SparseTensor): Replacement tensor for coordinates.
            canvas_shape (Tensor): (optional) canvas_shape.
        """
        return Coordinates2DWithCounts(
            coordinates=new_coords,
            canvas_shape=canvas_shape or self.canvas_shape,
            vertices_count=self.vertices_count,
        )
