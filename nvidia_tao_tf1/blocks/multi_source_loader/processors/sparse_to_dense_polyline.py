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
"""Class for mapping objects to output unique instance ids."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors import sparse_generators
from nvidia_tao_tf1.blocks.multi_source_loader.types import Polygon2DLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2D,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import Processor


def make_indices(sub_index, num_vertices):
    """Make indices for vertices sparse tensor.

    Args:
        sub_index (list of int): Sub-index associated with the polyline.
        num_vertices (int): Total number of vertices in the polyline.

    Returns:
        indices (list of list of int): Indices for sparse tensor.
    """
    indices = []
    for vert_index in range(num_vertices):
        indices.append(sub_index + [vert_index, 0])
        indices.append(sub_index + [vert_index, 1])
    return indices


def line_points(point1, point2):
    """Create a point list for chord joining point1 and point2.

    Args:
        point1 (list): 2D point [x, y].
        point2 (list): 2D point [x, y].

    Returns:
        pnt_list (list of vertices([x, y])): List of points joining point1 and point2.
    """
    flag_rev = 0
    # Find the minimum point between the two given points
    if (int(point1[0]) == int(point2[0]) and point1[1] > point2[1]) or (
        point1[0] > point2[0]
    ):
        flag_rev = 1

    # range_columns defines the number of columns in the image
    # that this line segment crosses
    range_columns = range(int(point1[0]), int(point2[0]), -1 if flag_rev else 1)
    point_list = []
    if len(range_columns) == 0:
        # Both the columns are adjacent or lie on the same column, add points with
        # varying y values
        range_rows = range(int(point1[1]), int(point2[1]), -1 if flag_rev else 1)
        for x in range_rows:
            point_list.append([float(point1[0]), float(x)])
    else:
        # Form the line equation and create a point list
        if point2[0] != point1[0]:
            slope = float(point2[1] - point1[1]) / float(point2[0] - point1[0])
            const = point1[1] - (slope * point1[0])
        else:
            slope = None
            const = None

        for x in range_columns:
            if slope is not None and const is not None:
                y = slope * float(x) + const
                point_list.append([float(x), y])

    return point_list


def sparse_to_dense_polyline(values, indices, shape, polyline_prefix_size):
    """Create dense polylines from sparse polylines.

    The function creates a dense polyline by calculating y for every integer x
    between the first vertex and last vertex of every polyline.

    Args:
        values (array of float): x and y values of vertices in the polyline.
        indices (array of array of int): Indices of the vertices sparse tensor.
        shape (array of int): Shape of the corresponding dense tensor.
        polyline_prefix_size (int): Number of indices columns that uniquely
                identify every polyline.

    Return:
        final_values: (array of float): x and y values of vertices in the polyline.
        final_indices (2D array of int): Indices of the output sparse tensor.
        final_shape (array of int): Shape of the dense output tensor.
    """
    final_indices = []
    final_values = []
    max_no_vertices = 0
    # Iterate through every polyline in the input sparse tensor and make the
    # polyline dense.
    for (polyline_sub_index, polyline_values, _) in sparse_generators.sparse_generator(
        polyline_prefix_size, values, indices
    ):
        total_list = []

        # Since polyline values contains x and y of every vertex, it should be even
        # in length.
        assert (
            len(polyline_values) % 2 == 0
        ), "Polyline values array should have even length."

        # Iterate through the vertices of the polyline and populate vertices for
        # every missing integer x value between any two vertices.
        for vert_ind in range(0, len(polyline_values), 2):
            point1, point2 = None, None

            # Populate point1 only if y value (vert_ind + 1) is still within bounds.
            if (vert_ind + 1) < len(polyline_values):
                point1 = [polyline_values[vert_ind], polyline_values[vert_ind + 1]]

            # Populate point2 only if x value (vert_ind + 2) and y value (vert_ind + 3)
            # are still within bounds.
            if (vert_ind + 2) < len(polyline_values) and (vert_ind + 3) < len(
                polyline_values
            ):
                point2 = [polyline_values[vert_ind + 2], polyline_values[vert_ind + 3]]

            # If point1 and point2 are both populated, then find the vertices
            # between them.
            if (point1 is not None) and (point2 is not None):
                total_list = total_list + line_points(point1, point2)

            # If point1 is populated and point2 is not, just add point1 to the total list
            # of points.
            if (point1 is not None) and (point2 is None):
                total_list = total_list + [point1]

        # Add the last point to the list if it is not None.
        if point2 is not None:
            total_list = total_list + [point2]

        # Total vertices for the current polyline.
        total_vertices = len(total_list)
        # Check if total_vertices of this polyline exceeds the max vertices of this
        # polyline.
        if max_no_vertices < total_vertices:
            max_no_vertices = total_vertices

        # Flatten the total_list of vertices from this form
        # [[x1, y1], [x2, y2], [x3, y3], ...] to [x1, y1, x2, y2, x3, y3, ...] form.
        dense_polyline_values = [coord for point in total_list for coord in point]
        # Create indices for the sparse tensor representing this polyline.
        dense_polyline_indices = make_indices(polyline_sub_index, total_vertices)
        # Add this polyline to the final polyline list.
        final_values += dense_polyline_values
        final_indices += dense_polyline_indices

    # Create final shape and arrays to return.
    final_shape = [shape[0], shape[1], shape[2], max_no_vertices, shape[4]]
    final_indices = np.array(final_indices, dtype=np.int64)
    final_values = np.array(final_values, np.float32)
    final_shape = np.array(final_shape, np.int64)
    return final_values, final_indices, final_shape


class SparseToDensePolyline(Processor):
    """SparseToDensePolyline processor creates a denser polyline from a sparse polyline.

    This processor will linearly interpolate vertices between all the adjacent points in a
    polyline. The resulting label will be a dense polyline with vertices for every integer
    x between the two connecting points.
    """

    @save_args
    def __init__(self, **kwargs):
        """Construct a SparseToDensePolyline processor.

        Args

        """
        super(SparseToDensePolyline, self).__init__(**kwargs)

    def call(self, polygon_2d_label):
        """Create dense polyline from sparse polyline.

        Args:
            polygon_2d_label (Polygon2DLabel): A label containing 2D polygons and their
                associated classes and attributes.

        Returns:
            (Polygon2DLabel): The label with the classes and attributes mapped to unique numeric
            id for each instance.
        """
        polyline_prefix_size = 3
        vertices = polygon_2d_label.vertices.coordinates

        # Create sparse to dense polyline.
        dense_vertices_values, dense_vertices_indices, dense_vertices_shape = tf.compat.v1.py_func(
            sparse_to_dense_polyline,
            [
                vertices.values,
                vertices.indices,
                vertices.dense_shape,
                polyline_prefix_size,
            ],
            [tf.float32, tf.int64, tf.int64],
        )

        # Create final vertices as Coordinates2D
        final_vertices = Coordinates2D(
            coordinates=tf.SparseTensor(
                values=dense_vertices_values,
                indices=dense_vertices_indices,
                dense_shape=dense_vertices_shape,
            ),
            canvas_shape=polygon_2d_label.vertices.canvas_shape,
        )
        return Polygon2DLabel(
            vertices=final_vertices,
            classes=polygon_2d_label.classes,
            attributes=polygon_2d_label.attributes,
        )
