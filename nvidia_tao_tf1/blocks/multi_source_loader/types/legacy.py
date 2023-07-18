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
"""Legacy LaneNet types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf


# TODO(vkallioniemi): These are only used from the old dataloader - delete once it gets deleted.


"""Value class for representing polygons.

Named tuples are compatible with TF datasets, conditionals and other control flow operations.

Based off of arguments passed to Maglev Rasterizer call here:
  moduluspy/modulus/processors/processors.py:PolygonRasterizer#call

This structure allows for a batch of more than one image, but is often used with just one.

Fields:
    polygons: a tensor of shape ``[P, 2]`` and type tf.float32. Each row contains the (x,y)
        coordinates of a single vertex of a single polygon for a single image. The dimension P
        is equal to the total number of vertices over all polygons.

    vertices_per_polygon: a tensor of shape ``[P]`` and type tf.int32. The elements of the
        tensor are the vertex counts for each polygon. Thus, the length of this list is equal to
        the number of polygons we will draw (P), and if we were to sum all the values in this
        list, the sum should equal the first dimension of the ``polygons`` list above. To get
        the vertices for the ``n``th polygon it would be
        ``vertices[sum(vertices_per_polyogn[0:n-1]):n]``

    class_ids_per_polygon: a tensor of shape ``[P]`` and type tf.int32
        ``vertices_per_polygon`` list above. Each list element is an ID representing
        the class to which each polygon belongs.

    polygons_per_image: A scalar or tensor of shape ``[I]`` and type tf.int32. The Number of
        polygons in an image. Scalar for a single image or tensor for multiple images.

        Note this is slightly different from maglev, where its `None` for a single image (the
        default)

    `attributes_per_polygon`: A tensor shape ``[P]`` and type tf.int32. Legacy field, where
        there is 0 or 1 attribute per polygon. This field is populated with the filtered list of
        attributes from JSONConverter.

    `attributes: A tensor of shape` ``[polygons_per_image]`` and type tf.int32. All the
        attributes associated with the polygons in this image. Mapping to polygons is done using
         the `attributes_per_polygon` field.

    `attribute_count_per_polygon`: At tensor of shape ``[vertices_per_polygon]`` and type
        tf.int32. This field is used to map attributes to a polygon. Similarly to `vertices`,
        `n`th class will have ``attributes_per_polygon[n]`` attributes associated with it. The
        specific attributes will be
        ``attributes[sum(attributes_per_polygon[0:n-1]):attributes_per_polygon[n]]``
"""
PolygonLabel = namedtuple(
    "PolygonLabel",
    [
        "polygons",
        "vertices_per_polygon",
        "class_ids_per_polygon",
        "attributes_per_polygon",
        "polygons_per_image",
        "attributes",
        "attribute_count_per_polygon",
    ],
)


def empty_polygon_label():
    """Method to create an empty PolygonLabel object."""
    return PolygonLabel(
        polygons=tf.constant([[]], shape=[0, 2], dtype=tf.float32),
        vertices_per_polygon=tf.constant([], shape=[1], dtype=tf.int32),
        class_ids_per_polygon=tf.constant([], shape=[1], dtype=tf.int32),
        attributes_per_polygon=tf.constant([], shape=[1], dtype=tf.int32),
        polygons_per_image=tf.constant([0], shape=[1], dtype=tf.int32),
        attributes=tf.constant([], shape=[1], dtype=tf.int32),
        attribute_count_per_polygon=tf.constant([0], shape=[1], dtype=tf.int32),
    )
