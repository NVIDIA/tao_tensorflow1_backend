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

"""Clip Polygon Processor."""

import tensorflow as tf
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import load_custom_tf_op, Processor


class ClipPolygon(Processor):
    """
    Processor to clip (crop) polygons.

    Op to clip polygons or polylines with an input polygon mask.

    Clipped polygons do not give any intra-polygon coordinate ordering guarantees. This is
    typically not a problem as lines or polygons are agnostic to direction.
    Polygons are assumed to be cyclical, and can therefore 'shift' indices in the array, and
    can even be inverted in direction. Polylines (`closed` is False) are not cyclical and can
    therefore only revert in direction, but can never be shifted.

    Self-intersecting polygons will be split into multiple non-intersecting polygons. This
    means that the amount of output polygons can increase or decrease. This does not apply to
    polylines (`closed` is False). Similarly, the amount of output polygons and polylines can
    decrease if they are clipped entirely.

    Args:
        closed (bool): if the polygon is closed or open (also known as polyline).
    """

    @save_args
    def __init__(self, closed, **kwargs):
        """__init__ method."""
        self.closed = closed
        super(ClipPolygon, self).__init__(**kwargs)

    def call(self, polygons, points_per_polygon, polygon_mask):
        """call method.

        Args:
            polygons: a (n, 2) fp32 tensor containing a long list of vertices of all the polygons
                The top-level list contains sub-lists with 2 elements each; each sub-list contains
                the x/y coordinates (in that order) of a single vertex of a single polygon for a
                single image (= raster map). The length of the top-level list is therefore equal to
                the total number of vertices over all polygons that we are drawing over all raster
                maps.
            points_per_polygon: a 1D int32 tensor. The elements of the list are the vertex counts
                for each polygon that we will draw during rasterization. Thus, the length of this
                list is equal to the number of polygons we will draw, and if we were to sum all the
                values in this list, the sum should equal the length of the ``polygons`` list above.
            polygon_mask: a (n,2) fp32 tensor containing a single polygon used as the clipping mask.

        Returns:
            Three tensors:
                clipped_polygons: same shape and type as `polygons` input, but clipped.
                clipped_points_per_polygon: same shape and type as `points_per_polygon` input,
                    but clipped.
                clipped_polygon_index_map: a 1D int32 tensor with the same length as
                    `clipped_points_per_polygon`. This contains a per-polygon map to the original
                    polygon index that was used as input. This is done because the amount of
                    input polygons might decrease or even increase. If those input polygons have
                    had metadata associated with them, we can map the output polygons to the
                    correct metadata using this tensor.
        """
        points_per_polygon = tf.cast(points_per_polygon, tf.int32)

        op = load_custom_tf_op("op_clip_polygon.so")
        clipped_polygons, clipped_points_per_polygon, clipped_polygon_index_map = op.clip_polygon(
            polygons=polygons,
            points_per_polygon=points_per_polygon,
            polygon_mask=polygon_mask,
            closed=self.closed,
        )

        return (clipped_polygons, clipped_points_per_polygon, clipped_polygon_index_map)
