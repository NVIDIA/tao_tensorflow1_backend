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
"""Processor for converting polylines into polygons."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2D,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.polygon2d_label import (
    Polygon2DLabel,
)
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args
from nvidia_tao_tf1.core.processors.processors import is_sparse, load_custom_tf_op


logger = logging.getLogger(__name__)


class PolylineToPolygon(TAOObject):
    """Processor that converts polylines to polygons to be later rasterized."""

    @save_args
    def __init__(self, class_id, line_width=1.0, debug=False):
        """
        Construct a PolylineToPolygon processor.

        Args:
            class_id (int): The class id that contains polylines to convert.
            line_width (float): The width of the resulting polygons.

        Raises:
            ValueError: When invalid arguments are provided.
        """
        super(PolylineToPolygon, self).__init__()
        if class_id < 0:
            raise ValueError(
                "class_id: {} must be a valid >= 0 number.".format(class_id)
            )
        if line_width <= 0.0:
            raise ValueError("line_width must be > 0. {} provided.".format(line_width))

        self.class_id = class_id
        self.line_width = float(line_width)
        self.debug = 1 if debug else 0

    def process(self, labels2d):
        """
        Convert a polyline to a set of polygons, to be consumed by the polygon rasterizer.

        Args:
            labels2d (Polygon2DLabel): A label containing 2D polygons/polylines and their
                associated classes and attributes. The first two dimensions of each tensor
                that this structure contains should be batch/example followed by a frame/time
                dimension. The rest of the dimensions encode type specific information. See
                Polygon2DLabel documentation for details.

        Returns:
            (Polygon2DLabel): A label with the same format as before, with components with the
                        label to be converted treated as polylines and then converted
                        to polygons.
        """
        logger.info(
            "Building polyline to polygon conversion for class {} with width {}.".format(
                self.class_id, self.line_width
            )
        )

        new_coordinates, new_class_ids = self._process(
            polygons=labels2d.vertices.coordinates,
            class_ids_per_polygon=labels2d.classes,
        )

        return Polygon2DLabel(
            vertices=Coordinates2D(
                coordinates=new_coordinates, canvas_shape=labels2d.vertices.canvas_shape
            ),
            classes=new_class_ids,
            attributes=labels2d.attributes,
        )

    def _process(self, polygons, class_ids_per_polygon):
        assert is_sparse(polygons)
        assert is_sparse(class_ids_per_polygon)

        op = load_custom_tf_op("op_polyline_to_polygon.so", __file__)
        (
            op_indices,
            op_values,
            op_dense_shape,
            op_class_ids_indices,
            op_class_ids_values,
            op_class_ids_shape,
        ) = op.polyline_to_polygon(
            polygon_indices=polygons.indices,
            polygon_values=polygons.values,
            polygon_dense_shape=polygons.dense_shape,
            class_ids_indices=class_ids_per_polygon.indices,
            class_ids_values=class_ids_per_polygon.values,
            class_ids_shape=class_ids_per_polygon.dense_shape,
            target_class_id=self.class_id,
            line_width=self.line_width,
            debug=self.debug,
        )

        polygons = tf.SparseTensor(
            indices=op_indices, values=op_values, dense_shape=op_dense_shape
        )

        class_ids = tf.SparseTensor(
            indices=op_class_ids_indices,
            values=op_class_ids_values,
            dense_shape=op_class_ids_shape,
        )

        return polygons, class_ids
