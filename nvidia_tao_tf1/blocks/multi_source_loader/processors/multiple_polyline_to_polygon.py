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


class MultiplePolylineToPolygon(TAOObject):
    """Processor that converts multiple polylines to polygons to be later rasterized."""

    @save_args
    def __init__(self, attribute_id_list, class_id_list):
        """
        Construct a MultiplePolylineToPolygon processor.

        Args:
            attribute_id_list: list of attribute ids that need to be mapped to class ids.
                               (The list should be unique as this will be used as key in a
                                one to one mapping)
            class_id_list: list of class ids that the attribute ids are mapped to.
        """
        super(MultiplePolylineToPolygon, self).__init__()
        self._attribute_id_list = attribute_id_list
        self._class_id_list = class_id_list

    def process(self, labels2d):
        """
        Combine polylines based on attribute ID into polygons.

        These polygons are consumed by the polygon rasterizer.

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
        logger.info("Building mulitple polyline to polygon conversion.")
        self.labels2d = labels2d
        self.polygons = labels2d.vertices.coordinates
        self.class_ids_per_polygon = labels2d.classes
        self.attributes_per_polygon = labels2d.attributes
        assert is_sparse(self.polygons)
        assert is_sparse(self.class_ids_per_polygon)
        assert is_sparse(self.attributes_per_polygon)
        output_labels = tf.cond(
            pred=tf.equal(tf.size(input=self.polygons.indices), 0),
            true_fn=self._no_op,
            false_fn=self._apply_op,
        )
        return output_labels

    def _no_op(self):
        """
        Return input as is to the output.

        Returns:
            (Polygon2DLabel): A label with the same format as before, with components with the
                        label to be converted treated as polylines and then converted
                        to polygons.
        """
        return Polygon2DLabel(
            vertices=Coordinates2D(
                coordinates=self.polygons,
                canvas_shape=self.labels2d.vertices.canvas_shape,
            ),
            classes=self.class_ids_per_polygon,
            attributes=self.attributes_per_polygon,
        )

    def _apply_op(self):
        """
        Apply Multiple Polyline to Polygon Op and return labels.

        Returns:
            (Polygon2DLabel): A label with the same format as before, with components with the
                        label to be converted treated as polylines and then converted
                        to polygons.
        """
        op = load_custom_tf_op("op_multiple_polyline_to_polygon.so", __file__)

        attribute_id_list = tf.constant(self._attribute_id_list, dtype=tf.int32)
        class_id_list = tf.constant(self._class_id_list, dtype=tf.int32)

        (
            op_indices,
            op_values,
            op_dense_shape,
            op_class_ids_indices,
            op_class_ids_values,
            op_class_ids_shape,
        ) = op.multiple_polyline_to_polygon(
            polygon_indices=self.polygons.indices,
            polygon_values=self.polygons.values,
            polygon_dense_shape=self.polygons.dense_shape,
            attribute_indices=self.attributes_per_polygon.indices,
            attribute_values=self.attributes_per_polygon.values,
            attribute_shape=self.attributes_per_polygon.dense_shape,
            attribute_id_list=attribute_id_list,
            class_id_list=class_id_list,
        )

        converted_polygons = tf.SparseTensor(
            indices=op_indices, values=op_values, dense_shape=op_dense_shape
        )

        converted_class_ids = tf.SparseTensor(
            indices=op_class_ids_indices,
            values=op_class_ids_values,
            dense_shape=op_class_ids_shape,
        )

        return Polygon2DLabel(
            vertices=Coordinates2D(
                coordinates=converted_polygons,
                canvas_shape=self.labels2d.vertices.canvas_shape,
            ),
            classes=converted_class_ids,
            attributes=self.attributes_per_polygon,
        )
