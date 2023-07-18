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
"""Processor for scaling and translating labels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_FIRST
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor import (
    Processor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    LABEL_MAP,
    Polygon2DLabel,
    PolygonLabel,
    SequenceExample,
)
from nvidia_tao_tf1.core.coreobject import save_args


class LabelAdjustment(Processor):
    """LabelAdjustment processor."""

    @save_args
    def __init__(self, scale=1.0, translation_x=0, translation_y=0):
        """Create a processor for scaling and translating the labels.

        Used as a workaround for datasets  where image and label coordinate systems do not match.

        Args:
            scale (float): Label scaling factor.
            translation_x (int): Label translation in x direction.
            translation_y (int): Label translation in y direction.
        """
        super(LabelAdjustment, self).__init__()
        if scale <= 0:
            raise ValueError("Scale: {} is not positive.".format(scale))
        if translation_x < 0:
            raise ValueError(
                "Translation x: {} cannot be a negative number.".format(translation_x)
            )
        if translation_y < 0:
            raise ValueError(
                "Translation y: {} cannot be a negative number.".format(translation_y)
            )

        self._scale = scale
        self._translation_x = translation_x
        self._translation_y = translation_y

    @property
    def supported_formats(self):
        """Data format of the image/frame tensors to process."""
        return [CHANNELS_FIRST]

    def can_compose(self, other):
        """
        Determine whether two processors can be composed into a single one.

        Args:
            other (Processor): Other processor instance.

        Returns:
            (Boolean): Always False - composition not supported.
        """
        return False

    def compose(self, other):
        """Compose two processors into a single one.

        Args:
            other (Processor): Other processor instance.

        Raises:
            NotImplementedError: Composition not supported.
        """
        raise NotImplementedError("Composition not supported.")

    def process(self, example):
        """
        Apply adjustments to the polygon/ polyline labels.

        Labels do not always have the correct size information (half vs full) and are not
        necessarily aligned with the images (center cropped datasets). The label adjustments
        parameters contain translation and scaling information to align the labels with images.

        Args:
            example (Example): Example with the labels to apply the adjustments on.

        Returns:
            (Example): Example with the adjusted labels.
        """
        # TODO(vkallioniemi): remove these adjustments and the related info from the specs once
        # all tfrecords have been regenerated with accurate information.
        # Sometimes the records do not contain correct size information for labels, so we have to
        # scale and translate them.
        if isinstance(example, SequenceExample):
            labels = example.labels

            if LABEL_MAP in labels:
                # Polygon2DLabel
                polygon_2d_label = labels[LABEL_MAP]

                # Coordinates2DWithCounts
                coordinates_2d = polygon_2d_label.vertices
                sparse_coordinates = coordinates_2d.coordinates

                def _transform_coordinates(coordinates):
                    vertices = tf.reshape(coordinates.values, [-1, 2])

                    scaled = tf.matmul(
                        vertices, tf.linalg.tensor_diag([self._scale, self._scale])
                    )

                    translated = tf.subtract(
                        scaled,
                        tf.constant(
                            [self._translation_x, self._translation_y], dtype=tf.float32
                        ),
                    )

                    translated = tf.reshape(
                        translated, tf.shape(input=coordinates.values)
                    )

                    return tf.SparseTensor(
                        indices=coordinates.indices,
                        values=translated,
                        dense_shape=coordinates.dense_shape,
                    )

                # In case where coordinates are empty do nothing. Transformations on empty Sparse
                # tensors were causing problems.
                transformed_coordinates = tf.cond(
                    pred=tf.greater(tf.size(input=sparse_coordinates.values), 0),
                    true_fn=lambda: _transform_coordinates(sparse_coordinates),
                    false_fn=lambda: sparse_coordinates,
                )

                transformed_coordinates_2d = coordinates_2d.replace_coordinates(
                    transformed_coordinates
                )
                translated_polygon_2d_label = Polygon2DLabel(
                    vertices=transformed_coordinates_2d,
                    classes=polygon_2d_label.classes,
                    attributes=polygon_2d_label.attributes,
                )

                labels[LABEL_MAP] = translated_polygon_2d_label

            return SequenceExample(instances=example.instances, labels=labels)

        # Legacy LaneNet dataloader and expect transformations to be applied
        # here. TODO(vkallioniemi or ehall): remove this functionality once we delete
        # the old dataloader.
        labels = example.labels[LABEL_MAP]
        polygon_2d_label = example.labels[LABEL_MAP].polygons
        scaled_polygons = tf.matmul(
            polygon_2d_label, tf.linalg.tensor_diag([self._scale, self._scale])
        )
        translated_polygon_2d_label = tf.subtract(
            scaled_polygons,
            tf.constant([self._translation_x, self._translation_y], dtype=tf.float32),
        )
        adjusted_polygon = PolygonLabel(
            polygons=translated_polygon_2d_label,
            vertices_per_polygon=labels.vertices_per_polygon,
            class_ids_per_polygon=labels.class_ids_per_polygon,
            attributes_per_polygon=labels.attributes_per_polygon,
            polygons_per_image=labels.polygons_per_image,
            attributes=labels.attributes,
            attribute_count_per_polygon=labels.attribute_count_per_polygon,
        )
        return Example(
            instances=example.instances, labels={LABEL_MAP: adjusted_polygon}
        )
