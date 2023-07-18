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
"""Processor which lossily crops images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_FIRST
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor import (
    Processor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Canvas2D,
    FEATURE_CAMERA,
    Images2D,
    LABEL_MAP,
    Polygon2DLabel,
    SequenceExample,
)
from nvidia_tao_tf1.core.coreobject import save_args


class LossyCrop(Processor):
    """Processor that lossily crops images."""

    @save_args
    def __init__(self, left, top, right, bottom):
        """Creates a processor for cropping frames and labels.

        The origin of the coordinate system is at the top-left corner. Coordinates keep increasing
        from left to right and from top to bottom.

              top
              --------
        left |        |
             |        | right
              --------
                bottom

        Args:
            left (int): Left edge before which contents will be discarded.
            top (int): Top edge above which contents will be discarded.
            right (int): Right edge after which contents will be discarded
            bottom (int): Bottom edge after which contents will be discarded.
        """
        super(LossyCrop, self).__init__()
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    @property
    def supported_formats(self):
        """Data formats supported by this processor.

        Returns:
            data_formats (list of 'DataFormat'): Input data formats that this processor supports.
        """
        return [CHANNELS_FIRST]

    def can_compose(self, other):
        """Can't compose in LossyCrop."""
        return False

    def compose(self, other):
        """Does not support composing."""
        raise NotImplementedError("ComposableProcessor.compose not implemented")

    def process(self, example):
        """
        Processes examples by applying filters in sequence.

        The filters are applied in a 2D convolution fashion.

        Args:
            example (Example): Examples to process in format specified by data_format.

        Returns:
            example (Example): Example with a 2D filter applied.
        """
        if not isinstance(example, SequenceExample):
            raise TypeError("Tried process input of type: {}".format(type(example)))

        instances = example.instances
        labels = example.labels

        canvas_shape = Canvas2D(
            height=tf.ones(self._bottom - self._top),
            width=tf.ones(self._right - self._left),
        )

        if FEATURE_CAMERA in instances:
            images2d = instances[FEATURE_CAMERA]
            images = images2d.images
            new_height = self._bottom - self._top
            new_width = self._right - self._left
            image_dims = len(images.shape)
            if image_dims == 5:
                cropped = tf.slice(
                    images,
                    tf.stack([0, 0, 0, self._top, self._left]),
                    tf.stack([-1, -1, -1, new_height, new_width]),
                )
            elif image_dims == 4:
                cropped = tf.slice(
                    images,
                    tf.stack([0, 0, self._top, self._left]),
                    tf.stack([-1, -1, new_height, new_width]),
                )
            else:
                raise RuntimeError("Unhandled image dims: {}".format(image_dims))

            instances[FEATURE_CAMERA] = Images2D(
                images=cropped, canvas_shape=canvas_shape
            )

        if LABEL_MAP in labels:
            polygon_2d_label = labels[LABEL_MAP]

            coordinates_2d = polygon_2d_label.vertices
            sparse_coordinates = coordinates_2d.coordinates

            def _transform_coordinates(coordinates):
                vertices = tf.reshape(coordinates.values, [-1, 2])

                translated = tf.subtract(
                    vertices, tf.constant([self._left, self._top], dtype=tf.float32)
                )

                translated = tf.reshape(translated, tf.shape(input=coordinates.values))

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
                transformed_coordinates, canvas_shape
            )
            translated_polygon_2d_label = Polygon2DLabel(
                vertices=transformed_coordinates_2d,
                classes=polygon_2d_label.classes,
                attributes=polygon_2d_label.attributes,
            )

            labels[LABEL_MAP] = translated_polygon_2d_label

        return SequenceExample(instances=instances, labels=labels)
