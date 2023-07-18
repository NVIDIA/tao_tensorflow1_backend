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
"""Processor for applying crop transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Canvas2D, Transform


class Crop(Processor):
    """Crop transform processor."""

    @save_args
    def __init__(self, left, top, right, bottom, **kwargs):
        """Construct a crop processor.

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
        super(Crop, self).__init__(**kwargs)
        if left < 0:
            raise ValueError("Crop.left ({}) is not positive.".format(left))
        if top < 0:
            raise ValueError("Crop.top ({}) is not positive.".format(top))
        if right < 0:
            raise ValueError("Crop.right ({}) is not positive.".format(right))
        if bottom < 0:
            raise ValueError("Crop.bottom ({}) is not positive.".format(bottom))
        if right <= left:
            raise ValueError(
                "Crop.right ({}) should be greater than Crop.left ({}).".format(
                    right, left
                )
            )
        if bottom <= top:
            raise ValueError(
                "Crop.bottom ({}) should be greater than Crop.top ({}).".format(
                    bottom, top
                )
            )
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def __repr__(self):
        """Return a string representation of the processor."""
        return "Crop(left={}, top={}, right={}, bottom={})".format(
            self._left, self._top, self._right, self._bottom
        )

    def call(self, transform):
        """Return a Transform that defines the Crop transformation.

        Args:
            transform (Transform): An input Transform instance to be processed.

        Returns:
            Transform: Final Transform instance representing a crop transform.
        """
        if not isinstance(transform, Transform):
            raise TypeError(
                "Expecting an argument of type 'Transform', "
                "given: {}.".format(type(transform).__name__)
            )
        # Translate top left corner up and towards left to to move it outside of the canvas.
        # Translation is expressed in sizes relative to the original canvas.
        translate_stm = spatial.translation_matrix(x=self._left, y=self._top)
        translate_stm = tf.broadcast_to(
            translate_stm, tf.shape(input=transform.spatial_transform_matrix)
        )

        # Reduce canvas size at bottom and right edges to move them outside of the canvas.
        final_shape = Canvas2D(
            width=self._right - self._left, height=self._bottom - self._top
        )

        processed_stm = tf.matmul(translate_stm, transform.spatial_transform_matrix)

        return Transform(
            canvas_shape=final_shape,
            color_transform_matrix=transform.color_transform_matrix,
            spatial_transform_matrix=processed_stm,
        )
