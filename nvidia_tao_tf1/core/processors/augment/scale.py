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
"""Processor for applying scale transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Canvas2D, Transform


class Scale(Processor):
    """Processor for fixed scaling transform."""

    @save_args
    def __init__(self, height, width, **kwargs):
        """Construct a Scale processor.

        Args:
            height (float) New height to which contents will be scaled up/down to.
            width (float) New width to which contents will be scaled up/down/to.
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(Scale, self).__init__(**kwargs)
        if height <= 0:
            raise ValueError("Scale.height ({}) is not positive.".format(height))
        if width <= 0:
            raise ValueError("Scale.width ({}) is not positive.".format(width))
        self._height = height
        self._width = width

    def __repr__(self):
        """Return a string representation of the processor."""
        return "Scale(height={}, width={})".format(self._height, self._width)

    def call(self, transform):
        """Return a Transform whose spatial transformation matrix is perturbed at random.

        Args:
            transform (Transform): An input Transform instance to be processed.

        Returns:
            Transform: Final Transform instance with spatial transform matrix perturbed.
        """
        if not isinstance(transform, Transform):
            raise TypeError(
                "Expecting an argument of type 'Transform', "
                "given: {}.".format(type(transform).__name__)
            )
        horizontal_ratio = transform.canvas_shape.width / self._width
        vertical_ratio = transform.canvas_shape.height / self._height

        stm_zoom = spatial.zoom_matrix(ratio=(horizontal_ratio, vertical_ratio))
        stm_zoom = tf.broadcast_to(
            stm_zoom, tf.shape(input=transform.spatial_transform_matrix)
        )

        processed_stm = tf.matmul(stm_zoom, transform.spatial_transform_matrix)

        return Transform(
            canvas_shape=Canvas2D(height=self._height, width=self._width),
            color_transform_matrix=transform.color_transform_matrix,
            spatial_transform_matrix=processed_stm,
        )
