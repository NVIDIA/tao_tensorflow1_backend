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
"""Processor for applying random contrast transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import color
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Transform


class RandomContrast(Processor):
    """Random contrast transform."""

    @save_args
    def __init__(self, scale_max, center, **kwargs):
        """Construct a RandomContrast processor.

        Args:
            scale_max (float): The scale (or slope) of the contrast, as rotated
                around the provided center point. This value is half of the standard
                deviation, where values of twice the standard deviation are truncated.
                A value of 0 will not affect the matrix.
            center (float): The center around which the contrast is 'tilted', this
                is generally equal to the middle of the pixel value range. This value is
                typically 0.5 with a maximum pixel value of 1, or 127.5 when the maximum
                value is 255.
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(RandomContrast, self).__init__(**kwargs)
        self._scale_max = scale_max
        self._center = center

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomContrast(scale_max={}, center={})".format(
            self._scale_max, self._center
        )

    def call(self, transform):
        """Return a Transform whose color transformation matrix is perturbed at random.

        Args:
            transform (Transform): An input Transform instance to be processed.

        Returns:
            Transform: Final Transform instance with color transform matrix perturbed.
        """
        if not isinstance(transform, Transform):
            raise TypeError(
                "Expecting an argument of type 'Transform', "
                "given: {}.".format(type(transform).__name__)
            )
        batch_size = None
        if transform.color_transform_matrix.shape.ndims == 3:
            batch_size = tf.shape(input=transform.color_transform_matrix)[0]
        ctm_contrast = color.random_contrast_matrix(
            scale_max=self._scale_max, center=self._center, batch_size=batch_size
        )
        processed_ctm = tf.matmul(ctm_contrast, transform.color_transform_matrix)

        return Transform(
            canvas_shape=transform.canvas_shape,
            color_transform_matrix=processed_ctm,
            spatial_transform_matrix=transform.spatial_transform_matrix,
        )
