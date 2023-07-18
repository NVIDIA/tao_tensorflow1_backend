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
"""Processor for applying random zoom transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Transform


class RandomZoom(Processor):
    """Random zoom transform."""

    @save_args
    def __init__(self, ratio_min=0.5, ratio_max=1.5, probability=0.5, **kwargs):
        """Construct a RandomZoom processor.

        Args:
            ratio_min (float): The lower bound of the zooming ratio's uniform distribution.
                A zooming ratio of 1.0 will not affect the image, while values higher than 1 will
                result in 'zooming out' (image gets rendered smaller than the canvas), and vice
                versa for values below 1.0.
            ratio_max (float): The upper bound of the zooming ratio's uniform distribution.
                A zooming ratio of 1.0 will not affect the image, while values higher than 1 will
                result in 'zooming out' (image gets rendered smaller than the canvas), and vice
                versa for values below 1.0.
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(RandomZoom, self).__init__(**kwargs)
        self._ratio_min = ratio_min
        self._ratio_max = ratio_max
        if probability < 0.0 or probability > 1.0:
            raise ValueError(
                "RandomZoom.probability ({}) is not within the range "
                "[0.0, 1.0].".format(probability)
            )
        self._probability = probability

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomZoom(ratio_min={}, ratio_max={}, probability={})".format(
            self._ratio_min, self._ratio_max, self._probability
        )

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
        batch_size = None
        batch_shape = []
        if transform.spatial_transform_matrix.shape.ndims == 3:
            batch_size = tf.shape(input=transform.spatial_transform_matrix)[0]
            batch_shape = [batch_size]
        probability = tf.random.uniform(batch_shape, minval=0.0, maxval=1.0)
        should_zoom = tf.less_equal(probability, self._probability)
        stm_zoom = spatial.random_zoom_matrix(
            ratio_min=self._ratio_min,
            ratio_max=self._ratio_max,
            width=transform.canvas_shape.width,
            height=transform.canvas_shape.height,
            batch_size=batch_size,
        )
        processed_stm = tf.compat.v1.where(
            should_zoom,
            tf.matmul(stm_zoom, transform.spatial_transform_matrix),
            transform.spatial_transform_matrix,
        )

        return Transform(
            canvas_shape=transform.canvas_shape,
            color_transform_matrix=transform.color_transform_matrix,
            spatial_transform_matrix=processed_stm,
        )
