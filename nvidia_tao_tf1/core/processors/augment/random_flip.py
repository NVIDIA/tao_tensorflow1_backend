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
"""Processor for applying random flip transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Transform


class RandomFlip(Processor):
    """Random flip transform."""

    @save_args
    def __init__(self, horizontal_probability=0.5, vertical_probability=0.0, **kwargs):
        """Construct a RandomFlip processor.

            Note that the default value of horizontal_probability is different from
            vertical_probability due to compatability issues for networks that
            currently use this processor but assumes vertical_probability is 0.

        Args:
            horizontal_probability (float): Probability between 0 to 1
                at which a left-right flip occurs. Defaults to 0.5.
            vertical_probability (float): Probability between 0 to 1
                at which a top-down flip occurs. Defaults to 0.0.
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(RandomFlip, self).__init__(**kwargs)
        if horizontal_probability < 0.0 or 1.0 < horizontal_probability:
            raise ValueError(
                "RandomFlip.horizontal_probability ({}) is not within the range "
                "[0.0, 1.0].".format(horizontal_probability)
            )
        if vertical_probability < 0.0 or 1.0 < vertical_probability:
            raise ValueError(
                "RandomFlip.vertical_probability ({}) is not within the range "
                "[0.0, 1.0].".format(vertical_probability)
            )
        self._horizontal_probability = horizontal_probability
        self._vertical_probability = vertical_probability

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomFlip(horizontal_probability={}, vertical_probability={})".format(
            self._horizontal_probability, self._vertical_probability
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
        if transform.spatial_transform_matrix.shape.ndims == 3:
            batch_size = tf.shape(input=transform.spatial_transform_matrix)[0]
        stm_flip = spatial.random_flip_matrix(
            horizontal_probability=self._horizontal_probability,
            vertical_probability=self._vertical_probability,
            height=transform.canvas_shape.height,
            width=transform.canvas_shape.width,
            batch_size=batch_size,
        )
        processed_stm = tf.matmul(stm_flip, transform.spatial_transform_matrix)

        return Transform(
            canvas_shape=transform.canvas_shape,
            color_transform_matrix=transform.color_transform_matrix,
            spatial_transform_matrix=processed_stm,
        )
