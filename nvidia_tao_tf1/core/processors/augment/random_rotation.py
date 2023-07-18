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
"""Perform random rotations on examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Transform


class RandomRotation(Processor):
    """Random rotate transform."""

    @save_args
    def __init__(self, min_angle, max_angle, probability, **kwargs):
        """Construct a RandomRotation processor.

        Args:
            min_angle (float): Minimum angle in degrees.
            max_angle (float): Maximum angle in degrees.
            probability (float): Probability at which rotation is performed.
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(RandomRotation, self).__init__(**kwargs)
        if probability < 0.0 or probability > 1.0:
            raise ValueError(
                "RandomRotation.probability ({}) is not within the range [0.0, 1.0].".format(
                    probability
                )
            )
        if min_angle < -360.0:
            raise ValueError(
                "RandomRotation.min_angle ({}) is smaller than -360.0 degrees.".format(
                    min_angle
                )
            )
        if max_angle > 360.0:
            raise ValueError(
                "RandomRotation.max_angle ({}) is greater than 360.0 degrees.".format(
                    max_angle
                )
            )
        if min_angle > max_angle:
            raise ValueError(
                "RandomRotation.min_angle ({})"
                " is greater than RandomRotation.max_angle ({}).".format(
                    min_angle, max_angle
                )
            )

        self._probability = probability
        self._min_angle = min_angle
        self._max_angle = max_angle

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomRotation(min_angle={}, max_angle={}, probability={})".format(
            self._min_angle, self._max_angle, self._probability
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

        batch_shape = []
        if transform.spatial_transform_matrix.shape.ndims == 3:
            batch_size = tf.shape(input=transform.spatial_transform_matrix)[0]
            batch_shape = [batch_size]
        angle = tf.random.uniform(
            batch_shape,
            minval=math.radians(self._min_angle),
            maxval=math.radians(self._max_angle),
        )
        rotate_stm = spatial.rotation_matrix(
            angle,
            width=transform.canvas_shape.width,
            height=transform.canvas_shape.height,
        )

        should_rotate = tf.less_equal(
            tf.random.uniform(batch_shape, 0.0, 1.0), self._probability
        )
        next_stm = tf.compat.v1.where(
            should_rotate,
            tf.matmul(rotate_stm, transform.spatial_transform_matrix),
            transform.spatial_transform_matrix,
        )

        return Transform(
            spatial_transform_matrix=next_stm,
            color_transform_matrix=transform.color_transform_matrix,
            canvas_shape=transform.canvas_shape,
        )
