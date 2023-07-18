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
"""Processor for applying random translation transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Transform


class RandomShear(Processor):
    """Random shear transform."""

    @save_args
    def __init__(self, max_ratio_x, max_ratio_y, probability=0.5, **kwargs):
        """Construct a RandomShear processor.

        Args:
            max_x (int): If shear transform occurs, this is the lower and higher bound the
                uniform distribution from which a ratio will be picked to sheared horizontally.
            max_y (int): If translation occurs, this is the lower and higher bound the
                uniform distribution from which a ratio will be picked to sheared vertically.
            probability (float): Probability at which translation occurs.
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(RandomShear, self).__init__(**kwargs)
        if max_ratio_x < 0.0:
            raise ValueError(
                "RandomShear.max_ratio_x ({}) is less than 0.".format(max_ratio_x)
            )
        self._max_ratio_x = max_ratio_x
        if max_ratio_y < 0.0:
            raise ValueError(
                "RandomShear.max_ratio_y ({}) is less than 0.".format(max_ratio_y)
            )
        self._max_ratio_y = max_ratio_y
        if probability < 0.0 or probability > 1.0:
            raise ValueError(
                "RandomShear.probability ({}) is not within the range "
                "[0.0, 1.0].".format(probability)
            )
        self._probability = probability

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomShear(max_ratio_x={}, max_ratio_y={}, probability={})".format(
            self._max_ratio_x, self._max_ratio_y, self._probability
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
        should_shear = tf.less_equal(probability, self._probability)
        stm_shear = spatial.random_shear_matrix(
            max_ratio_x=self._max_ratio_x,
            max_ratio_y=self._max_ratio_y,
            width=transform.canvas_shape.width,
            height=transform.canvas_shape.height,
            batch_size=batch_size,
        )
        processed_stm = tf.compat.v1.where(
            should_shear,
            tf.matmul(stm_shear, transform.spatial_transform_matrix),
            transform.spatial_transform_matrix,
        )

        return Transform(
            canvas_shape=transform.canvas_shape,
            color_transform_matrix=transform.color_transform_matrix,
            spatial_transform_matrix=processed_stm,
        )
