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
"""Processor for applying random hue and saturation transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import color
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Transform


class RandomHueSaturation(Processor):
    """Random brightness transform."""

    @save_args
    def __init__(self, hue_rotation_max, saturation_shift_max, **kwargs):
        """Construct a RandomHueSaturation processor.

        Args:
            hue_rotation_max (float): The maximum rotation angle (0-360). This used in a truncated
                normal distribution, with a zero mean. This rotation angle is half of the
                standard deviation, because twice the standard deviation will be truncated.
                A value of 0 will not affect the matrix.
            saturation_shift_max (float): The random uniform shift between 0 - 1 that changes the
                saturation. This value gives the negative and positive extent of the
                augmentation, where a value of 0 leaves the matrix unchanged.
                For example, a value of 1 can result in a saturation values bounded
                between of 0 (entirely desaturated) and 2 (twice the saturation).
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(RandomHueSaturation, self).__init__(**kwargs)
        if hue_rotation_max < 0.0 or hue_rotation_max > 360.0:
            raise ValueError(
                "RandomHueSaturation.hue_rotation_max ({})"
                " is not within the range [0.0, 360.0].".format(hue_rotation_max)
            )
        if saturation_shift_max < 0.0 or saturation_shift_max > 1.0:
            raise ValueError(
                "RandomHueSaturation.saturation_shift_max ({})"
                " is not within the range [0.0, 1.0].".format(saturation_shift_max)
            )
        self._hue_rotation_max = hue_rotation_max
        self._saturation_shift_max = saturation_shift_max

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomHueSaturation(hue_rotation_max={}, saturation_shift_max={})".format(
            self._hue_rotation_max, self._saturation_shift_max
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
        ctm_brightness = color.random_hue_saturation_matrix(
            hue_rotation_max=self._hue_rotation_max,
            saturation_shift_max=self._saturation_shift_max,
            batch_size=batch_size,
        )
        processed_ctm = tf.matmul(ctm_brightness, transform.color_transform_matrix)

        return Transform(
            canvas_shape=transform.canvas_shape,
            color_transform_matrix=processed_ctm,
            spatial_transform_matrix=transform.spatial_transform_matrix,
        )
