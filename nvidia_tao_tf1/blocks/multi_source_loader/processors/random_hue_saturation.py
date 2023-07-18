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
"""Processor for applying random hue and saturation augmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import RandomHueSaturation as _RandomHueSaturation


class RandomHueSaturation(TransformProcessor):
    """Augmentation processor that randomly perturbs the hue and saturation of colors."""

    @save_args
    def __init__(self, hue_rotation_max, saturation_shift_max):
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
        """
        super(RandomHueSaturation, self).__init__(
            _RandomHueSaturation(hue_rotation_max, saturation_shift_max)
        )
