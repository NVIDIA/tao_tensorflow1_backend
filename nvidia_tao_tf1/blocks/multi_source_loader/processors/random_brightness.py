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
from nvidia_tao_tf1.core.processors import RandomBrightness as _RandomBrightness


class RandomBrightness(TransformProcessor):
    """Augmentation processor that randomly perturbs color brightness."""

    @save_args
    def __init__(self, scale_max, uniform_across_channels):
        """Construct a RandomBrightness processor.

        Args:
            scale_max (float): The range of the brightness offsets. This value
                is half of the standard deviation, where values of twice the standard
                deviation are truncated. A value of 0 (default) will not affect the matrix.
            uniform_across_channels (bool): If true will apply the same brightness
                shift to all channels. If False, will apply a different brightness shift to each
                channel.
        """
        super(RandomBrightness, self).__init__(
            _RandomBrightness(scale_max, uniform_across_channels)
        )
