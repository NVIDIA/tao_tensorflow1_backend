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
"""Processor for applying random contrast augmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import RandomContrast as _RandomContrast


class RandomContrast(TransformProcessor):
    """Augmentation processor that randomly perturbs the contrast of images."""

    @save_args
    def __init__(self, scale_max, center):
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
        """
        super(RandomContrast, self).__init__(_RandomContrast(scale_max, center))
