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
"""Processor for applying random zoom augmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import RandomZoom as _RandomZoom


class RandomZoom(TransformProcessor):
    """Augmentation processor that randomly zooms in/out of images and labels."""

    @save_args
    def __init__(self, ratio_min=0.5, ratio_max=1.5, probability=1.0):
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
        """
        super(RandomZoom, self).__init__(_RandomZoom(ratio_min, ratio_max, probability))
