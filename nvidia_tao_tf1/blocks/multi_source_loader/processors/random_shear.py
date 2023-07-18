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
"""Processor for applying random rotation augmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import RandomShear as _RandomShear


class RandomShear(TransformProcessor):
    """Augmentation processor that randomly shear images and labels."""

    @save_args
    def __init__(self, max_ratio_x, max_ratio_y, probability):
        """Construct a RandomShear processor.

        Args:
            max_ratio_x (float): Maximum shear ratio in horizontal direction.
            max_ratio_y (float): Maximum shear ratio in vertical direction.
            probability (float): Probability at which rotation is performed.
        """
        super(RandomShear, self).__init__(
            _RandomShear(
                max_ratio_x=max_ratio_x,
                max_ratio_y=max_ratio_y,
                probability=probability,
            )
        )
