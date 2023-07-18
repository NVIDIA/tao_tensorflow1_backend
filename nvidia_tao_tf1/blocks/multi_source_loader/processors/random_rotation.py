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
from nvidia_tao_tf1.core.processors import RandomRotation as _RandomRotation


class RandomRotation(TransformProcessor):
    """Augmentation processor that randomly rotates images and labels."""

    @save_args
    def __init__(self, min_angle, max_angle, probability=1.0):
        """Construct a RandomRotation processor.

        Args:
            min_angle (float): Minimum angle in degrees.
            max_angle (float): Maximum angle in degrees.
            probability (float): Probability at which rotation is performed.
        """
        super(RandomRotation, self).__init__(
            _RandomRotation(
                min_angle=min_angle, max_angle=max_angle, probability=probability
            )
        )
