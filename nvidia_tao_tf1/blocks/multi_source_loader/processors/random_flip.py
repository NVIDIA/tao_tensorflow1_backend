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
"""Processor for applying random flip augmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import RandomFlip as _RandomFlip


class RandomFlip(TransformProcessor):
    """Augmentation processor that randomly flips images and labels.

    Note that the default value of horizontal_probability is different from
    vertical_probability due to compatability issues for networks that
    currently use this processor but assumes vertical_probability is 0.
    """

    @save_args
    def __init__(self, horizontal_probability=1.0, vertical_probability=0.0):
        """Construct a RandomFlip processor.

        Args:
            horizontal_probability (float): Probability between 0 to 1
                at which a left-right flip occurs. Defaults to 1.0.
            vertical_probability (float): Probability between 0 to 1
                at which a top-bottom flip occurs. Defaults to 0.0.
        """
        super(RandomFlip, self).__init__(
            _RandomFlip(horizontal_probability, vertical_probability)
        )
