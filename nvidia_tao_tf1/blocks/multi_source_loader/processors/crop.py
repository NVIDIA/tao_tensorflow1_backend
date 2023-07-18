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
"""Processor for cropping images and labels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import Crop as _Crop


class Crop(TransformProcessor):
    """Crop processor."""

    @save_args
    def __init__(self, left, top, right, bottom):
        """Create a processor for cropping frames and labels.

        The origin of the coordinate system is at the top-left corner. Coordinates keep increasing
        from left to right and from top to bottom.

              top
              --------
        left |        |
             |        | right
              --------
                bottom

        Args:
            left (int): Left edge before which contents will be discarded.
            top (int): Top edge above which contents will be discarded.
            right (int): Right edge after which contents will be discarded
            bottom (int): Bottom edge after which contents will be discarded.
        """
        super(Crop, self).__init__(
            _Crop(left=left, top=top, right=right, bottom=bottom)
        )
