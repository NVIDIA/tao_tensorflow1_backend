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
"""Processor for extracting random glimpses from images and labels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import RandomGlimpse as _RandomGlimpse


class RandomGlimpse(TransformProcessor):
    """Processor for extracting random glimpses of images and labels."""

    # Always crop the center region.
    CENTER = "center"
    # Crop at random location keeping the cropped region within original image bounds.
    RANDOM = "random"
    CROP_LOCATIONS = [CENTER, RANDOM]

    @save_args
    def __init__(self, height, width, crop_location=CENTER, crop_probability=0.5):
        """Construct a RandomGlimpse processor.

        Args:
            height (int) New height to which contents will be either cropped or scaled down to.
            width (int) New width to which contents will be either cropper or scaled down to.
            crop_location (str): Enumeration specifying how the crop location is selected.
            crop_probability (float): Probability at which a crop is performed.
        """
        super(RandomGlimpse, self).__init__(
            _RandomGlimpse(
                crop_location=crop_location,
                crop_probability=crop_probability,
                height=height,
                width=width,
            )
        )
