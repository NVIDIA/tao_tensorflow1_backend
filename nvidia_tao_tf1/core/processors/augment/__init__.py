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
"""Modulus augment processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import add_move, MovedModule
add_move(MovedModule('mock', 'mock', 'unittest.mock'))

from nvidia_tao_tf1.core.processors.augment import additive_noise
from nvidia_tao_tf1.core.processors.augment import blur
from nvidia_tao_tf1.core.processors.augment import color
from nvidia_tao_tf1.core.processors.augment import pixel_removal
from nvidia_tao_tf1.core.processors.augment import random_blur
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.augment import spatial_matrices_3D
from nvidia_tao_tf1.core.processors.augment.additive_noise import AdditiveNoise
from nvidia_tao_tf1.core.processors.augment.blur import Blur
from nvidia_tao_tf1.core.processors.augment.pixel_removal import PixelRemoval
from nvidia_tao_tf1.core.processors.augment.random_blur import RandomBlur

__all__ = (
    "color",
    "spatial",
    "spatial_matrices_3D",
    "blur",
    "additive_noise",
    "pixel_removal",
    "random_blur",
    "AdditiveNoise",
    "Blur",
    "PixelRemoval",
    "RandomBlur",
)
