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
"""Transformation encapsulates spatial, color and canvas size changes produced by Transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.coreobject import TAOObject, save_args


class Transformation(TAOObject):
    """Transformation encapsulates spatial, color and canvas size changes produced by Transforms."""

    @save_args
    def __init__(self, spatial_transform_matrix, color_transform_matrix, canvas_shape):
        """Construct transformation.

        Args:
            spatial_transform_matrix (Tensor): Spatial transform matrix.
            color_transform_matrix (Tensor): Color transform matrix.
            canvas_shape (Canvas2D): Shape of a 2 dimensional canvas.
        """
        super(Transformation, self).__init__()
        self.spatial_transform_matrix = spatial_transform_matrix
        self.color_transform_matrix = color_transform_matrix
        self.canvas_shape = canvas_shape
