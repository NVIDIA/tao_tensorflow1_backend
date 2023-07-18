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

"""Types used to compose Examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.multi_source_loader.types import test_fixtures
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import (
    Bbox2DLabel,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import (
    filter_bbox_label_based_on_minimum_dims,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2D,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2DWithCounts,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.images2d import (
    Images2D,
    LabelledImages2D
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.images2d_reference import (
    Images2DReference, LabelledImages2DReference, LabelledImages2DReferenceVal,
    set_augmentations, set_augmentations_val, set_auto_resize, set_h_tensor,
    set_h_tensor_val, set_image_channels, set_image_depth, set_max_side,
    set_min_side, set_w_tensor, set_w_tensor_val
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.legacy import (
    empty_polygon_label,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.legacy import PolygonLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types.partition_label import (
    PartitionLabel,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.polygon2d_label import (
    Polygon2DLabel,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.process_markers import (
    map_markers_to_orientations,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.process_markers import (
    map_orientation_to_markers,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    FEATURE_CAMERA,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    FEATURE_SESSION,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_DEPTH_DENSE_MAP,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_DEPTH_FREESPACE,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_FREESPACE_REGRESSION,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_FREESPACE_SEGMENTATION,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_MAP,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_OBJECT,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_PANOPTIC_SEGMENTATION,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_PATH,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    SequenceExample,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.session import Session
from nvidia_tao_tf1.blocks.multi_source_loader.types.tensor_transforms import (
    map_and_stack,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.tensor_transforms import (
    sparsify_dense_coordinates,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.tensor_transforms import (
    vector_and_counts_to_sparse_tensor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.transformed_example import (
    TransformedExample,
)
from nvidia_tao_tf1.core.types import Canvas2D
from nvidia_tao_tf1.core.types import Example


__all__ = (
    "Bbox2DLabel",
    "Canvas2D",
    "Coordinates2D",
    "Coordinates2DWithCounts",
    "empty_polygon_label",
    "Example",
    "FEATURE_CAMERA",
    "FEATURE_SESSION",
    "filter_bbox_label_based_on_minimum_dims",
    "Images2D",
    "Images2DReference",
    "LabelledImages2D",
    "LabelledImages2DReference",
    "LabelledImages2DReferenceVal",
    "set_image_channels",
    "set_image_depth",
    "LABEL_DEPTH_DENSE_MAP",
    "LABEL_DEPTH_FREESPACE",
    "LABEL_FREESPACE_REGRESSION",
    "LABEL_FREESPACE_SEGMENTATION",
    "LABEL_MAP",
    "LABEL_OBJECT",
    "LABEL_PANOPTIC_SEGMENTATION",
    "LABEL_PATH",
    "map_and_stack",
    "map_markers_to_orientations",
    "map_orientation_to_markers",
    "PartitionLabel",
    "Polygon2DLabel",
    "PolygonLabel",
    "SequenceExample",
    "Session",
    "set_augmentations",
    "set_augmentations_val",
    "set_auto_resize",
    "set_image_channels",
    "set_max_side",
    "set_min_side",
    "set_h_tensor",
    "set_h_tensor_val",
    "set_w_tensor",
    "set_w_tensor_val",
    "sparsify_dense_coordinates",
    "test_fixtures",
    "TransformedExample",
    "vector_and_counts_to_sparse_tensor",
)
