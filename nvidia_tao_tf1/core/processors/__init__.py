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
"""Modulus processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# # TODO(xiangbok): Split processors into their own seperate files.
from nvidia_tao_tf1.core.processors import augment
from nvidia_tao_tf1.core.processors.augment.color import ColorTransform 
from nvidia_tao_tf1.core.processors.augment.crop import Crop
from nvidia_tao_tf1.core.processors.augment.random_brightness import RandomBrightness
from nvidia_tao_tf1.core.processors.augment.random_contrast import RandomContrast
from nvidia_tao_tf1.core.processors.augment.random_flip import RandomFlip
from nvidia_tao_tf1.core.processors.augment.random_glimpse import RandomGlimpse
from nvidia_tao_tf1.core.processors.augment.random_hue_saturation import RandomHueSaturation
from nvidia_tao_tf1.core.processors.augment.random_rotation import RandomRotation
from nvidia_tao_tf1.core.processors.augment.random_shear import RandomShear
from nvidia_tao_tf1.core.processors.augment.random_translation import RandomTranslation
from nvidia_tao_tf1.core.processors.augment.random_zoom import RandomZoom
from nvidia_tao_tf1.core.processors.augment.scale import Scale
from nvidia_tao_tf1.core.processors.augment.spatial import PolygonTransform
from nvidia_tao_tf1.core.processors.augment.spatial import SpatialTransform
from nvidia_tao_tf1.core.processors.bbox_rasterizer import BboxRasterizer
# from nvidia_tao_tf1.core.processors.binary_to_distance import BinaryToDistance
# from nvidia_tao_tf1.core.processors.buffers import NamedTupleStagingArea
# from nvidia_tao_tf1.core.processors.buffers import TensorflowBuffer
from nvidia_tao_tf1.core.processors.clip_polygon import ClipPolygon
# from nvidia_tao_tf1.core.processors.cluster_one_sweep import ClusterOneSweep
# from nvidia_tao_tf1.core.processors.compute_pr_from_computed_dist import ComputePRFromDecodedDist
# from nvidia_tao_tf1.core.processors.dataset import GroupByWindowKeyDataset
# from nvidia_tao_tf1.core.processors.dataset import SqlDatasetV2
# from nvidia_tao_tf1.core.processors.dataset import VariableBatchDataset
# from nvidia_tao_tf1.core.processors.decode_dist import DecodeDist
from nvidia_tao_tf1.core.processors.decode_image import DecodeImage
# from nvidia_tao_tf1.core.processors.dense_map_summary import DenseMapSummary
# from nvidia_tao_tf1.core.processors.draw_polygon_outlines import DrawPolygonOutlines
# from nvidia_tao_tf1.core.processors.generate_dist_from_bezier import GenerateDistFromBezier
# from nvidia_tao_tf1.core.processors.generate_dist_from_lineseg import GenerateDistFromLineseg
# from nvidia_tao_tf1.core.processors.generate_lineseg_from_polygon import GenerateLinesegFromPolygon
# from nvidia_tao_tf1.core.processors.image_loader import ImageLoader
from nvidia_tao_tf1.core.processors.load_file import LoadFile
from nvidia_tao_tf1.core.processors.lookup_table import LookupTable
# from nvidia_tao_tf1.core.processors.merge_polylines import MergePolylines
# from nvidia_tao_tf1.core.processors.mix_up import MixUp
from nvidia_tao_tf1.core.processors.parse_example_proto import ParseExampleProto
# from nvidia_tao_tf1.core.processors.path_generator import PathGenerator
# from nvidia_tao_tf1.core.processors.pipeline import Pipeline
from nvidia_tao_tf1.core.processors.polygon_rasterizer import PolygonRasterizer

from nvidia_tao_tf1.core.processors.polygon_rasterizer import SparsePolygonRasterizer
from nvidia_tao_tf1.core.processors.processors import boolean_mask_sparse_tensor
from nvidia_tao_tf1.core.processors.processors import dense_to_sparse
from nvidia_tao_tf1.core.processors.processors import json_arrays_to_tensor
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.processors.processors import remove_empty_rows_from_sparse_tensor
from nvidia_tao_tf1.core.processors.processors import (
    sparse_coordinate_feature_to_vertices_and_counts,
)
from nvidia_tao_tf1.core.processors.processors import string_lower, string_upper
from nvidia_tao_tf1.core.processors.processors import to_dense_if_sparse_tensor_is_fully_dense
from nvidia_tao_tf1.core.processors.processors import values_and_count_to_sparse_tensor
from nvidia_tao_tf1.core.processors.tfrecords_iterator import TFRecordsIterator

from nvidia_tao_tf1.core.processors.transformers import ColorTransformer
from nvidia_tao_tf1.core.processors.transformers import SpatialTransformer

__all__ = (
    "augment",
    "ColorTransform",
    "PolygonTransform",
    "SpatialTransform",
    # "NamedTupleStagingArea",
    # "TensorflowBuffer",
    # "GroupByWindowKeyDataset",
    # "VariableBatchDataset",
    # "SqlDatasetV2",
    "BboxRasterizer",
    # "BinaryToDistance",
    "boolean_mask_sparse_tensor",
    "ClipPolygon",
    # "ClusterOneSweep",
    "ColorTransformer",
    # "ComputePRFromDecodedDist",
    "Crop",
    # "DecodeDist",
    "DecodeImage",
    "dense_to_sparse",
    # "DenseMapSummary",
    # "DrawPolygonOutlines",
    # "GenerateDistFromBezier",
    # "GenerateDistFromLineseg",
    # "GenerateLinesegFromPolygon",
    "json_arrays_to_tensor",
    # "ImageLoader",
    "LoadFile",
    "LookupTable",
    # "MergePolylines",
    # "MixUp",
    "ParseExampleProto",
    # "PathGenerator",
    # "Pipeline",
    "PolygonRasterizer",
    "Processor",
    "remove_empty_rows_from_sparse_tensor",
    "RandomBrightness",
    "RandomContrast",
    "RandomFlip",
    "RandomGlimpse",
    "RandomHueSaturation",
    "RandomRotation",
    "RandomTranslation",
    "RandomShear",
    "RandomZoom",
    "Scale",
    "SpatialTransformer",
    "sparse_coordinate_feature_to_vertices_and_counts",
    "SparsePolygonRasterizer",
    "string_lower",
    "string_upper",
    "TFRecordsIterator",
    "to_dense_if_sparse_tensor_is_fully_dense",
    "values_and_count_to_sparse_tensor",
)
