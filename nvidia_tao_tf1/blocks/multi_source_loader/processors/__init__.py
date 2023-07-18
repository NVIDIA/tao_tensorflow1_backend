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

"""Processors for transforming and augmenting data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from nvidia_tao_tf1.blocks.multi_source_loader.processors.asset_loader import (
    AssetLoader,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.bbox_clipper import (
    BboxClipper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.class_attribute_lookup_table import (  # noqa
    ClassAttributeLookupTable,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.class_attribute_mapper import (
    ClassAttributeMapper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.crop import Crop
from nvidia_tao_tf1.blocks.multi_source_loader.processors.filter2d_processor import (
    Filter2DProcessor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.instance_mapper import (
    InstanceMapper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.label_adjustment import (
    LabelAdjustment,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.lossy_crop import (
    LossyCrop,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.multiple_polyline_to_polygon import (  # noqa
    MultiplePolylineToPolygon,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.pipeline import Pipeline
from nvidia_tao_tf1.blocks.multi_source_loader.processors.polygon_rasterizer import (
    PolygonRasterizer,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.polyline_clipper import (
    PolylineClipper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.polyline_to_polygon import (
    PolylineToPolygon,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.priors_generator import (
    PriorsGenerator,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor import (
    Processor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_brightness import (
    RandomBrightness,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_contrast import (
    RandomContrast,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_flip import (
    RandomFlip,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_gaussian_blur import (
    RandomGaussianBlur,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_glimpse import (
    RandomGlimpse,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_hue_saturation import (
    RandomHueSaturation,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_rotation import (
    RandomRotation,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_shear import (
    RandomShear,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_translation import (
    RandomTranslation,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_zoom import (
    RandomZoom,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.rasterize_and_resize import (
    RasterizeAndResize,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.scale import Scale
from nvidia_tao_tf1.blocks.multi_source_loader.processors.sparse_to_dense_polyline import (
    SparseToDensePolyline,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.temporal_batcher import (
    TemporalBatcher,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.transform_processor import (
    TransformProcessor,
)


__all__ = (
    "AssetLoader",
    "BboxClipper",
    "ClassAttributeLookupTable",
    "ClassAttributeMapper",
    "Crop",
    "InstanceMapper",
    "LabelAdjustment",
    "LossyCrop",
    "MultiplePolylineToPolygon",
    "Pipeline",
    "PolylineClipper",
    "PolygonRasterizer",
    "PolylineToPolygon",
    "PriorsGenerator",
    "Processor",
    "RandomBrightness",
    "RandomContrast",
    "RandomFlip",
    "RandomGlimpse",
    "RandomHueSaturation",
    "RandomRotation",
    "RandomTranslation",
    "RandomShear",
    "RandomZoom",
    "RasterizeAndResize",
    "Scale",
    "SparseToDensePolyline",
    "TransformProcessor",
    "RandomGaussianBlur",
    "TemporalBatcher",
    "Filter2DProcessor",
)
