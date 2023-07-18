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

"""Data sources for ingesting datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.multi_source_loader.sources.bbox_to_polygon import (
    BboxToPolygon,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.data_source import (
    DataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.image_data_source import (
    ImageDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.imbalance_oversampling import (
    ImbalanceOverSamplingStrategy,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.synthetic_data_source import (
    SyntheticDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.tfrecords_data_source import (
    TFRecordsDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.video_data_source import (
    VideoDataSource,
)

__all__ = (
    "DataSource",
    "ImageDataSource",
    "ImbalanceOverSamplingStrategy",
    "BboxToPolygon",
    "SyntheticDataSource",
    "VideoDataSource",
    "TFRecordsDataSource",
)
