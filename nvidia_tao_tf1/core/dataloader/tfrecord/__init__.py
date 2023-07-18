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

"""Modulus dataloader tfrecord module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _bytes_feature
from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _convert_unicode_to_str
from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _float_feature
from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _int64_feature
from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _partition
from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _shard
from nvidia_tao_tf1.core.dataloader.tfrecord.converter_lib import _shuffle

__all__ = (
    "_bytes_feature",
    "_convert_unicode_to_str",
    "_float_feature",
    "_int64_feature",
    "_partition",
    "_shard",
    "_shuffle",
)
