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
"""Modulus model templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils import get_custom_objects
from nvidia_tao_tf1.core.models.templates import inception_v2_block
from nvidia_tao_tf1.core.models.templates import utils
from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.quantized_conv2dtranspose import QuantizedConv2DTranspose
from nvidia_tao_tf1.core.models.templates.quantized_dense import QuantizedDense
from nvidia_tao_tf1.core.models.templates.quantized_depthwiseconv2d import QuantizedDepthwiseConv2D

__all__ = ("inception_v2_block", "utils")

get_custom_objects()["QuantizedConv2D"] = QuantizedConv2D
get_custom_objects()["QuantizedConv2DTranspose"] = QuantizedConv2DTranspose
get_custom_objects()["QuantizedDepthwiseConv2D"] = QuantizedDepthwiseConv2D
get_custom_objects()["QuantizedDense"] = QuantizedDense
get_custom_objects()["QDQ"] = QDQ
