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
"""Modulus export APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

from nvidia_tao_tf1.core.export import caffe
from nvidia_tao_tf1.core.export._onnx import keras_to_onnx
from nvidia_tao_tf1.core.export._uff import keras_to_pb, keras_to_uff
from nvidia_tao_tf1.core.export.caffe import keras_to_caffe
from nvidia_tao_tf1.core.export.data import TensorFile


# Below lazily calls some functions that depend on TensorRT.
# TensorRT currently has a bug where it takes up memory upon importing, so we want to defer the
# import of tensorrt to when it is actually used.
# TODO(xiangbok): Remove lazy calling when fixed in TensorRT (bugfix not yet in release).
class LazyModuleMethodCall(object):
    def __init__(self, name, attr):
        self._name = name
        self._attr = attr

    def __call__(self, *args, **kwargs):
        module = importlib.import_module(name=self._name)
        return getattr(module, self._attr)(*args, **kwargs)


keras_to_tensorrt = LazyModuleMethodCall(
    "nvidia_tao_tf1.core.export._tensorrt", "keras_to_tensorrt"
)
load_tensorrt_engine = LazyModuleMethodCall(
    "nvidia_tao_tf1.core.export._tensorrt", "load_tensorrt_engine"
)
tf_to_tensorrt = LazyModuleMethodCall("nvidia_tao_tf1.core.export._tensorrt", "tf_to_tensorrt")

__all__ = (
    "caffe",
    "keras_to_caffe",
    "keras_to_onnx",
    "keras_to_pb",
    "keras_to_tensorrt",
    "keras_to_uff",
    "load_tensorrt_engine",
    "TensorFile",
    "tf_to_tensorrt",
)
