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
"""Modulus application building block: Losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.losses.absolute_difference_loss import AbsoluteDifferenceLoss
from nvidia_tao_tf1.blocks.losses.binary_crossentropy_loss import BinaryCrossentropyLoss
from nvidia_tao_tf1.blocks.losses.loss import Loss
from nvidia_tao_tf1.blocks.losses.mse_loss import MseLoss
from nvidia_tao_tf1.blocks.losses.sparse_softmax_cross_entropy import SparseSoftmaxCrossEntropy

__all__ = (
    "AbsoluteDifferenceLoss",
    "BinaryCrossentropyLoss",
    "Loss",
    "MseLoss",
    "SparseSoftmaxCrossEntropy",
)
