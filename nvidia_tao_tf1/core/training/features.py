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

"""Helper functions for for determinism and mixed precision run-time env var setup."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def enable_deterministic_training():
    """Enable deterministic training."""
    os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
    os.environ["TF_DETERMINISTIC_OPS"] = "true"
    os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"


def enable_mixed_precision():
    """Enable mixed precision for training."""
    # Notes from Nvidia public guide
    # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#tfamp
    # Read the Caveats section carefully before enabling!!!
    # Multi-GPU:
    # Automatic mixed precision does not currently support TensorFlow
    # Distributed Strategies. Instead, multi-GPU training needs to be with Horovod
    # (or TensorFlow device primitives). We expect this restriction to be relaxed in
    # a future release.
    # For additional control, use flags:
    # TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
    # TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING=1
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
