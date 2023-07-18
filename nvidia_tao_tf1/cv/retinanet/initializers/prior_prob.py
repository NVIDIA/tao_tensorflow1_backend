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

"""IVA RetinaNet base architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import keras
import numpy as np


class PriorProbability(keras.initializers.Initializer):
    """Apply a prior probability to the weights.

    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, probability=0.01):
        """Set prior probability."""
        self.probability = probability

    def get_config(self):
        """Get probability."""
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        """set bias to -log((1 - p)/p) for foreground."""
        result = np.ones(shape, dtype=dtype) * - math.log((1 - self.probability) / self.probability)

        return result
