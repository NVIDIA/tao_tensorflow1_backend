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
"""Processor for applying random translation augmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.gaussian_kernel import (
    gaussian_kernel,
)


@pytest.mark.parametrize(
    "size, mean, stddev, expected_value",
    [
        [2, 0.0, None, [0.7655643, 0.23443568]],
        [2, 0.0, 1.0, [0.62245935, 0.37754068]],
        [1, 0.0, None, [1.0]],
        [3, 0.0, None, [0.23899426, 0.52201146, 0.23899426]],
        [3, 0.0, 5.0, [0.33110374, 0.3377925, 0.33110374]],
    ],
)
def test_gaussian_kernel(size, mean, stddev, expected_value):
    value = gaussian_kernel(size, mean, stddev)
    with tf.compat.v1.Session() as sess:
        np_value = sess.run(value)
    np.testing.assert_array_equal(
        np_value.transpose(1, 0),
        np.expand_dims(np.asarray(expected_value, np.float32), axis=0),
    )
