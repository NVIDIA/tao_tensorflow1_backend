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
"""Test preprocess ops."""
import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.mask_rcnn.ops.preprocess_ops import normalize_image
# import pytest


def test_normalize_image():
    """Test normalize image with torch means."""
    input_height = 120
    input_width = 160
    input_value = 0.0
    dummy_image = tf.constant(input_value, shape=[input_height, input_width, 3])

    dummy_norm = normalize_image(dummy_image)

    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        output = sess.run(dummy_norm)
        assert np.allclose(output[0, 0, :], [-2.1651785, -2.0357141, -1.8124999])
