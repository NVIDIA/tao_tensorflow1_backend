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
"""test YOLO v4 input encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.yolo_v4.dataio.input_encoder import (
    YOLOv4InputEncoder,
    YOLOv4InputEncoderTensor
)


def test_input_encoder():
    """test input encoder."""
    encoder = YOLOv4InputEncoder(3)
    labels = np.array([
        [0, 0, 1, 45, 293, 272],
        [1, 0, 54, 24, 200, 226],
        [2, 0, 54, 126, 200, 226]],
        dtype=np.float32
    )
    encoded = encoder((320, 320), labels)
    print(encoded.shape)
    assert encoded.shape[-1] == 10


def test_input_encoder_tensor():
    """test input encoder with tf.data dataset."""
    encoder = YOLOv4InputEncoderTensor(
        320, 320, 3,
        feature_map_size=np.array([[10, 10], [20, 20], [40, 40]], dtype=np.int32),
        anchors=np.array([[(0.279, 0.216), (0.375, 0.476), (0.897, 0.784)],
                         [(0.072, 0.147), (0.149, 0.108), (0.142, 0.286)],
                         [(0.024, 0.031), (0.038, 0.072), (0.079, 0.055)]],
                         dtype=np.float32)
    )
    labels = np.array([
        [1, 0, 1, 45, 293, 272],
        [0, 0, 54, 24, 200, 226],
        [2, 0, 54, 126, 200, 226]],
        dtype=np.float32
    )
    labels = [tf.constant(labels, dtype=tf.float32)]
    encoded = encoder(labels)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(0)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    encoded_np = sess.run(encoded)
    assert encoded_np.shape[-1] == 10
