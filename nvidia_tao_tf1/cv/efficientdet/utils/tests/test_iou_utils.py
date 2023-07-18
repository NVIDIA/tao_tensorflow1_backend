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
"""EfficientDet IOU utils tests."""
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.efficientdet.utils import iou_utils


def test_iou_utils():
    tf.enable_eager_execution()
    pb = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                     dtype=tf.float32)
    tb = tf.constant(
        [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=tf.float32)
    zeros = tf.constant([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.float32)

    assert np.allclose(iou_utils.iou_loss(pb, tb, 'iou'), [0.875, 1.])
    assert np.allclose(iou_utils.iou_loss(pb, tb, 'ciou'), [1.408893, 1.548753])
    assert np.allclose(iou_utils.iou_loss(pb, tb, 'diou'), [1.406532, 1.531532])
    assert np.allclose(iou_utils.iou_loss(pb, tb, 'giou'), [1.075000, 1.933333])
    assert np.allclose(iou_utils.iou_loss(pb, zeros, 'giou'), [0, 0])
    assert np.allclose(iou_utils.iou_loss(pb, zeros, 'diou'), [0, 0])
    assert np.allclose(iou_utils.iou_loss(pb, zeros, 'ciou'), [0, 0])
    assert np.allclose(iou_utils.iou_loss(pb, zeros, 'iou'), [0, 0])
