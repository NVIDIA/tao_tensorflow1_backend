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
"""test custom Unet Custom loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.unet.utils.model_fn import dice_coef


def test_loss():
    y_true = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1., 0.]])

    y_pred = np.array([[0, 0, 0.9, 0], [0, 0, 0.1, 0], [1, 1, 0.1, 1.]])
    dice_loss = tf.reduce_mean(1 - dice_coef(tf.constant(y_pred), tf.constant(y_true)),
                               name='dice_loss')
    with tf.Session() as sess:
        assert abs(sess.run(dice_loss)) == 0.5858765827258523
