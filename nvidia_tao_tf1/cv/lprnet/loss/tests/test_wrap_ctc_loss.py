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
"""test lprnet loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.cv.lprnet.loss.wrap_ctc_loss import WrapCTCLoss


def test_loss():
    loss = WrapCTCLoss(8)
    y_true = [[0, 0, 0, 0, 0, 0, 0, 0, 24, 8]]
    y_pred_one = [1.0, 0, 0]
    y_pred = [[y_pred_one for _ in range(24)]]

    with tf.Session() as sess:
        sess.run(loss.compute_loss(tf.constant(y_true), tf.constant(y_pred)))
