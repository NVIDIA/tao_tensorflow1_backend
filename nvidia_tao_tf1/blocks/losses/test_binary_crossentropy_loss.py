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

"""Test for binary crossentropy loss function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.losses.binary_crossentropy_loss import BinaryCrossentropyLoss
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


def test_serialization_and_deserialization():
    """Test that class is a TAOObject that can be serialized and deserialized."""
    bce = BinaryCrossentropyLoss()
    deserialized_bce = deserialize_tao_object(bce.serialize())
    assert bce.__name__ == deserialized_bce.__name__

    # Test the underlying loss_fn implementation gets serialized/deserialized correctly.
    y_true = tf.constant([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    y_pred = tf.constant([0.9, 0.6, 0.4, 0.7], dtype=np.float32)
    with tf.compat.v1.Session():
        assert bce(y_true, y_pred).eval() == deserialized_bce(y_true, y_pred).eval()
