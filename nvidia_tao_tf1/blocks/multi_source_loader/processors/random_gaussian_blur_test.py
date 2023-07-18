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

"""Tests for RandomRotation processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_gaussian_blur import (
    RandomGaussianBlur,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestRandomGaussianBlur(ProcessorTestCase):
    def test_forwards_init_value_errors_to_caller(self):
        with pytest.raises(ValueError):
            RandomGaussianBlur(
                min_filter_size=5, max_filter_size=2, max_stddev=1.0, probability=2.0
            )

    def test_get_filters(self):
        random_gaussian_blur = RandomGaussianBlur(
            min_filter_size=5, max_filter_size=5, max_stddev=1.0, probability=1.0
        )
        with tf.compat.v1.Session() as sess:
            np_gaussian_filter_list = sess.run(random_gaussian_blur.get_filters())

        filter_list_length = len(np_gaussian_filter_list)
        filter_sum_0 = np.sum(np_gaussian_filter_list[0])
        filter_sum_1 = np.sum(np_gaussian_filter_list[1])
        filter_0_shape = np_gaussian_filter_list[0].shape
        filter_1_shape = np_gaussian_filter_list[1].shape

        np.testing.assert_equal(filter_list_length, 2)
        np.allclose(filter_sum_0, 1.0, atol=1e-6)
        np.allclose(filter_sum_1, 1.0, atol=1e-6)
        np.testing.assert_equal(filter_0_shape, (5, 1))
        np.testing.assert_equal(filter_1_shape, (1, 5))

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        random_gaussian_blur = RandomGaussianBlur(
            min_filter_size=5, max_filter_size=5, max_stddev=1.0, probability=1.0
        )
        random_gaussian_blur_dict = random_gaussian_blur.serialize()
        deserialized_dict = deserialize_tao_object(random_gaussian_blur_dict)
        random_gaussian_blur._min_filter_size == deserialized_dict._min_filter_size
        random_gaussian_blur._max_filter_size == deserialized_dict._max_filter_size
