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

"""Tests for RandomBlur processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from mock import patch

import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.core.augment.random_blur import RandomBlur


@pytest.mark.parametrize("blur_choices, blur_probability, channels, message",
                         [([1, 2, 3], 0.5, 1, "RandomBlur.blur_choices ([1, 2, 3]) contains "
                                              "an even kernel size (2)."),
                          ([1, -1, 3], 0.5, 1, "RandomBlur.blur_choices ([1, -1, 3]) contains "
                                               "an invalid kernel size (-1)."),
                          ([1, 5, 3], 1.5, 1, "RandomBlur.blur_probability (1.5) is not "
                                              "within the range [0, 1]."),
                          ([1, 2, 3], 0.5, 2, "RandomBlur.blur_choices ([1, 2, 3]) contains an "
                                              "even kernel size (2)."),
                          ([1, -1, 3], 0.5, 2, "RandomBlur.blur_choices ([1, -1, 3]) contains an "
                           "invalid kernel size (-1)."),
                          ([1, 5, 3], 1.5, 2, "RandomBlur.blur_probability (1.5) is not within "
                           "the range [0, 1]."),
                          ([1, 2, 3], 0.5, 3, "RandomBlur.blur_choices ([1, 2, 3]) contains "
                                              "an even kernel size (2)."),
                          ([1, -1, 3], 0.5, 3, "RandomBlur.blur_choices ([1, -1, 3]) contains an "
                           "invalid kernel size (-1)."),
                          ([1, 5, 3], 1.5, 3, "RandomBlur.blur_probability (1.5) is not within "
                           "the range [0, 1]."),
                          ])
def test_invalid_random_blur_parameters(blur_choices, blur_probability, channels, message):
    """Test RandomBlur processor constructor error handling on invalid arguments."""
    with pytest.raises(ValueError) as exc:
        RandomBlur(blur_choices=blur_choices,
                   blur_probability=blur_probability,
                   channels=channels)
    assert message in str(exc)


@pytest.mark.parametrize("channels",
                         [1, 2, 3])
def test_random_blur_call_with_no_blur_choices(channels):
    """Test RandomBlur processor call."""
    op = RandomBlur(blur_choices=[],
                    blur_probability=1.0,
                    channels=channels)
    width = height = 10
    image = tf.ones([height, width, 3]) * 2
    out_image = op(image, height, width)
    assert_op = tf.assert_equal(image, out_image)

    with tf.Session() as sess:
        sess.run(assert_op)


@pytest.mark.parametrize("channels",
                         [1, 2, 3])
@patch("nvidia_tao_tf1.cv.core.augment.random_blur.tf.nn.depthwise_conv2d",
       side_effect=tf.nn.depthwise_conv2d)
def test_delegates_to_tf_depthwise_conv2d(spied_depthwise_conv2d, channels):
    op = RandomBlur(blur_choices=[5],
                    blur_probability=1,
                    channels=channels)
    width = height = 10
    image = tf.ones([height, width, channels]) * 2
    out_image = op(image, height, width)
    spied_depthwise_conv2d.assert_called_once()
    with tf.Session():
        assert tf.shape(out_image).eval().tolist() == [10, 10, channels]
