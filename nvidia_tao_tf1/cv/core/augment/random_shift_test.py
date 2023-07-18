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

"""Tests for RandomShift processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.core.augment.random_shift import RandomShift


@pytest.mark.parametrize("shift_percent_max, shift_probability, frame_shape, message",
                         [(-1, 0.5, [10, 10, 3],
                           "RandomShift.shift_percent_max (-1) is not within the range [0, 1]."),
                          (0.05, 1.5, [10, 10, 3],
                           "RandomShift.shift_probability (1.5) is not within the range [0, 1]."),
                          ])
def test_invalid_random_shift_parameters(shift_percent_max, shift_probability, frame_shape,
                                         message):
    """Test RandomShift processor constructor error handling on invalid arguments."""
    with pytest.raises(ValueError) as exc:
        RandomShift(shift_percent_max=shift_percent_max,
                    shift_probability=shift_probability,
                    frame_shape=frame_shape)
    assert message in str(exc)


def test_random_shift_call_with_no_percent():
    """Test RandomShift processor call."""
    op = RandomShift(shift_percent_max=0.0,
                     shift_probability=1.0,
                     frame_shape=[25, 25, 3])
    bbox = {
        'x': tf.constant(10.),
        'y': tf.constant(10.),
        'h': tf.constant(10.),
        'w': tf.constant(10.),
    }
    out_bbox = op(bbox)

    with tf.Session() as sess:
        bbox = sess.run(bbox)
        out_bbox = sess.run(out_bbox)
        for key in bbox:
            assert bbox[key] == 10
            assert bbox[key] == out_bbox[key]


def test_random_shift_call_with_no_prob():
    """Test RandomShift processor call."""
    op = RandomShift(shift_percent_max=1.0,
                     shift_probability=0.0,
                     frame_shape=[25, 25, 3])
    bbox = {
        'x': tf.constant(10.),
        'y': tf.constant(10.),
        'h': tf.constant(10.),
        'w': tf.constant(10.),
    }
    out_bbox = op(bbox)

    with tf.Session() as sess:
        bbox = sess.run(bbox)
        out_bbox = sess.run(out_bbox)
        for key in bbox:
            assert bbox[key] == 10
            assert bbox[key] == out_bbox[key]


def test_random_shift_call_():
    """Test RandomShift processor call."""
    op = RandomShift(shift_percent_max=1.0,
                     shift_probability=1.0,
                     frame_shape=[25, 25, 3])
    bbox = {
        'x': tf.constant(10.),
        'y': tf.constant(10.),
        'h': tf.constant(10.),
        'w': tf.constant(10.),
    }
    out_bbox = op(bbox)

    with tf.Session() as sess:
        bbox = sess.run(bbox)
        out_bbox = sess.run(out_bbox)
        for key in bbox:
            assert bbox[key] == 10
            assert bbox[key] != out_bbox[key]
