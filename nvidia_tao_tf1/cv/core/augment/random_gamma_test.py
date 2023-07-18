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

"""Tests for RandomGamma processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from mock import patch

import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.core.augment.random_gamma import RandomGamma


@pytest.mark.parametrize("gamma_type, gamma_mu, gamma_std, gamma_max, gamma_min, gamma_probability,"
                         "message",
                         [('uniform', 1.0, 1.0, 2, -1, 0.5,
                           "RandomGamma.gamma_min (-1) is not positive."),
                          ('uniform', 1.0, 1.0, -0.1, 1, 0.5,
                           "RandomGamma.gamma_max (-0.1) is not positive."),
                          ('uniform', 1.0, 1.0, 0.1, 1, 0.5,
                           "RandomGamma.gamma_max (0.1) is less than RandomGamma.gamma_min (1)."),
                          ('uniform', 1.0, 1.0, 2, 1, -0.1,
                           "RandomGamma.gamma_probability (-0.1) is not within the range [0, 1]."),
                          ('uniform', 1.0, 1.0, 2, 1, 1.1,
                           "RandomGamma.gamma_probability (1.1) is not within the range [0, 1]."),
                          ('uniform', 1.0, 1.0, 2, 2, 0.7,
                           "RandomGamma.gamma_max (2) is equal to RandomGamma.gamma_min (2) but "
                           "is not 1.0."),
                          ('test', 1.0, 1.0, 2, 2, 0.7,
                           "RandomGamma.gamma_type (test) is not one of ['normal', 'uniform']."),
                          ('normal', -1.0, 1.0, 2, 2, 0.7,
                           "RandomGamma.gamma_mu (-1.0) is not positive."),
                          ('normal', 1.0, -1.0, 2, 2, 0.7,
                           "RandomGamma.gamma_std (-1.0) is not positive."),
                          ])
def test_invalid_random_gamma_parameters(gamma_type, gamma_mu, gamma_std, gamma_max, gamma_min,
                                         gamma_probability, message):
    """Test RandomGamma processor constructor error handling on invalid arguments."""
    with pytest.raises(ValueError) as exc:
        RandomGamma(gamma_type=gamma_type,
                    gamma_mu=gamma_mu,
                    gamma_std=gamma_std,
                    gamma_max=gamma_max,
                    gamma_min=gamma_min,
                    gamma_probability=gamma_probability)
    assert message in str(exc)


def test_random_gamma_call_with_same_gamma_one():
    """Test RandomGamma processor call."""
    op = RandomGamma(gamma_type='uniform',
                     gamma_mu=1.0,
                     gamma_std=0.3,
                     gamma_max=1,
                     gamma_min=1,
                     gamma_probability=1.0)
    image = tf.ones([10, 10, 3]) * 2
    out_image = op(image)
    assert_op = tf.assert_equal(image, out_image)

    with tf.Session() as sess:
        sess.run(assert_op)


@patch("nvidia_tao_tf1.cv.core.augment.random_gamma.tf.image.adjust_gamma",
       side_effect=tf.image.adjust_gamma)
@patch("moduluspy.nvidia_tao_tf1.core.processors.augment.random_glimpse.tf.random_uniform")
def test_delegates_to_tf_image_adjust_gamma_uniform(mocked_random_uniform, spied_adjust_gamma):
    mocked_random_uniform.return_value = tf.constant(2, dtype=tf.float32)

    op = RandomGamma(gamma_type='uniform',
                     gamma_mu=1.0,
                     gamma_std=0.3,
                     gamma_max=2,
                     gamma_min=1,
                     gamma_probability=1.0)
    image = tf.ones([10, 10, 3]) * 2
    op(image)
    spied_adjust_gamma.assert_called_with(image, gamma=mocked_random_uniform.return_value)
