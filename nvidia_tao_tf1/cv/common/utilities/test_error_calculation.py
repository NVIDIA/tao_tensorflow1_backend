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
"""Test utility script: error calculation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from nvidia_tao_tf1.cv.common.utilities.error_calculation \
    import compute_error_joint, compute_error_theta_phi, compute_error_xyz


def sample_results():
    """Return a sample xyz results list."""
    sample = [[313.0, 94.0, -173.0, 300.0, 82.0, -123.0],
              [-110.0, -354.0, -51.0, -8.0, -360.0, -63.0],
              [-800.0, -166.0, -106.0, -726.0, -132.0, 2.0],
              [-523.0, 67.0, -169.0, -478.0, 123.0, -31.0]]

    return sample


def sample_results_theta_phi():
    """Return a sample theta-phi results list."""
    sample = [[0.0692, 0.5036, 0.1708, 0.5964],
              [0.2982, -0.0896, 0.4051, 0.0851],
              [0.0403, -0.3247, 0.0627, -0.4986],
              [-0.2298, 0.7030, -0.1134, 0.5984]]

    return sample


def sample_results_joint():
    """Return a sample joint results list."""
    sample = [[-582.28, -145.09, -111.27, 0.0692, 0.5036, -697.39, -238.32, -64.77, 0.1708, 0.5964],
              [43.85, -333.97, -56.72, 0.2982, -0.0896, -77.05, -441.13, 10.44, 0.4051, 0.0851],
              [258.68, -176.96, -99.21, 0.0403, -0.3247, 445.05, -132.13, 2.12, 0.0627, -0.4986],
              [-867.76, 136.02, -189.95, -0.2298, 0.7030, -864.74, 58.92, -170.83, -0.1134, 0.5984]]

    return sample


def test_compute_error():
    """Test compute_error function."""
    results = sample_results()

    final_errors, num_results = compute_error_xyz(results)

    exp_num_results = 4
    exp_final_errors = [(5.85, 3.3109),
                        (2.70, 1.97230),
                        (7.70, 4.90815),
                        (11.168964, 3.8728855),
                        (6.8286329, 3.1200596)]

    assert exp_num_results == num_results

    for i in range(5):
        np.testing.assert_almost_equal(exp_final_errors[i][0], final_errors[i][0], 2)
        np.testing.assert_almost_equal(exp_final_errors[i][1], final_errors[i][1], 2)

    # Test error calculation for theta_phi.
    results_tp = sample_results_theta_phi()

    final_errors_tp, num_results_tp = compute_error_theta_phi(results_tp)

    exp_final_errors = [(0.1658, 0.0218),
                        (0.0868, 0.0375),
                        (0.1365, 0.038),
                        (0.1685, 0.0248)]

    assert exp_num_results == num_results_tp

    for i in range(4):
        np.testing.assert_almost_equal(exp_final_errors[i][0], final_errors_tp[i][0], 2)
        np.testing.assert_almost_equal(exp_final_errors[i][1], final_errors_tp[i][1], 2)

    # Test error calculation for joint.
    results_joint = sample_results_joint()

    final_errors_joint, num_results_joint, num_results_tp = compute_error_joint(results_joint)

    exp_final_errors = [(10.635, 6.5895),
                        (8.058, 2.322),
                        (5.8527, 3.0017),
                        (15.6632, 4.9776),
                        (14.4632, 4.2027),
                        (0.1658, 0.0218),
                        (0.0868, 0.0375),
                        (0.1365, 0.038),
                        (0.1685, 0.0248)]

    assert exp_num_results == num_results_joint
    assert exp_num_results == num_results_tp

    for i in range(9):
        np.testing.assert_almost_equal(exp_final_errors[i][0], final_errors_joint[i][0], 2)
        np.testing.assert_almost_equal(exp_final_errors[i][1], final_errors_joint[i][1], 2)
