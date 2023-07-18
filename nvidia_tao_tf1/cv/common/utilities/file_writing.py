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

"""Utility function definitions for writing results to text file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


def write_to_file_xyz(final_errors, num_results, save_dir, mode):
    """Write the final error results to a text file in result directory.

    Args:
        final_errors (list): List of tuples of final errors.
        num_results (int): Number of samples that the errors are calculated from.
        save_dir (str): The directory to output result text file.
        mode (str): Current mode in evaluation, can be 'testing' or 'kpi_testing'.
    """
    output_filename = os.path.join(save_dir, mode+'_error.txt')
    with open(output_filename, 'w') as f:
        f.write('#XYZ evaluation (' + str(num_results) + ' samples)' + '\n\n')
        f.write('RMSE error [cm]: ' + '\n')
        f.write('x, mean: ' + str(round(final_errors[0][0], 2)) +
                ' std: ' + str(round(final_errors[0][1], 2)) + '\n')
        f.write('y, mean: ' + str(round(final_errors[1][0], 2)) +
                ' std: ' + str(round(final_errors[1][1], 2)) + '\n')
        f.write('z, mean: ' + str(round(final_errors[2][0], 2)) +
                ' std: ' + str(round(final_errors[2][1], 2)) + '\n')
        f.write('d_xyz, mean: ' + str(round(final_errors[3][0], 2)) +
                ' std: ' + str(round(final_errors[3][1], 2)) + '\n')
        f.write('d_xy, mean: ' + str(round(final_errors[4][0], 2)) +
                ' std: ' + str(round(final_errors[4][1], 2)) + '\n')
        f.close()


def write_to_file_theta_phi(final_errors, num_results, save_dir, mode):
    """Write the final error results to a text file in result directory.

    Args:
        final_errors (list): List of tuples of final errors.
        num_results (int): Number of samples that the errors are calculated from.
        save_dir (str): The directory to output result text file.
        mode (str): Current mode in evaluation, can be 'testing' or 'kpi_testing'.
    """
    output_filename = os.path.join(save_dir, mode+'_error.txt')
    with open(output_filename, 'a+') as f:
        f.write('#Theta-Phi evaluation (' + str(num_results) + ' samples)' + '\n\n')
        f.write('Cosine gaze vector error: ' + '\n')
        f.write('[radian] mean: ' + str(round(final_errors[0][0], 2)) +
                ' std: ' + str(round(final_errors[0][1], 2)) + '\n')
        f.write('[degree] mean: ' + str(round(final_errors[0][0] * 180 / np.pi, 2)) +
                ' std: ' + str(round(final_errors[0][1] * 180 / np.pi, 2)) + '\n\n\n')

        f.write('Euclidean error [radian]: ' + '\n')
        f.write('theta, mean: ' + str(round(final_errors[1][0], 2)) +
                ' std: ' + str(round(final_errors[1][1], 2)) + '\n')
        f.write('phi, mean: ' + str(round(final_errors[2][0], 2)) +
                ' std: ' + str(round(final_errors[2][1], 2)) + '\n')
        f.write('d_tp, mean: ' + str(round(final_errors[3][0], 2)) +
                ' std: ' + str(round(final_errors[3][1], 2)) + '\n\n')

        f.write('Euclidean error [degree]: ' + '\n')
        f.write('theta, mean: ' + str(round(final_errors[1][0] * 180 / np.pi, 2)) +
                ' std: ' + str(round(final_errors[1][1] * 180 / np.pi, 2)) + '\n')
        f.write('phi, mean: ' + str(round(final_errors[2][0] * 180 / np.pi, 2)) +
                ' std: ' + str(round(final_errors[2][1] * 180 / np.pi, 2)) + '\n')
        f.write('d_tp, mean: ' + str(round(final_errors[3][0] * 180 / np.pi, 2)) +
                ' std: ' + str(round(final_errors[3][1] * 180 / np.pi, 2)) + '\n\n\n')

        f.write('Projected error in cm in screen phy space (at 100 cm): ' + '\n')
        user_dist = 100.0
        f.write('x, mean:' + str(round(np.tan(final_errors[2][0]) * user_dist, 2)) +
                ' std: ' + str(round(np.tan(final_errors[2][1]) * user_dist, 2)) + '\n')
        f.write('y, mean:' + str(round(np.tan(final_errors[1][0]) * user_dist, 2)) +
                ' std: ' + str(round(np.tan(final_errors[1][1]) * user_dist, 2)) + '\n')
        mean_d_xy = np.sqrt(np.power(np.tan(final_errors[2][0]) * user_dist, 2) +
                            np.power(np.tan(final_errors[1][0]) * user_dist, 2))
        std_d_xy = np.sqrt(np.power(np.tan(final_errors[2][1]) * user_dist, 2) +
                           np.power(np.tan(final_errors[1][1]) * user_dist, 2))
        f.write('d_xy, mean:' + str(round(mean_d_xy, 2)) +
                ' std: ' + str(round(std_d_xy, 2)) + '\n')
        f.close()


def write_to_file_joint(final_errors, num_results_xyz, num_results_tp, save_dir, mode):
    """Write the final error results to a text file in result directory.

    Args:
        final_errors (list): List of tuples of final errors.
        num_results_xyz (int): Number of xyz samples that the errors are calculated from.
        num_results_tp (int): Number of tp samples that the errors are calculated from.
        save_dir (str): The directory to output result text file.
        mode (str): Current mode in evaluation, can be 'testing' or 'kpi_testing'.
    """
    write_to_file_xyz(final_errors, num_results_xyz, save_dir, mode)
    output_filename = os.path.join(save_dir, mode + '_error.txt')
    f = open(output_filename, 'a')
    f.write('--------------------------------------' + '\n')
    f.write('--------------------------------------' + '\n\n')
    f.close()
    write_to_file_theta_phi(final_errors[5:], num_results_tp, save_dir, mode)
