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

"""Utility function definitions for error calculations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def rmse_2D(gt, pred):
    """Compute the two-dimensional rmse using ground truth and prediction results.

    Args:
        gt (Nx2 array): Array of ground truth for each data point.
        pred (Nx2 array): Array of prediction for each data point.
    Returns:
        mean_diff (1x2 array): Mean error in each dimension.
        std_diff (1x2 array): Std error in each dimension.
        mean_rmse_xy (float): Mean rmse error.
        std_rmse_xy (float): Std rmse error.
    """
    mean_diff = np.mean(np.absolute(gt - pred), axis=0)
    std_diff = np.std(np.absolute(gt - pred), axis=0)

    mean_rmse_xy = np.mean(np.sqrt(np.sum(np.square(gt - pred), axis=1)))
    std_rmse_xy = np.std(np.sqrt(np.sum(np.square(gt - pred), axis=1)))

    return mean_diff, std_diff, mean_rmse_xy, std_rmse_xy


def rmse_3D(gt, pred):
    """Compute the three-dimensional rmse using ground truth and prediction results.

    Args:
        gt (Nx3 array): Array of ground truth for each data point.
        pred (Nx3 array): Array of prediction for each data point.
    Returns:
        mean_diff (1x3 array): Mean error in each dimension.
        std_diff (1x3 array): Std error in each dimension.
        mean_rmse_xyz (float): Mean 3D rmse error.
        std_rmse_xyz (float): Std 3D rmse error.
        mean_rmse_xy (float): Mean 2D rmse error.
        std_rmse_xy (float): Std 2D rmse error.
    """
    mean_diff = np.mean(np.absolute(gt - pred), axis=0)
    std_diff = np.std(np.absolute(gt - pred), axis=0)

    mean_rmse_xyz = np.mean(np.sqrt(np.sum(np.square(gt - pred), axis=1)))
    std_rmse_xyz = np.std(np.sqrt(np.sum(np.square(gt - pred), axis=1)))

    mean_rmse_xy = np.mean(np.sqrt(np.sum(np.square(gt[:, :2] - pred[:, :2]), axis=1)))
    std_rmse_xy = np.std(np.sqrt(np.sum(np.square(gt[:, :2] - pred[:, :2]), axis=1)))

    return mean_diff, std_diff, mean_rmse_xyz, std_rmse_xyz, mean_rmse_xy, std_rmse_xy


def compute_error_xyz(results):
    """Compute the final error for xyz model using ground truth and prediction results.

    Args:
        results (array): List of ground truth and prediction for each data point.
    Returns:
        final_errors (list): Computed errors.
        num_results (int): Number of evaluated samples.
    """
    if isinstance(results, list):
        results = np.asarray(results)
    results /= 10.

    # Calculate the rmse error.
    mean_diff, std_diff, mean_rmse_xyz, std_rmse_xyz, mean_rmse_xy, std_rmse_xy = \
        rmse_3D(results[:, :3], results[:, 3:])

    final_errors = [(mean_diff[0], std_diff[0]),
                    (mean_diff[1], std_diff[1]),
                    (mean_diff[2], std_diff[2])]

    final_errors.append((mean_rmse_xyz, std_rmse_xyz))
    final_errors.append((mean_rmse_xy, std_rmse_xy))
    num_results = results.shape[0]

    return final_errors, num_results


def compute_error_theta_phi(results):
    """Compute the final error for theta-phi model using ground truth and prediction results.

    Args:
        results (array): List of ground truth and prediction for each data point.
    Returns:
        final_errors (list): Computed errors.
        num_results (int): Number of evaluated samples.
    """
    if isinstance(results, list):
        results = np.asarray(results)

    # Remove samples without theta-phi ground truth.
    results = results[results[:, 0] != -1.0]

    # 1) Calculate the angular gaze vector error.
    gt_gv = np.zeros(shape=(results.shape[0], 3), dtype=results.dtype)
    gt_gv[:, 0] = -np.cos(results[:, 0]) * np.sin(results[:, 1])
    gt_gv[:, 1] = -np.sin(results[:, 0])
    gt_gv[:, 2] = -np.cos(results[:, 0]) * np.cos(results[:, 1])

    pr_gv = np.zeros(shape=(results.shape[0], 3), dtype=results.dtype)
    pr_gv[:, 0] = -np.cos(results[:, 2]) * np.sin(results[:, 3])
    pr_gv[:, 1] = -np.sin(results[:, 2])
    pr_gv[:, 2] = -np.cos(results[:, 2]) * np.cos(results[:, 3])

    err_cos = np.arccos(np.sum(gt_gv[:, :3] * pr_gv[:, :3], axis=1))

    final_errors = [(np.mean(err_cos), np.std(err_cos))]

    # 2) Calculate the theta-phi rmse error.
    mean_tp_diff, std_tp_diff, mean_rmse_tp, std_rmse_tp = rmse_2D(results[:, :2], results[:, 2:])

    final_errors.append((mean_tp_diff[0], std_tp_diff[0]))
    final_errors.append((mean_tp_diff[1], std_tp_diff[1]))
    final_errors.append((mean_rmse_tp, std_rmse_tp))

    num_results = results.shape[0]

    return final_errors, num_results


def compute_error_joint(results):
    """Compute the final error using ground truth and prediction results.

    Args:
        results (array): List of ground truth and prediction for each data point.
    Returns:
        final_errors (list): Computed errors.
        num_results_xyz (int): Number of evaluated samples for xyz.
        num_results_tp (int): Number of evaluated samples for theta-phi.
    """
    if isinstance(results, list):
        results = np.asarray(results)

    # XYZ Error Calculation.
    val_xyz = np.concatenate((results[:, 0:3], results[:, 5:8]), axis=1)
    final_errors_xyz, num_results_xyz = compute_error_xyz(val_xyz)

    # Theta-Phi Error Calculation.
    val_tp = np.concatenate((results[:, 3:5], results[:, 8:10]), axis=1)
    final_errors_tp, num_results_tp = compute_error_theta_phi(val_tp)

    final_errors = final_errors_xyz + final_errors_tp

    return final_errors, num_results_xyz, num_results_tp
