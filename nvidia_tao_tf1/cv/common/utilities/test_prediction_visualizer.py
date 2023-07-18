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
"""Test GazeNet prediction visualizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from nvidia_tao_tf1.cv.common.utilities.kpi_visualization import KpiVisualizer
from nvidia_tao_tf1.cv.common.utilities.prediction_visualization import PredictionVisualizer


def test_xyz_model():
    """Test GazeNet Kpi Visualization on reading csv with correct number of lines and rows"""
    write_csv = False
    kpi_bucket_file = ''
    visualize_set_id = ['testset']
    expected_rows = 1
    expected_columns = 22
    model_type = 'xyz'
    bad_users = ''
    buckets_to_visualize = 'all clean hard easy car car_center car_right center-camera ' \
                           'left-center-camera bottom-center-camera top-center-camera ' \
                           'right-center-camera right-camera glasses no-glasses occluded'
    predictions = ['/home/copilot.cosmos10/RealTimePipeline/set/s400_KPI/Data/user5_x/'
                   'frame0_1194_881_vc0_02.png -463.238555908 94.0375442505 -176.797409058'
                   ' -486.134002686 257.865386963 -141.173324585']
    path_info = {
        'root_path': '',
        'set_directory_path': ['nvidia_tao_tf1/cv/common/utilities/testdata'],
        'ground_truth_folder_name': [''],
        'ground_truth_file_folder_name': [''],
    }
    theta_phi_degrees = False

    _kpi_visualizer = KpiVisualizer(visualize_set_id, kpi_bucket_file, path_info)

    # kpi_bucket_file will be none on invalid file names.
    assert _kpi_visualizer._kpi_bucket_file is None

    dfTable = _kpi_visualizer(
        output_path='nvidia_tao_tf1/cv/common/utilities/csv',
        write_csv=write_csv
    )

    assert len(dfTable.index) == expected_rows
    assert len(dfTable.columns) == expected_columns

    _prediction_visualizer = PredictionVisualizer(model_type, visualize_set_id,
                                                  bad_users, buckets_to_visualize, path_info,
                                                  tp_degrees=theta_phi_degrees)

    df_all = _prediction_visualizer._kpi_prediction_linker.getDataFrame(predictions, dfTable)

    expected_num_users = 1
    num_users = len(df_all.drop_duplicates(['user_id'], keep='first'))
    assert num_users == expected_num_users

    expected_avg_cam_err = 16.921240584744147
    avg_cam_err = df_all['cam_error'].mean()
    assert math.isclose(
        avg_cam_err, expected_avg_cam_err,
        abs_tol=0.0, rel_tol=1e-6
    )

    expected_degree_err = 9.225318552646291
    degree_err = df_all['degree_error'].mean()
    assert math.isclose(
        degree_err, expected_degree_err,
        abs_tol=0.0, rel_tol=1e-6
    )


def test_joint_model():
    """Test GazeNet Kpi Visualization on reading csv with correct number of lines and rows"""
    write_csv = True
    kpi_bucket_file = ''
    visualize_set_id = ['testset']
    expected_rows = 1
    expected_columns = 22
    model_type = 'joint'
    bad_users = ''
    buckets_to_visualize = 'all clean hard easy car car_center car_right center-camera ' \
                           'left-center-camera bottom-center-camera top-center-camera ' \
                           'right-center-camera right-camera glasses no-glasses occluded'
    predictions = ['/home/copilot.cosmos10/RealTimePipeline/set/s400_KPI/'
                   'Data/user5_x/frame0_1194_881_vc0_02.png -463.238555908'
                   ' 94.0375442505 -176.797409058 -0.224229842424 0.427173137665'
                   ' -645.198913574 29.1056632996 -205.970291138 -0.116438232362 0.543292880058']
    path_info = {
        'root_path': '',
        'set_directory_path': ['nvidia_tao_tf1/cv/common/utilities/testdata'],
        'ground_truth_folder_name': [''],
        'ground_truth_file_folder_name': [''],
    }
    theta_phi_degrees = False

    _kpi_visualizer = KpiVisualizer(visualize_set_id, kpi_bucket_file, path_info)

    # kpi_bucket_file will be none on invalid file names.
    assert _kpi_visualizer._kpi_bucket_file is None

    dfTable = _kpi_visualizer(
        output_path='nvidia_tao_tf1/cv/common/utilities/csv',
        write_csv=write_csv
    )

    assert len(dfTable.index) == expected_rows
    assert len(dfTable.columns) == expected_columns

    _prediction_visualizer = PredictionVisualizer(model_type, visualize_set_id,
                                                  bad_users, buckets_to_visualize, path_info,
                                                  tp_degrees=theta_phi_degrees)

    for linker in _prediction_visualizer._kpi_prediction_linker:
        df_all = linker.getDataFrame(predictions, dfTable)

        if linker.model_type == 'joint_xyz':
            expected_num_users = 1
            num_users = len(df_all.drop_duplicates(['user_id'], keep='first'))
            assert num_users == expected_num_users

            expected_avg_cam_err = 19.538877238587443
            avg_cam_err = df_all['cam_error'].mean()
            assert math.isclose(
                avg_cam_err, expected_avg_cam_err,
                abs_tol=0.0, rel_tol=1e-6
            )

            expected_degree_err = 9.016554860339454
            degree_err = df_all['degree_error'].mean()
            assert math.isclose(
                degree_err, expected_degree_err,
                abs_tol=0.0, rel_tol=1e-6
            )

        elif linker.model_type == 'joint_tp':
            expected_num_users = 1
            num_users = len(df_all.drop_duplicates(['user_id'], keep='first'))
            assert num_users == expected_num_users

            expected_avg_cam_err = 16.202673528241796
            avg_cam_err = df_all['cam_error'].mean()
            assert math.isclose(
                avg_cam_err, expected_avg_cam_err,
                abs_tol=0.0, rel_tol=1e-6
            )

            expected_degree_err = 9.020981051018254
            degree_err = df_all['degree_error'].mean()
            # @vpraveen: Relaxing this test condition for
            # CI failures.
            assert math.isclose(
                degree_err, expected_degree_err,
                rel_tol=1e-5, abs_tol=0.0
            )

        elif linker.model_type == 'joint':
            expected_num_users = 1
            num_users = len(df_all.drop_duplicates(['user_id'], keep='first'))
            assert num_users == expected_num_users

            expected_avg_cam_err = 17.52980743787898
            avg_cam_err = df_all['cam_error'].mean()
            assert math.isclose(
                avg_cam_err, expected_avg_cam_err,
                abs_tol=0.0, rel_tol=1e-6
            )

            expected_degree_err = 9.553054268825521
            degree_err = df_all['degree_error'].mean()
            assert math.isclose(
                degree_err, expected_degree_err,
                abs_tol=0.0, rel_tol=1e-6
            )


def test_theta_phi_model_radian():
    """Test GazeNet Kpi Visualization on reading csv with correct number of lines and rows"""
    write_csv = True
    kpi_bucket_file = ''
    visualize_set_id = ['testset']
    expected_rows = 1
    expected_columns = 22
    model_type = 'theta_phi'
    bad_users = ''
    buckets_to_visualize = 'all clean hard easy car car_center car_right center-camera ' \
                           'left-center-camera bottom-center-camera top-center-camera ' \
                           'right-center-camera right-camera glasses no-glasses occluded'
    predictions = ['/home/copilot.cosmos10/RealTimePipeline/set/s400_KPI/'
                   'Data/user5_x/frame0_1194_881_vc0_02.png -0.224229842424'
                   ' 0.427173137665 -0.180691748857 0.44389718771']
    path_info = {
        'root_path': '',
        'set_directory_path': ['nvidia_tao_tf1/cv/common/utilities/testdata'],
        'ground_truth_folder_name': [''],
        'ground_truth_file_folder_name': [''],
    }
    theta_phi_degrees = False

    _kpi_visualizer = KpiVisualizer(visualize_set_id, kpi_bucket_file, path_info)

    # kpi_bucket_file will be none on invalid file names.
    assert _kpi_visualizer._kpi_bucket_file is None

    dfTable = _kpi_visualizer(
        output_path='nvidia_tao_tf1/cv/common/utilities/csv',
        write_csv=write_csv
    )

    assert len(dfTable.index) == expected_rows
    assert len(dfTable.columns) == expected_columns

    _prediction_visualizer = PredictionVisualizer(model_type, visualize_set_id,
                                                  bad_users, buckets_to_visualize, path_info,
                                                  tp_degrees=theta_phi_degrees)

    df_all = _prediction_visualizer._kpi_prediction_linker.getDataFrame(predictions, dfTable)

    expected_num_users = 1
    num_users = len(df_all.drop_duplicates(['user_id'], keep='first'))
    assert num_users == expected_num_users

    # @vpraveen: Converting these to isclose and relaxing the
    # test condition due to CI failures.
    expected_avg_cam_err = 5.047768956137818
    avg_cam_err = df_all['cam_error'].mean()
    assert math.isclose(
        avg_cam_err, expected_avg_cam_err,
        abs_tol=0.0, rel_tol=1e-6
    )

    expected_degree_err = 2.683263685977131
    degree_err = df_all['degree_error'].mean()
    # @vpraveen: Converting these to isclose and relaxing the
    # test condition due to CI failures.
    assert math.isclose(
        degree_err, expected_degree_err,
        abs_tol=0.0, rel_tol=1e-6
    )


def test_theta_phi_model_degrees():
    """Test GazeNet Kpi Visualization on reading csv with correct number of lines and rows"""
    write_csv = True
    kpi_bucket_file = ''
    visualize_set_id = ['testset']
    expected_rows = 1
    expected_columns = 22
    model_type = 'theta_phi'
    bad_users = ''
    buckets_to_visualize = 'all clean hard easy car car_center car_right center-camera ' \
                           'left-center-camera bottom-center-camera top-center-camera ' \
                           'right-center-camera right-camera glasses no-glasses occluded'
    predictions = ['/home/copilot.cosmos10/RealTimePipeline/set/s400_KPI/'
                   'Data/user5_x/frame0_1194_881_vc0_02.png -0.224229842424'
                   ' 0.427173137665 -10.3528746023 25.4334353935']
    path_info = {
        'root_path': '',
        'set_directory_path': ['nvidia_tao_tf1/cv/common/utilities/testdata'],
        'ground_truth_folder_name': [''],
        'ground_truth_file_folder_name': [''],
    }
    theta_phi_degrees = True

    _kpi_visualizer = KpiVisualizer(visualize_set_id, kpi_bucket_file, path_info)

    # kpi_bucket_file will be none on invalid file names.
    assert _kpi_visualizer._kpi_bucket_file is None

    dfTable = _kpi_visualizer(
        output_path='nvidia_tao_tf1/cv/common/utilities/csv',
        write_csv=write_csv
    )

    assert len(dfTable.index) == expected_rows
    assert len(dfTable.columns) == expected_columns

    _prediction_visualizer = PredictionVisualizer(model_type, visualize_set_id,
                                                  bad_users, buckets_to_visualize, path_info,
                                                  tp_degrees=theta_phi_degrees)

    df_all = _prediction_visualizer._kpi_prediction_linker.getDataFrame(predictions, dfTable)

    expected_num_users = 1
    num_users = len(df_all.drop_duplicates(['user_id'], keep='first'))
    assert num_users == expected_num_users

    expected_avg_cam_err = 5.04776895621619
    avg_cam_err = df_all['cam_error'].mean()
    assert math.isclose(
        avg_cam_err, expected_avg_cam_err,
        abs_tol=0.0, rel_tol=1e-6
    )

    expected_degree_err = 2.683263686014905
    degree_err = df_all['degree_error'].mean()
    assert math.isclose(
        degree_err, expected_degree_err,
        abs_tol=0.0, rel_tol=1e-6
    )
