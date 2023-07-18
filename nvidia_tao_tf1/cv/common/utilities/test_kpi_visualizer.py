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
"""Test GazeNet kpi visualizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.common.utilities.kpi_visualization import KpiVisualizer


def test_write_csv():
    """Test GazeNet Kpi Visualization on reading csv with correct number of lines and rows"""
    write_csv = True
    path_info = {
        'root_path': '',
        'set_directory_path': ['nvidia_tao_tf1/cv/common/utilities/testdata'],
        'ground_truth_folder_name': [''],
        'ground_truth_file_folder_name': [''],
    }
    kpi_bucket_file = ''
    visualize_set_id = ['testset']
    expected_rows = 1
    expected_columns = 22

    _kpi_visualizer = KpiVisualizer(visualize_set_id, kpi_bucket_file, path_info)

    # kpi_bucket_file will be none on invalid file names.
    assert _kpi_visualizer._kpi_bucket_file is None

    dfTable = _kpi_visualizer(
        output_path='nvidia_tao_tf1/cv/common/utilities/csv',
        write_csv=write_csv
    )

    assert len(dfTable.index) == expected_rows
    assert len(dfTable.columns) == expected_columns
