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

"""Encapsulate implementation for gaze custom data set."""

from collections import namedtuple
import os
import numpy as np
from nvidia_tao_tf1.cv.common.dataio.set_strategy import SetStrategy
from nvidia_tao_tf1.cv.common.dataio.utils import mkdir


class GazeCustomSetStrategy(SetStrategy):
    """Class encapsulates implementation specific to gaze sets."""

    CustomerPathStruct = namedtuple('CustomerPathStruct',
                                    '''error_path,
                                    data_path,
                                    norm_data_path,
                                    config_path,
                                    label_path,
                                    landmarks_path''')

    def __init__(
        self,
        data_root_path,
        norm_data_folder_name,
        data_strategy_type,
        landmarks_folder_name
    ):
        """Initialize parameters.

        Args:
            set_id (str): Set for which to generate tfrecords.
            experiment_folder_suffix (str): Suffix of experiment folder containing tfrecords.
            tfrecord_folder_name (str): Folder name of folder containing tfrecords.
            gt_folder_name (str): Folder name of folder containing ground truth txt files.
            landmarks_folder_name (str): Folder name of predicted landmarks, or None to disable.
            set_label_sorter (SetLabelSorter object): Object to sort set as DataFactory / Nvhelnet.
        """
        super(GazeCustomSetStrategy, self).__init__(
            landmarks_folder_name=landmarks_folder_name,
            set_id="",
            experiment_folder_suffix="",
            tfrecord_folder_name="",
            gt_folder_name="",
            set_label_sorter="")
        self._use_filtered = False
        self._use_undistort = False
        self._data_root_path = data_root_path
        self._norm_data_folder_name = norm_data_folder_name
        self._strategy_type = data_strategy_type
        self._landmarks_folder_name = landmarks_folder_name

        self._set_source_paths()
        self._set_camera_parameters()

    def _set_camera_parameters(self):
        def _load_cam_intrinsics(self):
            file_path = os.path.join(self._paths.config_path, 'camera_parameters.txt')

            camera_matrix = np.loadtxt(file_path, delimiter=',', max_rows=3)
            # Distortion coeffs has a single value line by line, below camera_matrix.
            distortion_coeffs = np.loadtxt(file_path, skiprows=4).transpose()
            # Only the first 5 distortion coefficients are correct.
            distortion_coeffs = distortion_coeffs[:5]

            theta_phi_distortion_coeffs = distortion_coeffs
            return [camera_matrix, distortion_coeffs, theta_phi_distortion_coeffs]

        def _load_cam_extrinsics(self):
            R_file_path = os.path.join(self._paths.config_path, 'R.txt')
            T_file_path = os.path.join(self._paths.config_path, 'T.txt')

            R = np.loadtxt(R_file_path, delimiter=',')
            T = np.loadtxt(T_file_path)

            extrinsics = (R, T)
            return extrinsics

        def _load_screen_parameters(self):
            scrpW, scrpH = 1920.0, 1080.0  # Default vals.
            # Try to find a config file for the resolution.
            resolution = os.path.join(self._paths.config_path, 'resolution.txt')
            if os.path.isfile(resolution):
                with open(resolution) as f:
                    scrpW = float(f.readline())
                    scrpH = float(f.readline())

            # Check which of board_size or TV_size is available.
            board_size = os.path.join(self._paths.config_path, 'board_size.txt')
            tv_size = os.path.join(self._paths.config_path, 'TV_size')
            if os.path.isfile(board_size):
                if os.path.isfile(tv_size):
                    raise IOError("Both board_size.txt and TV_size exist in {}"
                                  .format(os.path.join(self._paths.config_path)))
                size_file = board_size
            elif os.path.isfile(tv_size):
                size_file = tv_size
            else:
                raise IOError("Neither board_size.txt nor TV_size exists in {}"
                              .format(os.path.join(self._paths.config_path)))
            with open(size_file) as f:
                scrmW = float(f.readline())
                scrmH = float(f.readline())

            screens = (scrmW, scrmH, scrpW, scrpH)
            return screens

        self._cam_intrinsics = _load_cam_intrinsics(self)
        self._cam_extrinsics = _load_cam_extrinsics(self)
        self._screen_params = _load_screen_parameters(self)

    def _get_json_path(self, set_id_path):
        return self._check_paths(set_id_path, [
            'json_datafactory_v2',
            'json_datafactory'
        ])

    def _set_source_paths(self):
        root_path = self._data_root_path
        if os.path.exists(root_path):
            data_path = os.path.join(root_path, 'Data')
            config_path = os.path.join(root_path, 'Config')
            label_path = self._get_json_path(root_path)
            norm_data_path = os.path.join(root_path, self._norm_data_folder_name)
            error_path = os.path.join(root_path, 'errors')
            mkdir(error_path)

            landmarks_path = None
            if self._landmarks_folder_name is not None:
                landmarks_path = os.path.join(root_path, self._landmarks_folder_name)

            self._paths = self.CustomerPathStruct(
                error_path=error_path,
                data_path=data_path,
                norm_data_path=norm_data_path,
                config_path=config_path,
                label_path=label_path,
                landmarks_path=landmarks_path)

    def get_pts(self, pts, frame_width, frame_height):
        """Return undistorted points if required, otherwise return distored points."""
        return pts
