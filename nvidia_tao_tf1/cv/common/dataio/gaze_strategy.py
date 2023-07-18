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

"""Encapsulate implementation for gaze sets."""

import os
import cv2
import numpy as np
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.common.dataio.set_strategy import SetStrategy
from nvidia_tao_tf1.cv.common.dataio.utils import is_int, mkdir


class GazeStrategy(SetStrategy):
    """Class encapsulates implementation specific to gaze sets."""

    gaze_parent_set_path = '/home/copilot.cosmos10/RealTimePipeline/set'
    gaze_error_path = '/home/copilot.cosmos10/RealTimePipeline/errors'

    gaze_cosmos639_parent_path = '/home/driveix.cosmos639/GazeData'
    gaze_cosmos639_filter_path = '/home/driveix.cosmos639/GazeData/filteredFrames'

    def __init__(
        self,
        set_id,
        experiment_folder_suffix,
        tfrecord_folder_name,
        gt_folder_name,
        use_filtered,
        use_undistort,
        landmarks_folder_name,
        set_label_sorter,
        set_root_path,
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
        super(GazeStrategy, self).__init__(
            set_id,
            experiment_folder_suffix,
            tfrecord_folder_name,
            gt_folder_name,
            landmarks_folder_name,
            set_label_sorter)
        self._use_filtered = use_filtered
        self._use_undistort = use_undistort
        if len(set_root_path) == 0:
            self._gaze_parent_set_path = self.gaze_parent_set_path
        else:
            self._gaze_parent_set_path = set_root_path

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
            extrinsics = {}
            for region_name in self._paths.regions:
                R_file_path = os.path.join(self._paths.config_path, region_name, 'R.txt')
                T_file_path = os.path.join(self._paths.config_path, region_name, 'T.txt')

                R = np.loadtxt(R_file_path, delimiter=',')
                T = np.loadtxt(T_file_path)

                extrinsics[region_name] = (R, T)
            return extrinsics

        def _load_screen_parameters(self):
            screens = {}
            for region_name in self._paths.regions:
                scrpW, scrpH = 1920.0, 1080.0  # Default vals.
                # Try to find a config file for the resolution.
                resolution = os.path.join(self._paths.config_path, region_name, 'resolution.txt')
                if os.path.isfile(resolution):
                    with open(resolution) as f:
                        scrpW = float(f.readline())
                        scrpH = float(f.readline())

                # Check which of board_size or TV_size is available.
                board_size = os.path.join(self._paths.config_path, region_name, 'board_size.txt')
                tv_size = os.path.join(self._paths.config_path, region_name, 'TV_size')
                if os.path.isfile(board_size):
                    if os.path.isfile(tv_size):
                        raise IOError("Both board_size.txt and TV_size exist in {}"
                                      .format(os.path.join(self._paths.config_path, region_name)))
                    size_file = board_size
                elif os.path.isfile(tv_size):
                    size_file = tv_size
                else:
                    raise IOError("Neither board_size.txt nor TV_size exists in {}"
                                  .format(os.path.join(self._paths.config_path, region_name)))
                with open(size_file) as f:
                    scrmW = float(f.readline())
                    scrmH = float(f.readline())

                screens[region_name] = (scrmW, scrmH, scrpW, scrpH)
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
        def _get_region_names(config_path):
            """Read the region names from the folders in config."""
            if not os.path.exists(config_path):
                raise IOError('{} doesnot exist' .format(config_path))
            configs = os.listdir(config_path)
            regions = [
                config
                for config in configs
                if os.path.isdir(os.path.join(config_path, config))
                and config.startswith('region_')
            ]
            if not regions:
                # On bench collection has no regions.
                if 'incar' in self._set_id:
                    raise IOError('In car data set should have region_ folders in {}'
                                  .format(config_path))
                return ['']
            if 'incar' not in self._set_id:
                raise IOError('On bench data set should not have region_ folders in {}'
                              .format(config_path))
            return regions

        old_cosmos_set_path = os.path.join(
            self._gaze_parent_set_path, self._set_id)
        if os.path.exists(old_cosmos_set_path):
            strategy_type, info_source_path, self.experiment_folder_name = \
                self._set_label_sorter.get_info_source_path(
                    self._get_json_path,
                    old_cosmos_set_path,
                    old_cosmos_set_path)

            lm_path = self._get_landmarks_path(old_cosmos_set_path)
            if lm_path is not None:
                self.experiment_folder_name = self.fpe_expr_folder \
                                              + self._experiment_folder_suffix

            experiment_folder_path = os.path.join(
                old_cosmos_set_path, self.experiment_folder_name)
            mkdir(experiment_folder_path)

            data_path = os.path.join(old_cosmos_set_path, 'Data')
            if self._use_undistort:
                data_path = os.path.join(old_cosmos_set_path, 'undistortedData')
            config_path = os.path.join(old_cosmos_set_path, 'Config')
            regions = _get_region_names(config_path)

            self._strategy_type, self._paths = strategy_type, self.PathStruct(
                error_path=os.path.join(self.gaze_error_path),
                data_path=data_path,
                config_path=config_path,
                tfrecord_path=os.path.join(
                    old_cosmos_set_path,
                    self.experiment_folder_name,
                    self._tfrecord_folder_name),
                gt_path=os.path.join(
                    old_cosmos_set_path,
                    self.experiment_folder_name,
                    self._gt_folder_name),
                info_source_path=info_source_path,
                filtered_path=None,
                landmarks_path=lm_path,
                regions=regions)
        else:
            new_cosmos_base_path = self.gaze_cosmos639_parent_path
            new_cosmos_raw_path = os.path.join(
                new_cosmos_base_path, 'orgData', self._set_id)
            new_cosmos_generated_path = os.path.join(
                new_cosmos_base_path, 'postData', self._set_id)

            if (
                not os.path.exists(new_cosmos_raw_path) or
                not os.path.exists(new_cosmos_generated_path)
            ):
                raise IOError('Unable to find a data source')

            strategy_type, info_source_path, self.experiment_folder_name = \
                self._set_label_sorter.get_info_source_path(
                    self._get_json_path,
                    new_cosmos_raw_path,
                    new_cosmos_generated_path)
            data_path = os.path.join(new_cosmos_raw_path, 'pngData')
            if not os.path.exists(data_path):
                data_path = os.path.join(new_cosmos_raw_path, 'Data')
            if self._use_undistort:
                data_path = os.path.join(new_cosmos_raw_path, 'undistortedData')
            config_path = os.path.join(new_cosmos_raw_path, 'Config')
            regions = _get_region_names(config_path)

            filtered_path = None
            if self._use_filtered:
                filtered_path = os.path.join(
                    self.gaze_cosmos639_filter_path,
                    self._set_id.rsplit('-', 1)[0])

            lm_path = self._get_landmarks_path(new_cosmos_generated_path)
            if lm_path is not None:
                self.experiment_folder_name = self.fpe_expr_folder + self._experiment_folder_suffix

            experiment_folder_path = os.path.join(
                new_cosmos_generated_path,
                self.experiment_folder_name)
            mkdir(experiment_folder_path)

            self._strategy_type, self._paths = strategy_type, self.PathStruct(
                error_path=os.path.join(new_cosmos_base_path, 'errors'),
                data_path=data_path,
                config_path=config_path,
                tfrecord_path=os.path.join(
                    new_cosmos_generated_path,
                    self.experiment_folder_name,
                    self._tfrecord_folder_name),
                gt_path=os.path.join(
                    new_cosmos_generated_path,
                    self.experiment_folder_name,
                    self._gt_folder_name),
                info_source_path=info_source_path,
                filtered_path=filtered_path,
                landmarks_path=lm_path,
                regions=regions)

    @staticmethod
    def _get_screen(frame_name):
        frame_split = frame_name.split('_')
        x_index = 0
        y_index = 1
        if not is_int(frame_split[0]):
            x_index += 1
            y_index += 1
        return float(frame_split[x_index]), float(frame_split[y_index])

    @staticmethod
    def _get_gaze_cam(gaze_screen, screen_params, cam_extrinsics):
        scrmW, scrmH, scrpW, scrpH = screen_params
        gaze_camera_spc = np.zeros(2)

        gaze_camera_spc[0] = gaze_screen[0] * scrmW / scrpW
        gaze_camera_spc[1] = gaze_screen[1] * scrmH / scrpH

        gaze_in_homogeneous = np.zeros(3)
        gaze_in_homogeneous[:2] = gaze_camera_spc
        gaze_in_homogeneous[2] = 1.0

        # Output gaze in mm
        R, T = cam_extrinsics
        return R.dot(gaze_in_homogeneous) + T.transpose()

    def extract_gaze_info(self, frame_data_dict, frame_name, region_name):
        """Extract gaze information from screen and camera parameters."""
        gaze_screen = self._get_screen(frame_name)
        frame_data_dict['label/gaze_screen_x'] = gaze_screen[0]
        frame_data_dict['label/gaze_screen_y'] = gaze_screen[1]

        gaze_cam = self._get_gaze_cam(gaze_screen, self._screen_params[region_name],
                                      self._cam_extrinsics[region_name])
        frame_data_dict['label/gaze_cam_x'] = gaze_cam[0]
        frame_data_dict['label/gaze_cam_y'] = gaze_cam[1]
        frame_data_dict['label/gaze_cam_z'] = gaze_cam[2]

    def get_pts(self, pts, frame_width, frame_height):
        """Return undistorted points if required, otherwise return distored points."""
        if not self._use_undistort:
            return pts

        reshaped_pts = np.asarray(pts, dtype=np.float64)
        reshaped_pts.shape = (-1, 1, 2)

        cam_matrix, distortion_coeffs, _ = self._cam_intrinsics
        pts_undistorted = cv2.undistortPoints(
            reshaped_pts,
            cam_matrix,
            distortion_coeffs,
            R=None,
            P=cam_matrix)

        pts_undistorted[:, :, 0] = np.clip(pts_undistorted[:, :, 0], 0, frame_width)
        pts_undistorted[:, :, 1] = np.clip(pts_undistorted[:, :, 1], 0, frame_height)

        pts_undistorted.shape = (-1, 1)

        return pts_undistorted.flatten().tolist()
