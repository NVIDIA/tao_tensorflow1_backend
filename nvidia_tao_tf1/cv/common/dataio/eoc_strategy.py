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

"""Encapsulate implementation for eye-open-close (EOC) sets."""

import os
from nvidia_tao_tf1.cv.common.dataio.set_strategy import SetStrategy
from nvidia_tao_tf1.cv.common.dataio.utils import mkdir


class EocStrategy(SetStrategy):
    """Class encapsulates implementation specific to EOC sets."""

    eoc_parent_set_path = '/home/projects1_copilot/RealTimePipeline/set'
    eoc_error_path = '/home/projects1_copilot/RealTimePipeline/errors'

    def __init__(
        self,
        set_id,
        experiment_folder_suffix,
        tfrecord_folder_name,
        gt_folder_name,
        landmarks_folder_name,
        set_label_sorter
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
        super(EocStrategy, self).__init__(
            set_id,
            experiment_folder_suffix,
            tfrecord_folder_name,
            gt_folder_name,
            landmarks_folder_name,
            set_label_sorter)

        self._set_source_paths()
        self._set_camera_parameters()

    def _set_camera_parameters(self):
        self._cam_intrinsics, self._cam_extrinsics, self._screen_params = None, None, None

    def _get_json_path(self, set_id_path):
        return self._check_paths(set_id_path, [
            'json_datafactory_v3',
            'json_datafactory_v2',
            'json_datafactory'
        ])

    def _set_source_paths(self):
        eoc_cosmos_set_path = os.path.join(
            self.eoc_parent_set_path,
            self._set_id)

        strategy_type, info_source_path, self.experiment_folder_name = \
            self._set_label_sorter.get_info_source_path(
                self._get_json_path,
                eoc_cosmos_set_path,
                eoc_cosmos_set_path)

        lm_path = self._get_landmarks_path(eoc_cosmos_set_path)
        if lm_path is not None:
            self.experiment_folder_name = self.fpe_expr_folder + self._experiment_folder_suffix

        experiment_folder_path = os.path.join(eoc_cosmos_set_path, self.experiment_folder_name)
        mkdir(experiment_folder_path)

        self._strategy_type, self._paths = strategy_type, self.PathStruct(
            error_path=os.path.join(self.eoc_error_path),
            data_path=os.path.join(eoc_cosmos_set_path, 'Data'),
            config_path=None,
            tfrecord_path=os.path.join(
                eoc_cosmos_set_path,
                self.experiment_folder_name,
                self._tfrecord_folder_name),
            gt_path=os.path.join(
                eoc_cosmos_set_path,
                self.experiment_folder_name,
                self._gt_folder_name),
            info_source_path=info_source_path,
            filtered_path=None,
            landmarks_path=lm_path,
            regions=[''])

    def extract_gaze_info(self, frame_data_dict, frame_name, region_name):
        """No gaze information available."""
        frame_data_dict['label/gaze_screen_x'] = -1
        frame_data_dict['label/gaze_screen_y'] = -1

        frame_data_dict['label/gaze_cam_x'] = -1
        frame_data_dict['label/gaze_cam_y'] = -1
        frame_data_dict['label/gaze_cam_z'] = -1

    def get_pts(self, pts, frame_width, frame_height):
        """EOC sets have no undistort data, so it should always return original pts."""
        return pts
