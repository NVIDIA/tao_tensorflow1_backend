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

"""Abstracted methods of set strategy (gaze / eoc) and select strategy for set."""

from nvidia_tao_tf1.cv.common.dataio.eoc_strategy import EocStrategy
from nvidia_tao_tf1.cv.common.dataio.gaze_strategy import GazeStrategy
from nvidia_tao_tf1.cv.common.dataio.set_label_sorter import SetLabelSorter


class SetStrategyGenerator(object):
    """Class for generating set strategy and retrieving strategy output."""

    def __init__(
        self,
        set_id,
        experiment_folder_suffix,
        tfrecord_folder_name,
        gt_folder_name,
        use_filtered,
        use_undistort,
        landmarks_folder_name,
        sdklabels_folder_name,
        set_root_path
    ):
        """Sort set into sdk or json and generate gaze / eoc strategy."""
        self._landmarks_folder_name = landmarks_folder_name
        self._use_undistort = use_undistort
        set_label_sorter = SetLabelSorter(
            experiment_folder_suffix,
            sdklabels_folder_name)

        if 'eoc' in set_id.lower():
            self._strategy = EocStrategy(
                set_id,
                experiment_folder_suffix,
                tfrecord_folder_name,
                gt_folder_name,
                landmarks_folder_name,
                set_label_sorter)
        else:
            self._strategy = GazeStrategy(
                set_id,
                experiment_folder_suffix,
                tfrecord_folder_name,
                gt_folder_name,
                use_filtered,
                use_undistort,
                landmarks_folder_name,
                set_label_sorter,
                set_root_path)

    def get_camera_parameters(self):
        """Return strategy's camera parameters."""
        return self._strategy.get_camera_parameters()

    def get_source_paths(self):
        """Return strategy's needed input file paths."""
        return self._strategy.get_source_paths()

    def extract_gaze_info(self, frame_data_dict, frame_name, region_name):
        """Return strategy's extracted info about gaze."""
        self._strategy.extract_gaze_info(frame_data_dict, frame_name, region_name)

    def get_pts(self, pts, frame_width, frame_height):
        """Return strategy's landmark points (can be undistorted / distorted)."""
        return self._strategy.get_pts(pts, frame_width, frame_height)

    def use_undistort(self):
        """Return whether user wants to use undistorted frames and landmarks."""
        return self._use_undistort
