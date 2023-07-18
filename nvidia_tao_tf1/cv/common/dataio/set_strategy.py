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

"""Allow client to use common methodss regardless of set type."""

import abc
from collections import namedtuple
import os


class SetStrategy(object):
    """Abstract class for common methods between types of sets."""

    __metaclass__ = abc.ABCMeta

    PathStruct = namedtuple('PathStruct',
                            '''error_path,
                            data_path,
                            config_path,
                            tfrecord_path,
                            gt_path,
                            info_source_path,
                            filtered_path,
                            landmarks_path,
                            regions''')

    fpe_expr_folder = 'Ground_Truth_Fpegaze_'

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
        self._set_id = set_id
        self._experiment_folder_suffix = experiment_folder_suffix
        self._tfrecord_folder_name = tfrecord_folder_name
        self._gt_folder_name = gt_folder_name
        self._landmarks_folder_name = landmarks_folder_name
        self._set_label_sorter = set_label_sorter

        self.experiment_folder_name = None

    @staticmethod
    def _check_paths(parent_path, possible_paths):
        for path in possible_paths:
            full_path = os.path.join(parent_path, path)
            if os.path.exists(full_path):
                return full_path

        return None

    def _get_landmarks_path(self, parent):
        """Get the path to obtain landmarks from, using self._landmarks_folder_name.

        Args:
            parent (path-like): Path where _landmarks_folder_name should be living.

        Returns:
            lm_path (path-like): Path to the landmark folder.
        """
        if self._landmarks_folder_name is None:
            return None

        lm_path = os.path.join(parent, self._landmarks_folder_name, 'facelandmark')
        # The check to see if path is valid happens here, not in main.
        if not os.path.isdir(lm_path):
            raise IOError("Could not find landmarks-folder: {} is not a valid directory"
                          .format(lm_path))
        return lm_path

    @abc.abstractmethod
    def _set_source_paths(self):
        pass

    def get_source_paths(self):
        """Get needed input file paths."""
        return self._strategy_type, self._paths

    @abc.abstractmethod
    def _set_camera_parameters(self, config_path):
        pass

    def get_camera_parameters(self):
        """Get camera parameters."""
        return self._cam_intrinsics, self._cam_extrinsics, self._screen_params

    @abc.abstractmethod
    def _get_json_path(self, set_id_path):
        pass

    @abc.abstractmethod
    def extract_gaze_info(self, frame_data_dict, frame_name):
        """Return strategy's extracted info about gaze."""
        pass

    @abc.abstractmethod
    def get_pts(self, pts, frame_width, frame_height):
        """Return strategy's landmark points (can be undistorted / distorted)."""
        pass
