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

"""Manage generation of normalized data and prepare data for gazenet inference."""


from nvidia_tao_tf1.cv.common.dataio.custom_data_converter import CustomDataConverter
from nvidia_tao_tf1.cv.common.dataio.custom_jsonlabels_strategy import CustomJsonLabelsStrategy
from nvidia_tao_tf1.cv.common.dataio.gaze_custom_set_strategy import GazeCustomSetStrategy
from nvidia_tao_tf1.cv.common.dataio.utils import PipelineReporter


class CustomerDataManager(object):
    """Manage the flow of data source files to tfrecord generation."""

    def __init__(self,
                 data_root_path,
                 norm_data_folder_name='norm_data',
                 data_strategy_type='json',
                 landmarks_folder_name=None,
                 sdklabels_folder_name=None,
                 save_images=True):
        """Initialize parameters.

        Args:
            data_root_path (str): root path of data
            norm_data_folder_name (str): Normalized data folder name.
            data_strategy_type (str): specify the data labeling resources.
            landmarks_folder_name (str): Folder name to obtain predicted fpe landmarks from.
            sdklabels_folder_name (str): Folder name to obtain predicted nvhelnet sdk labels from.
            save_images (bool): Flag to determine if want to save normalized faces and images.
        """
        self._data_root_path = data_root_path
        self._data_strategy_type = data_strategy_type
        self._landmarks_folder_name = landmarks_folder_name
        self._sdklabels_folder_name = sdklabels_folder_name
        self._norm_data_folder_name = norm_data_folder_name
        self._save_images = save_images
        self._frame_dict = None

        self._set_strategy = GazeCustomSetStrategy(
            data_root_path=self._data_root_path,
            norm_data_folder_name=self._norm_data_folder_name,
            data_strategy_type=self._data_strategy_type,
            landmarks_folder_name=self._landmarks_folder_name)
        self._strategy_type, self._paths = self._set_strategy.get_source_paths()

        self._logger = PipelineReporter(log_path=self._paths.error_path,
                                        script_name='custom_data_manager',
                                        set_id="")

        strategy = None
        if self._strategy_type == 'json':
            strategy = CustomJsonLabelsStrategy(
                data_root_path=self._data_root_path,
                norm_data_folder_name=self._norm_data_folder_name,
                set_strategy=self._set_strategy,
                save_images=self._save_images,
                logger=self._logger
            )
            self._frame_dict = strategy.get_data()
        else:
            self._logger.add_error('No such strategy {}'.format(self._strategy_type))
            raise ValueError('Invalid strategy.')

        converter = CustomDataConverter(use_lm_pred=self._landmarks_folder_name is not None)
        self._tfrecords_data = converter.generate_frame_tfrecords(self._frame_dict)

    def get_data(self):
        """get frame dictionary data."""

        return self._frame_dict

    def get_tfrecords_data(self):
        """get tfrecords data."""

        return self._tfrecords_data

    def get_data_number(self):
        """get data number."""

        return len(self._frame_dict)

    def get_frame_dict(self):
        """get frame dictionary."""

        return self._frame_dict
