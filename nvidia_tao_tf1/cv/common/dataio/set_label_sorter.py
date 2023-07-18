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

"""Sort set type based on input labels and generate either DataFactory or Nvhelnet folder."""

import os


class SetLabelSorter(object):
    """Class to sort experiment folder created for set."""

    json_expr_folder_prefix = 'Ground_Truth_DataFactory_'
    sdk_expr_folder_prefix = 'Ground_Truth_Nvhelnet_'

    def __init__(
        self,
        experiment_folder_suffix,
        sdklabels_folder_name,
    ):
        """Initialize parameters."""
        self._experiment_folder_suffix = experiment_folder_suffix
        self._sdklabels_folder_name = sdklabels_folder_name

    def get_info_source_path(
        self,
        get_json_path,
        parent_json_path,
        parent_sdklabels_path,
    ):
        """Get required input file paths for tfrecord generation.

        Args:
            get_json_path (function): Function which returns json files path.
            parent_json_path (path): Parent path of json folder.
            parent_sdklabels_path (path): Parent path of nvhelnet sdk labels folder.

        Returns:
            strategy_type (str): Type of strategy (sdk / json) determined by input files.
            info_source_path (path): Label source path.
            experiment_folder_name (str): Name of experiment folder which contains tfrecords.
        """
        def _get_sdklabels_folder(parent):
            sdklabels_path = os.path.join(parent, self._sdklabels_folder_name)
            # The check to see if path is valid happens here, not in main.
            if not os.path.isdir(sdklabels_path):
                if not os.path.isdir(sdklabels_path):
                    print("Could not find sdklabels-folder: {} is not a valid directory"
                          .format(sdklabels_path))
                sdklabels_path = sdklabels_path.replace('postData', 'orgData')
                if not os.path.isdir(sdklabels_path):
                    raise IOError("Could not find sdklabels-folder: {} is not a valid directory"
                                  .format(sdklabels_path))
            return sdklabels_path

        info_source_path = None

        if self._sdklabels_folder_name is None:
            # Consider json files first and only if sdklabels is not given.
            # If not found, fall back to a known good nvhelnet sdk folder.
            strategy_type = 'json'
            info_source_path = get_json_path(parent_json_path)
            experiment_folder_name = self.json_expr_folder_prefix + self._experiment_folder_suffix

        if info_source_path is None:
            if self._sdklabels_folder_name is None:
                self._sdklabels_folder_name = 'Nvhelnet_v11.2'
            strategy_type = 'sdk'
            info_source_path = _get_sdklabels_folder(parent_sdklabels_path)
            experiment_folder_name = self.sdk_expr_folder_prefix + self._experiment_folder_suffix

        return strategy_type, info_source_path, experiment_folder_name
