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

"""Test set label sorter."""

import os
import shutil
import tempfile
import unittest
from nvidia_tao_tf1.cv.common.dataio.set_label_sorter import SetLabelSorter


class SetLabelSorterTest(unittest.TestCase):
    """Test SetLabelSorter."""

    experiment_folder_suffix = 'test'

    def setUp(self):
        self.json_folder = tempfile.mkdtemp()

        self.nvhelnet_parent_folder = tempfile.mkdtemp()
        self.nvhelnet_folder = os.path.join(self.nvhelnet_parent_folder, 'Nvhelnet_v11.2')
        self.nvhelnet_forced = os.path.join(self.nvhelnet_parent_folder, 'Nvhelnet_forced')
        os.makedirs(self.nvhelnet_folder)
        os.makedirs(self.nvhelnet_forced)

    def tearDown(self):
        shutil.rmtree(self.json_folder)
        shutil.rmtree(self.nvhelnet_parent_folder)

    def test_get_info_source_path_json(self):
        strategy_type, info_source_path, experiment_folder_name = \
            SetLabelSorter(
                self.experiment_folder_suffix,
                None).get_info_source_path(
                    lambda _: self.json_folder,
                    None,
                    self.nvhelnet_parent_folder)

        self.assertEqual(strategy_type, 'json')
        self.assertEqual(info_source_path, self.json_folder)
        self.assertEqual(
            experiment_folder_name,
            'Ground_Truth_DataFactory_' + self.experiment_folder_suffix)

    def test_get_info_source_path_sdk(self):
        strategy_type, info_source_path, experiment_folder_name = \
            SetLabelSorter(
                self.experiment_folder_suffix,
                None).get_info_source_path(
                    lambda _: None,
                    None,
                    self.nvhelnet_parent_folder)

        self.assertEqual(strategy_type, 'sdk')
        self.assertEqual(info_source_path, self.nvhelnet_folder)
        self.assertEqual(
            experiment_folder_name,
            'Ground_Truth_Nvhelnet_' + self.experiment_folder_suffix)

    def test_get_info_source_path_force_sdk(self):
        strategy_type, info_source_path, experiment_folder_name = \
            SetLabelSorter(
                self.experiment_folder_suffix,
                'Nvhelnet_forced').get_info_source_path(
                    lambda _: self.json_folder,
                    None,
                    self.nvhelnet_parent_folder)

        self.assertEqual(strategy_type, 'sdk')
        self.assertEqual(info_source_path, self.nvhelnet_forced)
        self.assertEqual(
            experiment_folder_name,
            'Ground_Truth_Nvhelnet_' + self.experiment_folder_suffix)

    def test_get_info_source_path_sdk_nonexisting(self):
        with self.assertRaises(IOError):
            SetLabelSorter(
                self.experiment_folder_suffix,
                'Nvhelnet_nonexisting').get_info_source_path(
                    lambda _: self.json_folder,
                    None,
                    self.nvhelnet_parent_folder)
