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

"""Test eoc strategy."""

import unittest
import mock
from nvidia_tao_tf1.cv.common.dataio.eoc_strategy import EocStrategy
from nvidia_tao_tf1.cv.common.dataio.set_label_sorter import SetLabelSorter


class EocStrategyTest(unittest.TestCase):
    """Test EocStrategy."""

    set_id = 'test-eoc-set'
    experiment_folder_suffix = 'test_suffix'
    tfrecord_folder_name = 'TfRecords'
    gt_folder_name = 'GT'

    @mock.patch('os.makedirs')
    def setUp(self, mock_makedirs):
        self.label_sorter = SetLabelSorter(
            self.experiment_folder_suffix,
            False)
        self.label_sorter.get_info_source_path = mock.MagicMock(
            return_value=[
                'json',
                'test-input-path',
                'Ground_Truth_DataFactory_' + self.experiment_folder_suffix])
        self.eoc_strategy = EocStrategy(
            self.set_id,
            self.experiment_folder_suffix,
            self.tfrecord_folder_name,
            self.gt_folder_name,
            None,
            self.label_sorter)

        mock_makedirs.assert_called_with(
            '/home/projects1_copilot/RealTimePipeline/set/' + self.set_id +
            '/Ground_Truth_DataFactory_' + self.experiment_folder_suffix)

    def test_get_camera_parameters(self):
        cam_params = self.eoc_strategy.get_camera_parameters()

        self.assertEqual(
            (None, None, None),
            cam_params)

    def test_extract_gaze_info(self):
        frame_data_dict = {}
        frame_name = ''
        expected_frame_dict = {
            'label/gaze_screen_x': -1,
            'label/gaze_screen_y': -1,
            'label/gaze_cam_x': -1,
            'label/gaze_cam_y': -1,
            'label/gaze_cam_z': -1,
        }

        self.eoc_strategy.extract_gaze_info(frame_data_dict, frame_name, '')

        self.assertDictEqual(
            expected_frame_dict,
            frame_data_dict)

    @mock.patch('os.makedirs')
    @mock.patch('os.listdir', return_value=['v1'])
    @mock.patch('os.path.isdir', return_value=True)
    def test_using_lm_pred(self, mock_isdir, mock_listdir, mock_makedirs):
        EocStrategy(
            self.set_id,
            self.experiment_folder_suffix,
            self.tfrecord_folder_name,
            self.gt_folder_name,
            'fpenet_results/v2_SS80_v9',
            self.label_sorter)

        mock_makedirs.assert_called_with('/home/projects1_copilot/RealTimePipeline/set/' +
                                         self.set_id +
                                         '/Ground_Truth_Fpegaze_' + self.experiment_folder_suffix)
