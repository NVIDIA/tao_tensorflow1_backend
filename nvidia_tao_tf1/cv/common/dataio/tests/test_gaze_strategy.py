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

"""Test gaze strategy."""

import sys
import unittest
import mock
import numpy as np
from nvidia_tao_tf1.cv.common.dataio.gaze_strategy import GazeStrategy
from nvidia_tao_tf1.cv.common.dataio.set_label_sorter import SetLabelSorter


# Data collected from germany-1-gaze-1 config
test_R = 'nvidia_tao_tf1/cv/common/dataio/testdata/R.txt'
test_T = 'nvidia_tao_tf1/cv/common/dataio/testdata/T.txt'
test_cam_params = 'nvidia_tao_tf1/cv/common/dataio/testdata/camera_parameters.txt'
test_TV = 'nvidia_tao_tf1/cv/common/dataio/testdata/TV_size'
test_resolution = 'nvidia_tao_tf1/cv/common/dataio/testdata/resolution.txt'

real_open = open
real_loadtxt = np.loadtxt

if sys.version_info >= (3, 0):
    _BUILTIN_OPEN = 'builtins.open'
else:
    _BUILTIN_OPEN = '__builtin__.open'


def _get_test_file(fname):
    if 'R.txt' in fname:
        fname = test_R
    elif 'T.txt' in fname:
        fname = test_T
    elif 'camera_parameters' in fname:
        fname = test_cam_params
    elif 'TV_size' in fname:
        fname = test_TV
    elif 'resolution' in fname:
        fname = test_resolution
    else:
        raise IOError('Could not find file for ' + fname)
    return fname


def _open(fname, *args, **kwargs):
    return real_open(_get_test_file(fname), *args, **kwargs)


def _loadtxt(fname, *args, **kwargs):
    return real_loadtxt(_get_test_file(fname), *args, **kwargs)


def _use_cosmos639(*args, **kwargs):
    cosmos_path = args[0]
    if 'cosmos' not in cosmos_path:
        return True
    if 'cosmos10' in cosmos_path:
        return False
    if 'cosmos639' in cosmos_path:
        return True

    raise IOError('No such NFS cosmos storage for gaze strategy')


class GazeStrategyTest(unittest.TestCase):
    """Test GazeStrategy."""

    set_id = 'test-gaze-set'
    experiment_folder_suffix = 'test_suffix'
    tfrecord_folder_name = 'TfRecords'
    gt_folder_name = 'GT'

    @mock.patch(_BUILTIN_OPEN, wraps=_open)
    @mock.patch('numpy.loadtxt', wraps=_loadtxt)
    @mock.patch('os.path.exists', return_value=True)
    @mock.patch('os.path.isfile', wraps=lambda f: 'board_size' not in f)
    @mock.patch('os.makedirs')
    @mock.patch('os.listdir', return_value=[])
    def setUp(self, mock_listdir, mock_makedirs,
              mock_isfile, mock_pathexists, mock_loadtxt, mock_open):
        self.label_sorter = SetLabelSorter(
            self.experiment_folder_suffix,
            False)
        self.label_sorter.get_info_source_path = mock.MagicMock(
            return_value=[
                'json',
                'test-input-path',
                'Ground_Truth_DataFactory_' + self.experiment_folder_suffix])
        self.gaze_strategy = GazeStrategy(
            self.set_id,
            self.experiment_folder_suffix,
            self.tfrecord_folder_name,
            self.gt_folder_name,
            False,
            False,
            None,
            self.label_sorter,
            "")

    def test_get_camera_parameters(self):
        expected_distortion_coeffs = np.asarray([
            -0.38252,
            0.195521,
            0.000719038,
            0.00196389,
            -0.0346336])

        cam_intrinsics, cam_extrinsics, screen_params = self.gaze_strategy.get_camera_parameters()

        assert(len(cam_intrinsics) == 3)
        np.testing.assert_array_equal(cam_intrinsics[0], np.asarray([
            [1329.98, 0, 646.475],
            [0, 1329.48, 390.789],
            [0, 0, 1]]))
        np.testing.assert_array_equal(cam_intrinsics[1], expected_distortion_coeffs)
        np.testing.assert_array_equal(cam_intrinsics[2], expected_distortion_coeffs[:5])
        assert(len(cam_extrinsics['']) == 2)
        np.testing.assert_array_equal(cam_extrinsics[''][0], np.asarray([
            [-0.999935, 0.00027103, -0.0113785],
            [-1.12976e-05, 0.999692, 0.0248051],
            [0.0113817, 0.0248036, -0.999628]]))
        np.testing.assert_array_equal(cam_extrinsics[''][1], np.asarray([
            374.8966, -433.8045, -72.2462]))
        assert(len(screen_params['']) == 4)
        self.assertTupleEqual(screen_params[''], (1650.0, 930.0, 1920.0, 1080.0))

    def test_extract_gaze_info(self):
        frame_data_dict = {}
        frame_name = 'video1_1716_540_vc00_02.png'
        expected_frame_dict = {
            'label/gaze_screen_x': 1716.0,
            'label/gaze_screen_y': 540.0,
            'label/gaze_cam_x': -1099.5804443359375,
            'label/gaze_cam_y': 31.0604248046875,
            'label/gaze_cam_z': -44.927703857421875
        }

        self.gaze_strategy.extract_gaze_info(frame_data_dict, frame_name, '')

        self.assertEqual(expected_frame_dict.keys(), frame_data_dict.keys())
        for key, val in frame_data_dict.items():
            self.assertAlmostEqual(val, expected_frame_dict[key], 3)

    def test_get_pts_calibrated(self):
        original_pts = [35.0, 25.0]
        pts = self.gaze_strategy.get_pts(original_pts, 1280, 800)

        self.assertEqual(pts, original_pts)

    @mock.patch(_BUILTIN_OPEN, wraps=_open)
    @mock.patch('numpy.loadtxt', wraps=_loadtxt)
    @mock.patch('os.path.exists', return_value=True)
    @mock.patch('os.path.isfile', wraps=lambda f: 'board_size' not in f)
    @mock.patch('os.makedirs')
    @mock.patch('os.listdir', return_value=[])
    def test_get_pts_not_calibrated(self, mock_listdir, mock_makedirs, mock_isfile,
                                    mock_pathexists, mock_loadtxt, mock_open):
        original_pts = [450.0, 525.0]
        undistort_gaze_strategy = GazeStrategy(
            self.set_id,
            self.experiment_folder_suffix,
            self.tfrecord_folder_name,
            self.gt_folder_name,
            False,
            True,
            None,
            self.label_sorter,
            "")

        pts = undistort_gaze_strategy.get_pts(original_pts, 1280, 800)

        self.assertListEqual(pts, [447.3650540173241, 526.7086949792829])

    @mock.patch(_BUILTIN_OPEN, wraps=_open)
    @mock.patch('numpy.loadtxt', wraps=_loadtxt)
    @mock.patch('os.path.exists', return_value=True)
    @mock.patch('os.path.isfile', wraps=lambda f: 'board_size' not in f)
    @mock.patch('os.makedirs')
    @mock.patch('os.listdir', return_value=['v1'])
    @mock.patch('os.path.isdir', return_value=True)
    def test_using_lm_pred_cosmos10(self,
                                    mock_isdir,
                                    mock_listdir,
                                    mock_makedirs,
                                    mock_isfile,
                                    mock_pathexists,
                                    mock_loadtxt,
                                    mock_open):
        GazeStrategy(
            self.set_id,
            self.experiment_folder_suffix,
            self.tfrecord_folder_name,
            self.gt_folder_name,
            False,
            False,
            'fpenet_results/v2_SS80_v9',
            self.label_sorter,
            "")

        mock_makedirs.assert_called_with('/home/copilot.cosmos10/RealTimePipeline/set/' +
                                         self.set_id +
                                         '/Ground_Truth_Fpegaze_' + self.experiment_folder_suffix)

    @mock.patch(_BUILTIN_OPEN, wraps=_open)
    @mock.patch('numpy.loadtxt', wraps=_loadtxt)
    @mock.patch('os.path.exists', side_effect=_use_cosmos639)
    @mock.patch('os.path.isfile', wraps=lambda f: 'board_size' not in f)
    @mock.patch('os.makedirs')
    @mock.patch('os.listdir', return_value=['v1'])
    @mock.patch('os.path.isdir', return_value=True)
    def test_using_lm_pred_cosmos639(self,
                                     mock_isdir,
                                     mock_listdir,
                                     mock_makedirs,
                                     mock_isfile,
                                     mock_pathexists,
                                     mock_loadtxt,
                                     mock_open):
        GazeStrategy(
            self.set_id,
            self.experiment_folder_suffix,
            self.tfrecord_folder_name,
            self.gt_folder_name,
            False,
            False,
            'fpenet_results/v2_SS80_v9',
            self.label_sorter,
            "")

        mock_makedirs.assert_called_with('/home/driveix.cosmos639/GazeData/postData/' +
                                         self.set_id +
                                         '/Ground_Truth_Fpegaze_' + self.experiment_folder_suffix)
