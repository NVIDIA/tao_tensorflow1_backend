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

"""Test gt converter."""

import os
import shutil
import tempfile
import unittest
import numpy as np
from nvidia_tao_tf1.cv.common.dataio.data_converter import DataConverter
from nvidia_tao_tf1.cv.common.dataio.gt_converter import GtConverter


class GtConverterTest(unittest.TestCase):
    """Test GtConverter."""

    def setUp(self):
        self.user_gt_path = tempfile.mkdtemp()
        self.combined_gt_path = tempfile.mkdtemp()

        self.json_combined_all = DataConverter.read_tfrecords(
            'nvidia_tao_tf1/cv/common/dataio/testdata/combined.tfrecords',
            False,
            False)

        users = ['TYEMA8OgcTvGl6ct', 'WfsYHoMi_AmWnAIL', 'VSZMWvxcTBNbp_ZW']
        self.users_dict = {}
        self.json_combined = []
        sample = 0
        for record in self.json_combined_all:
            path = record['train/image_frame_name'].split(b'/')
            orig_user_name = path[-2]
            user_name = users[sample]
            print(type(orig_user_name), type(user_name))
            print(type(record['train/image_frame_name']))
            frame_name = path[-1][:-4].decode()  # get rid of .png in path

            # Type casting is necessary
            record['train/image_frame_name'] = record['train/image_frame_name'].\
                replace(orig_user_name, user_name.encode()).decode()
            record['train/eye_features'] = np.asarray(record['train/eye_features'], np.float32)
            record['train/landmarks'] = np.asarray(record['train/landmarks'], np.float32)
            record['train/landmarks_occ'] = np.asarray(record['train/landmarks_occ'], np.int64)
            record['train/norm_landmarks'] = np.asarray(record['train/norm_landmarks'], np.float32)
            record['train/landmarks_3D'] = np.asarray(record['train/landmarks_3D'], np.float32)
            record['train/norm_face_cnv_mat'] = -1.0 * np.ones((9,), dtype=np.float32)
            record['train/norm_leye_cnv_mat'] = -1.0 * np.ones((9,), dtype=np.float32)
            record['train/norm_reye_cnv_mat'] = -1.0 * np.ones((9,), dtype=np.float32)
            record['label/face_cam_x'] = record['label/mid_cam_x']
            record['label/face_cam_y'] = record['label/mid_cam_y']
            record['label/face_cam_z'] = record['label/mid_cam_z']
            self.users_dict[user_name] = {'': {frame_name: record}}
            self.json_combined.append(record)
            sample += 1
            if sample > 2:
                return

    def tearDown(self):
        shutil.rmtree(self.user_gt_path)
        shutil.rmtree(self.combined_gt_path)

    def test_write_landmarks(self):
        GtConverter(
            self.user_gt_path,
            False).write_landmarks(self.users_dict)

        assert os.path.exists(os.path.join(
            self.user_gt_path,
            'TYEMA8OgcTvGl6ct_landmarks.txt'))
        assert os.path.exists(os.path.join(
            self.user_gt_path,
            'WfsYHoMi_AmWnAIL_landmarks.txt'))
        assert os.path.exists(os.path.join(
            self.user_gt_path,
            'VSZMWvxcTBNbp_ZW_landmarks.txt'))

    def test_write_gt_files(self):
        GtConverter(
            self.user_gt_path,
            False).write_gt_files(self.users_dict)

        assert os.path.exists(os.path.join(
            self.user_gt_path,
            'TYEMA8OgcTvGl6ct.txt'))
        assert os.path.exists(os.path.join(
            self.user_gt_path,
            'WfsYHoMi_AmWnAIL.txt'))
        assert os.path.exists(os.path.join(
            self.user_gt_path,
            'VSZMWvxcTBNbp_ZW.txt'))

    def test_write_combined_landmarks(self):
        GtConverter(
            self.combined_gt_path,
            False).write_combined_landmarks(
                self.users_dict,
                ['TYEMA8OgcTvGl6ct'],
                ['WfsYHoMi_AmWnAIL'],
                ['VSZMWvxcTBNbp_ZW'])

        assert os.path.exists(os.path.join(
            self.combined_gt_path,
            'test_landmarks.txt'))
        assert os.path.exists(os.path.join(
            self.combined_gt_path,
            'validate_landmarks.txt'))
        assert os.path.exists(os.path.join(
            self.combined_gt_path,
            'train_landmarks.txt'))
        test_path = os.path.join(self.combined_gt_path, 'test_landmarks.txt')
        assert os.path.exists(test_path)
        validate_path = os.path.join(self.combined_gt_path, 'validate_landmarks.txt')
        assert os.path.exists(validate_path)
        train_path = os.path.join(self.combined_gt_path, 'train_landmarks.txt')
        assert os.path.exists(train_path)
        test_gt = open(test_path).readlines()
        self.assertEqual(len(test_gt), 1)
        self.assertIn('TYEMA8OgcTvGl6ct', test_gt[0])
        validate_gt = open(validate_path).readlines()
        self.assertEqual(len(validate_gt), 1)
        self.assertIn('WfsYHoMi_AmWnAIL', validate_gt[0])
        train_gt = open(train_path).readlines()
        self.assertEqual(len(train_gt), 1)
        self.assertIn('VSZMWvxcTBNbp_ZW', train_gt[0])

    def test_write_combined_gt_files(self):
        GtConverter(
            self.combined_gt_path,
            False).write_combined_gt_files(
                self.users_dict,
                ['TYEMA8OgcTvGl6ct'],
                ['WfsYHoMi_AmWnAIL'],
                ['VSZMWvxcTBNbp_ZW'])

        assert os.path.exists(os.path.join(
            self.combined_gt_path,
            'test.txt'))
        assert os.path.exists(os.path.join(
            self.combined_gt_path,
            'validate.txt'))
        assert os.path.exists(os.path.join(
            self.combined_gt_path,
            'train.txt'))
        test_path = os.path.join(self.combined_gt_path, 'test.txt')
        assert os.path.exists(test_path)
        validate_path = os.path.join(self.combined_gt_path, 'validate.txt')
        assert os.path.exists(validate_path)
        train_path = os.path.join(self.combined_gt_path, 'train.txt')
        assert os.path.exists(train_path)
        test_gt = open(test_path).readlines()
        self.assertEqual(len(test_gt), 1)
        self.assertIn('TYEMA8OgcTvGl6ct', test_gt[0])
        validate_gt = open(validate_path).readlines()
        self.assertEqual(len(validate_gt), 1)
        self.assertIn('WfsYHoMi_AmWnAIL', validate_gt[0])
        train_gt = open(train_path).readlines()
        self.assertEqual(len(train_gt), 1)
        self.assertIn('VSZMWvxcTBNbp_ZW', train_gt[0])
