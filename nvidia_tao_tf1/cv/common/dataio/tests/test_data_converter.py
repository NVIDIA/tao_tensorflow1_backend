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

"""Test data converter."""

import os
import shutil
import tempfile
import unittest
import numpy as np
from nvidia_tao_tf1.cv.common.dataio.data_converter import DataConverter


class DataConverterTest(unittest.TestCase):
    """Test DataConverter."""

    def setUp(self):
        self.user_tfrecord_path = tempfile.mkdtemp()
        self.combined_tfrecord_path = tempfile.mkdtemp()

        # Using 3 samples each from train, test, validate of s498-gaze-0
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
            frame_name = path[-1][:-4].decode()  # get rid of .png in path

            # Type casting is necessary
            record['train/image_frame_name'] = record['train/image_frame_name'].\
                replace(orig_user_name, user_name.encode()).decode()
            record['train/eye_features'] = np.asarray(record['train/eye_features'], np.float32)
            record['train/landmarks'] = np.asarray(record['train/landmarks'], np.float32)
            record['train/landmarks_occ'] = np.asarray(record['train/landmarks_occ'], np.int64)
            record['train/norm_landmarks'] = np.asarray(record['train/norm_landmarks'], np.float32)
            record['train/landmarks_3D'] = np.asarray(record['train/landmarks_3D'], np.float32)
            self.users_dict[user_name] = {'': {frame_name: record}}
            self.json_combined.append(record)
            sample += 1
            if sample > 2:
                return

    def tearDown(self):
        shutil.rmtree(self.user_tfrecord_path)
        shutil.rmtree(self.combined_tfrecord_path)

    def test_read_tfrecords(self):
        self.assertEqual(len(self.json_combined), 3)
        self.assertSetEqual(
            set(self.json_combined[0]),
            set(DataConverter.feature_to_type))

    def test_write_user_tfrecords(self):
        DataConverter(
            self.user_tfrecord_path,
            False).write_user_tfrecords(self.users_dict)

        assert os.path.exists(os.path.join(
            self.user_tfrecord_path,
            'TYEMA8OgcTvGl6ct.tfrecords'))
        assert os.path.exists(os.path.join(
            self.user_tfrecord_path,
            'WfsYHoMi_AmWnAIL.tfrecords'))
        assert os.path.exists(os.path.join(
            self.user_tfrecord_path,
            'VSZMWvxcTBNbp_ZW.tfrecords'))

    def test_write_combined_tfrecords(self):
        DataConverter(
            self.combined_tfrecord_path,
            False).write_combined_tfrecords(
                self.users_dict,
                ['TYEMA8OgcTvGl6ct'],
                ['WfsYHoMi_AmWnAIL'],
                ['VSZMWvxcTBNbp_ZW'])

        test_path = os.path.join(self.combined_tfrecord_path, 'test.tfrecords')
        assert os.path.exists(test_path)
        validate_path = os.path.join(self.combined_tfrecord_path, 'validate.tfrecords')
        assert os.path.exists(validate_path)
        train_path = os.path.join(self.combined_tfrecord_path, 'train.tfrecords')
        assert os.path.exists(train_path)
        test_json = DataConverter.read_tfrecords(test_path, False, False)
        self.assertEqual(len(test_json), 1)
        self.assertIn(b'TYEMA8OgcTvGl6ct', test_json[0]['train/image_frame_name'])
        validate_json = DataConverter.read_tfrecords(validate_path, False, False)
        self.assertEqual(len(validate_json), 1)
        self.assertIn(b'WfsYHoMi_AmWnAIL', validate_json[0]['train/image_frame_name'])
        train_json = DataConverter.read_tfrecords(train_path, False, False)
        self.assertEqual(len(train_json), 1)
        self.assertIn(b'VSZMWvxcTBNbp_ZW', train_json[0]['train/image_frame_name'])
