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
"""Test FpeNet DataIO dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
from yaml import load
from nvidia_tao_tf1.cv.fpenet.dataio.generate_dataset import tfrecord_manager

test_config_path = 'nvidia_tao_tf1/cv/fpenet/dataio/testdata/test_config.yaml'


class GenerateDatasetTest(unittest.TestCase):
    def test_generate_dataset(self):
        '''Test if a sample dataset is generated.'''
        with open(test_config_path, 'r') as f:
            args = load(f)

        tfrecord_manager(args)

        # tests on generated files
        tfRecordPath = os.path.join(args['save_root_path'],
                                    args['save_path'],
                                    'testdata',
                                    args['tfrecord_folder'])

        recordfile = os.path.join(tfRecordPath, args['tfrecord_name'])
        assert os.path.exists(recordfile)
