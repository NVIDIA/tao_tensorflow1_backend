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

"""Test common functions."""

import datetime
import os
import shutil
import tempfile
import unittest
from nvidia_tao_tf1.cv.common.dataio.utils import PipelineReporter


class PipelineReporterTest(unittest.TestCase):
    """Test PipelineReporter."""

    script_name = 'test'
    set_id = 'test-set'

    def setUp(self):
        self.error_path = tempfile.mkdtemp()
        self.reporter = PipelineReporter(
            self.error_path,
            self.script_name,
            self.set_id)
        self.reporter.add_error('Error')
        self.reporter.add_info('Info')
        self.reporter.add_warning('Warning')

    def tearDown(self):
        shutil.rmtree(self.error_path)

    def test_write_to_log(self):
        run_folder_name = self.script_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d')
        log_path_today = os.path.join(self.error_path, run_folder_name)
        error_path = os.path.join(log_path_today, 'Error')
        warning_path = os.path.join(log_path_today, 'Warning')
        err_lines = ''
        warning_lines = ''

        self.reporter.write_to_log()

        err_lines = open(os.path.join(error_path, self.set_id + '.log'), 'r').read()
        warning_lines = open(os.path.join(warning_path, self.set_id + '.log'), 'r').read()

        self.assertIn('Error\n', err_lines)
        self.assertIn('Info\n', err_lines)
        self.assertNotIn('Warning\n', err_lines)
        self.assertIn('Warning\n', warning_lines)
        self.assertNotIn('Error\n', warning_lines)
        self.assertNotIn('Info\n', warning_lines)
