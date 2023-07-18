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
'''Unit test for spec loader.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import pytest
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader import spec_loader
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader.spec_wrapper import ExperimentSpec


@pytest.fixture(scope='function')
def _spec_file():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    frcnn_root_dir = os.path.dirname(parent_dir)
    return os.path.join(frcnn_root_dir, 'experiment_spec/default_spec_ci.txt')


@pytest.fixture(scope='function')
def _out_spec_file():
    os_handle, out_file_name = tempfile.mkstemp()
    os.close(os_handle)
    return out_file_name


def test_spec_loader(_spec_file, _out_spec_file):
    spec = spec_loader.load_experiment_spec(_spec_file)
    spec_loader.write_spec_to_disk(spec, _out_spec_file)


def test_spec_wrapper(_spec_file):
    spec = spec_loader.load_experiment_spec(_spec_file)
    spec_obj = ExperimentSpec(spec)
    assert spec_obj, (
        "Invalid spec file: {}".format(_spec_file)
    )
