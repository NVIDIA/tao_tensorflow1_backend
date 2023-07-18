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
'''Unit test for FasterRCNN model exporter interfaces.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pytest

from nvidia_tao_tf1.cv.faster_rcnn.export.exporter import FrcnnExporter


@pytest.fixture()
def _spec_file():
    '''default spec file.'''
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(parent_dir, 'experiment_spec/default_spec_ci.txt')


def test_export_args(_spec_file):
    '''test to make sure the exporter raises proper errors when some args are missing.'''
    # we are just testing the interfaces, so using spec file as model file
    # should be OK
    with pytest.raises(AssertionError):
        FrcnnExporter(experiment_spec_path=None,
                      model_path=_spec_file,
                      key='tlt')
    with pytest.raises(AssertionError):
        FrcnnExporter(experiment_spec_path='',
                      model_path=_spec_file,
                      key='tlt')
    with pytest.raises(AssertionError):
        FrcnnExporter(experiment_spec_path=_spec_file,
                      model_path='',
                      key='tlt')
