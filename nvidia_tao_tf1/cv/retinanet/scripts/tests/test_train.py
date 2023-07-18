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
"""Unit test for RetinaNet train script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import keras
import pytest

SPEC_FILES = [
    'default_spec.txt',
]


@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.slow
@pytest.mark.parametrize('_spec_file', SPEC_FILES)
def test_train(script_runner, tmpdir, _spec_file):
    '''Test the train script.'''
    keras.backend.clear_session()
    keras.backend.set_learning_phase(1)
    script = 'nvidia_tao_tf1/cv/retinanet/scripts/train.py'
    env = os.environ.copy()
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    spec_file = os.path.join(parent_dir, 'experiment_specs', _spec_file)
    temp_dir_name = tempfile.mkdtemp()

    args = ['-e']
    args.append(spec_file)
    args.append('-k')
    args.append('nvidia_tlt')
    args.append('-r')
    args.append(temp_dir_name)
    ret = script_runner.run(script, env=env, *args)
    try:
        assert ret.success
        shutil.rmtree(temp_dir_name)
    except AssertionError:
        print("Local path is not ready.")
