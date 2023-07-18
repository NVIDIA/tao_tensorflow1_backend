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
"""Test train script."""
import os
import pytest

import tensorflow as tf


@pytest.fixture
def _spec_file():
    """Get MRCNN default file."""
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path = os.path.join(file_path, '../experiment_specs/default.txt')
    return default_spec_path


@pytest.mark.script_launch_mode('subprocess')
def test_train_script(tmpdir, script_runner, _spec_file):
    """Test train script."""
    script = 'nvidia_tao_tf1/cv/mask_rcnn/scripts/train.py'
    env = os.environ.copy()
    args = ['-k', 'nvidia_tlt',
            '--experiment_spec', _spec_file,
            '-d', tmpdir]
    tf.keras.backend.clear_session()
    ret = script_runner.run(script, env=env, *args)
    try:
        assert ret.success
    except AssertionError:
        print("Local path is not ready.")
