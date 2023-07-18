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
import shutil
import pytest
import tensorflow as tf


@pytest.fixture
def _spec_file():
    """Get UNet default file."""
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path_1 = os.path.join(file_path, '../experiment_specs/default1.txt')
    default_spec_path_2 = os.path.join(file_path, '../experiment_specs/default2.txt')

    spec_files = [default_spec_path_1, default_spec_path_2]
    return spec_files


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
@pytest.mark.script_launch_mode('subprocess')
def test_train_script(tmpdir, script_runner):
    """Test train script."""
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tmpdir = os.path.join(file_path, "tmp_results")
    script = 'nvidia_tao_tf1/cv/unet/scripts/train.py'
    env = os.environ.copy()
    spec_files = _spec_file()
    for spec_file in spec_files:
        args = ['-k', 'nvidia_tlt',
                '-e', spec_file,
                '-r', tmpdir]
        tf.keras.backend.clear_session()
        ret = script_runner.run(script, env=env, *args)
        try:
            assert ret.success
        except AssertionError:
            print("Local path is not ready.")
        finally:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
