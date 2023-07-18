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
"""Test Dataset converter script."""

import os
import shutil
import tempfile
import pytest


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
@pytest.mark.script_launch_mode('subprocess')
def test_dataset_convert(script_runner):
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    coco_json_file = os.path.join(file_path, 'tests/test_data/instances_val2017.json')
    num_images = 5
    results_dir = tempfile.mkdtemp()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    env = os.environ.copy()
    script = "nvidia_tao_tf1/cv/unet/scripts/dataset_convert.py"
    args = ['-f', coco_json_file,
            '-r', results_dir,
            '-n', str(num_images)]

    ret = script_runner.run(script, env=env, *args)
    # before abort, remove the created temp files when exception raises
    try:
        assert ret.success, "The dataset convert failed."
        assert (len([f for f in os.listdir(results_dir) if f.endswith(".png")]) == num_images), \
            "All the images were not converted to VOC."
    except AssertionError:
        raise AssertionError(ret.stdout + ret.stderr)
    finally:
        shutil.rmtree(results_dir)
