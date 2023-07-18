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

"""Test for tensorflow add-on features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import tempfile

import pytest


def _execute_shell_cmd(cmd, debug=False, exit_on_return_code=True):
    """Execute the given command in a shell.

    Args:
        cmd(str): Complete command to execute
        debug(boolean): Debug mode; will log the commands and its output to the console.
        exit_on_return_code (bool): Will call exit with the return code of the subprocess.

    Returns:
        stdout(str): Command output captured from the console/stderr stream in case of error.
    """
    if debug:
        print("Executing cmd: '{}'".format(cmd))
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    proc_out = process.communicate()
    proc_stdout = proc_out[0].strip()
    proc_stderr = proc_out[1].strip()
    if debug:
        print(proc_stdout)
    # Always print error message if any.
    if proc_stderr:
        print(proc_stderr)
    if exit_on_return_code and process.returncode:
        exit(process.returncode)

    if proc_stdout:
        return proc_stdout.decode("utf-8")

    return proc_stderr.decode("utf-8")


_params = [
    # No need to test for differences in models generated with non-deterministic
    # training for `train`; it might generate models that may differ, or may
    # match on different runs, hence we can't really test for that.
    ("nvidia_tao_tf1/core/training/train.py", False, True),
    # ("moduluspy/modulus/training/train_eager", True, False),
]


@pytest.mark.parametrize("app, test_for_non_deterministic_diff, diff", _params)
def test_determinism(app, test_for_non_deterministic_diff, diff):
    """Test determinism of generated models."""

    temp_dir = tempfile.mkdtemp()
    # Generate two models with determinism ON and one with determinism OFF.
    model1_file = os.path.join(temp_dir, "model_1.hdf5")
    model2_file = os.path.join(temp_dir, "model_2.hdf5")
    model3_file = os.path.join(temp_dir, "model_3.hdf5")

    _execute_shell_cmd("{} -o {} --deterministic --seed=42".format(app, model1_file))
    _execute_shell_cmd("{} -o {} --deterministic --seed=42".format(app, model2_file))
    if test_for_non_deterministic_diff:
        _execute_shell_cmd("{} -o {} --seed=42".format(app, model3_file))

    # Model 1 and 2 should be same.
    assert "" == _execute_shell_cmd("h5diff {} {}".format(model1_file, model2_file))

    # Model 1 and 3 trained with determinism flag on/off respectively.
    # They might differ depending on eager-ness of TF and other conditions.
    # In case they differ, we expect the h5diff to return err code.
    if test_for_non_deterministic_diff:
        if diff:
            assert "Differences found" in _execute_shell_cmd(
                "h5diff {} {}".format(model1_file, model3_file),
                exit_on_return_code=False,
            )
        else:
            assert "" == _execute_shell_cmd(
                "h5diff {} {}".format(model1_file, model3_file),
                exit_on_return_code=False,
            )
