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

"""Tests for FpeNet train script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mock import MagicMock
import pytest

from nvidia_tao_tf1.cv.fpenet.scripts.train import main


@pytest.mark.parametrize("results_dir", ['results_dir', None])
@pytest.mark.parametrize("is_master", [True, False])
@pytest.mark.parametrize("log_level", ['INFO', 'ERROR'])
def test_train_script_main(mocker, log_level, is_master, results_dir):
    """
    Test FpeNet train script main function.

    Args:
        mocker (Mocker obj): Mocker instance for replaying of expectations on mock objects.
        log_level (str): Log level. Options 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        is_master (bool): distribution to check current process is the master process.
        results_dir (str): Result checkpoint path.
    Returns:
        None
    """
    test_spec = expected_spec = {
        'checkpoint_dir': 'fake_dir',
    }
    scripts_module = "nvidia_tao_tf1.cv.fpenet.scripts"
    mocked_mkdir_p = mocker.patch(f'{scripts_module}.train.mkdir_p')
    mocked_trainer = MagicMock()
    mocked_deserialize = mocker.patch(
        f'{scripts_module}.train.nvidia_tao_tf1.core.coreobject.deserialize_tao_object',
        return_value=mocked_trainer
    )
    mocker.patch(
        f'{scripts_module}.train.nvidia_tao_tf1.core.distribution.'
        'distribution.Distributor.is_master',
        return_value=is_master
    )
    mocker.patch(
        f'{scripts_module}.train.yaml.load',
        return_value=test_spec
    )
    args = ['-l', log_level]
    if results_dir:
        args += ['-r', results_dir]
        expected_spec['checkpoint_dir'] = 'results_dir'
    args += ['-k', '0']
    main(args)

    mocked_deserialize.assert_called_once_with(expected_spec)
    mocked_trainer.build.assert_called_once()
    mocked_trainer.train.assert_called_once()
    if is_master:
        mocked_mkdir_p.assert_called_once()
        mocked_trainer.to_yaml.assert_called_once()
