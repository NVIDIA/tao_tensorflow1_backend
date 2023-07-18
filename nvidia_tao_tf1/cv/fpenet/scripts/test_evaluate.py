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

"""Tests for FpeNet evaluate_model script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from mock import MagicMock
import pytest

from nvidia_tao_tf1.cv.fpenet.scripts.evaluate import main

EVALUATOR_SCRIPT = "nvidia_tao_tf1.cv.fpenet.scripts.evaluate"


@pytest.mark.parametrize("exp_spec", ['default.yaml', None])
@pytest.mark.parametrize("eval_type", ['kpi_testing'])
@pytest.mark.parametrize("log_level", ['INFO', 'ERROR'])
def test_evaluate_script_main(mocker, tmpdir, log_level, eval_type, exp_spec):
    """Test GazeNet evaluate script main function."""
    test_spec = expected_spec = {
        'config': {
            'checkpoint_dir': 'original_path',
        }
    }
    mocked_trainer = MagicMock()
    mocked_deserialize = mocker.patch(
        f'{EVALUATOR_SCRIPT}.nvidia_tao_tf1.core.coreobject.deserialize_tao_object',
        return_value=mocked_trainer
    )
    mocker.patch(
        f'{EVALUATOR_SCRIPT}.yaml.load',
        return_value=test_spec
    )
    mocked_config_path = mocker.patch(
        f'{EVALUATOR_SCRIPT}.os.path.isfile',
        return_value=True
    )
    # Patching the os.path.exists call
    mocker.patch(
        f'{EVALUATOR_SCRIPT}.os.path.exists',
        return_value=True
    )
    mocker.patch(
        f'{EVALUATOR_SCRIPT}.open',
        return_value=open('nvidia_tao_tf1/cv/fpenet/experiment_specs/default.yaml', 'r')
    )
    model_path = os.path.join(str(tmpdir), 'fake_path')
    results_dir = os.path.join(str(tmpdir), 'results')
    if exp_spec is not None:
        yaml_path = os.path.join(model_path, exp_spec)
        args = ['-l', log_level, '-m', model_path, '-type', eval_type, '-e', exp_spec, '-k', '0',
                '-r', results_dir]
    else:
        # Default case looks for 'experiment_spec.yaml'.
        yaml_path = os.path.join(model_path, 'experiment_spec.yaml')
        args = ['-l', log_level, '-m', model_path, '-type', eval_type, '-k', '0',
                '-r', results_dir]
    main(args)

    assert mocked_config_path.call_count == 2

    mocked_deserialize.assert_called_once_with(expected_spec)
    mocked_trainer.build.assert_called_once_with(
        eval_mode=eval_type,
        eval_model_path=model_path
    )
    mocked_trainer.run_testing.assert_called_once()
