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
"""test spec loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import pytest
from nvidia_tao_tf1.cv.lprnet.utils.spec_loader import load_experiment_spec, spec_validator


def test_spec_loader():
    experiment_spec = load_experiment_spec(merge_from_default=True)
    assert experiment_spec.eval_config.validation_period_during_training > 0
    assert experiment_spec.training_config.num_epochs > 0
    assert experiment_spec.lpr_config.hidden_units == 512
    assert experiment_spec.lpr_config.max_label_length == 8
    assert experiment_spec.augmentation_config.max_rotate_degree == 5


def catch_assert_error(spec):
    with pytest.raises(AssertionError):
        spec_validator(spec)


def test_spec_validator():
    experiment_spec = load_experiment_spec(merge_from_default=True)
    # lpr_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.lpr_config.hidden_units = 0
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.lpr_config.arch = "baselin"
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.lpr_config.arch = ""
    catch_assert_error(test_spec)
    # train_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.training_config.num_epochs = 0
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.training_config.learning_rate.soft_start_annealing_schedule.soft_start = 0
    catch_assert_error(test_spec)
    test_spec.training_config.early_stopping.monitor = "losss"
    catch_assert_error(test_spec)
    # eval_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.eval_config.batch_size = 0
    catch_assert_error(test_spec)
    # aug_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.augmentation_config.output_channel = 4
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.augmentation_config.gaussian_kernel_size[:] = [0]
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.augmentation_config.blur_prob = 1.1
    catch_assert_error(test_spec)
    # dataset_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.dataset_config.data_sources[0].label_directory_path = ""
    catch_assert_error(test_spec)
