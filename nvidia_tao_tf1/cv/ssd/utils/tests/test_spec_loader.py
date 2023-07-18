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
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import load_experiment_spec, spec_validator,\
    validate_eval_spec, validate_train_spec


def test_spec_loader():
    experiment_spec, is_dssd = load_experiment_spec()
    assert is_dssd
    assert len(experiment_spec.ssd_config.arch) > 3
    assert experiment_spec.eval_config.validation_period_during_training > 0
    assert experiment_spec.training_config.num_epochs > 0
    assert experiment_spec.nms_config.top_k > 0
    with pytest.raises(AssertionError):
        experiment_spec, is_dssd = load_experiment_spec(arch_check='random')


def catch_assert_error(spec):
    with pytest.raises(AssertionError):
        spec_validator(spec)


def assert_check(spec):
    try:
        spec_validator(spec)
    except AssertionError:
        return False

    return True


def test_spec_validator():
    experiment_spec, _ = load_experiment_spec()
    # ssd_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.ssd_config.aspect_ratios_global = "[]"
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.ssd_config.aspect_ratios = "[[1.0, 2.0, 0.5],  [1.0, 2.0, 0.5, 3.0, 1.0/3.0], \
        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],  \
            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],  \
                [1.0, 2.0, 0.5],  [1.0, 2.0, 0.5]]"
    test_spec.ssd_config.aspect_ratios_global = ""
    assert assert_check(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.ssd_config.aspect_ratios = "[[-1.0, 2.0, 0.5],  [1.0, 2.0, 0.5, 3.0, 1.0/3.0], \
        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],  \
            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],  \
                [1.0, 2.0, 0.5],  [1.0, 2.0, 0.5]]"
    test_spec.ssd_config.aspect_ratios_global = ""
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.ssd_config.variances = "[0.1, 0.1, 0.2, 0.2, 1]"
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.ssd_config.arch = "renset"
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.ssd_config.arch = "mobilenet_v2"
    test_spec.ssd_config.nlayers = 0
    assert assert_check(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.ssd_config.arch = "efficientnet_b1"
    test_spec.ssd_config.nlayers = 0
    assert assert_check(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.ssd_config.arch = ""
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
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.eval_config.matching_iou_threshold = 1.1
    catch_assert_error(test_spec)
    # nms_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.nms_config.clustering_iou_threshold = 1.1
    catch_assert_error(test_spec)
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.nms_config.top_k = 0
    catch_assert_error(test_spec)
    # aug_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.augmentation_config.output_channel = 4
    catch_assert_error(test_spec)
    # dataset_config_test:
    test_spec = copy.deepcopy(experiment_spec)
    test_spec.dataset_config.data_sources[0].label_directory_path = ""
    with pytest.raises(AssertionError):
        validate_train_spec(test_spec)

    test_spec = copy.deepcopy(experiment_spec)
    test_spec.dataset_config.validation_data_sources[0].label_directory_path = ""
    with pytest.raises(AssertionError):
        validate_eval_spec(test_spec)
