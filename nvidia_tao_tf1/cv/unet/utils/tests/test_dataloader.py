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

"""Test UNet dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest
import tensorflow as tf
from nvidia_tao_tf1.cv.unet.model.build_unet_model import build_model, select_model_proto
from nvidia_tao_tf1.cv.unet.model.utilities import build_target_class_list, get_num_unique_train_ids
from nvidia_tao_tf1.cv.unet.model.utilities import get_train_class_mapping
from nvidia_tao_tf1.cv.unet.model.utilities import initialize, initialize_params
from nvidia_tao_tf1.cv.unet.model.utilities import update_model_params
from nvidia_tao_tf1.cv.unet.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.unet.utils.data_loader import Dataset


tf.compat.v1.enable_eager_execution()


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
@pytest.mark.parametrize("data_config",
                         [
                          '../../unet/experiment_specs/test_isbi.txt',
                          '../../unet/experiment_specs/test_peoplesemseg.txt'
                         ])
def test_unet_dataloader(data_config):
    bs = 1
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    spec_path = os.path.join(file_path, data_config)

    experiment_spec = load_experiment_spec(spec_path, merge_from_default=False)

    # Initialize the environment
    initialize(experiment_spec)
    # Initialize Params
    params = initialize_params(experiment_spec)
    target_classes = build_target_class_list(
        experiment_spec.dataset_config.data_class_config)
    num_classes = get_num_unique_train_ids(target_classes)
    target_classes_train_mapping = get_train_class_mapping(target_classes)
    print(target_classes_train_mapping)

    model_config = select_model_proto(experiment_spec)
    unet_model = build_model(m_config=model_config,
                             target_class_names=target_classes)
    params = update_model_params(params=params, unet_model=unet_model,
                                 experiment_spec=experiment_spec,
                                 target_classes=target_classes
                                 )

    dataset = Dataset(batch_size=bs,
                      fold=params.crossvalidation_idx,
                      augment=params.augment,
                      params=params,
                      phase="train",
                      target_classes=target_classes)

    dataset_iterator = dataset.input_fn()

    iterator = dataset_iterator.make_one_shot_iterator()
    data, data_y = iterator.get_next()
    data_X = data["features"]
    img_arr_np, mask_arr_np = data_X.numpy(), data_y.numpy()
    assert(img_arr_np.shape == (1, model_config.model_input_channels,
                                model_config.model_input_height,
                                model_config.model_input_width))
    assert(mask_arr_np.shape == (1, num_classes, model_config.model_input_height,
                                 model_config.model_input_width))
