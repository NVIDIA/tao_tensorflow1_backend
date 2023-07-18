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
"""test UNet Model Building."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras
import pytest
from nvidia_tao_tf1.cv.unet.model.build_unet_model import build_model, select_model_proto
from nvidia_tao_tf1.cv.unet.model.utilities import build_target_class_list, get_num_unique_train_ids
from nvidia_tao_tf1.cv.unet.model.utilities import initialize
from nvidia_tao_tf1.cv.unet.proto.model_config_pb2 import ModelConfig
from nvidia_tao_tf1.cv.unet.spec_handler.spec_loader import load_experiment_spec


@pytest.mark.parametrize("arch, nlayers",
                         [
                          ('vanilla_unet_dynamic', 1),
                          ('efficientnet_b0', 1),
                          ('resnet', 10),
                          ('resnet', 18),
                          ('resnet', 34),
                          ('resnet', 50),
                          ('resnet', 101),
                          ('vgg', 16),
                          ('vgg', 19)
                         ])
def test_unet(arch, nlayers):
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    spec_path = os.path.join(file_path, '../../unet/experiment_specs/test_isbi.txt')
    experiment_spec = load_experiment_spec(spec_path, merge_from_default=False)

    for init in [ModelConfig.KernelInitializer.HE_UNIFORM,
                 ModelConfig.KernelInitializer.HE_NORMAL,
                 ModelConfig.KernelInitializer.GLOROT_UNIFORM]:
        # Initialize the environment
        initialize(experiment_spec)
        target_classes = build_target_class_list(
            experiment_spec.dataset_config.data_class_config)
        model_config = select_model_proto(experiment_spec)

        model_config.arch = arch
        model_config.num_layers = nlayers
        model_config.initializer = init
        for use_batch_norm in [True, False]:
            for freeze_bn in [False, False]:
                model_config.use_batch_norm = use_batch_norm
                model_config.freeze_bn = freeze_bn
        unet_model = build_model(m_config=model_config,
                                 target_class_names=target_classes)

        img_height, img_width, img_channels = \
            model_config.model_input_height, \
            model_config.model_input_width, \
            model_config.model_input_channels

        num_classes = get_num_unique_train_ids(target_classes)
        unet_model.construct_model(input_shape=(img_channels,
                                                img_height, img_width))
        assert(unet_model.keras_model.layers[-1].output_shape[1:] ==
               (num_classes, img_height, img_width))
        unet_model.keras_model.summary()
        keras.backend.clear_session()
