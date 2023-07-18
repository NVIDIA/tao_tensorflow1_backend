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
"""Tests for BpNetLite model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from nvidia_tao_tf1.core.models.templates.utils import count_layers_by_class_name
from nvidia_tao_tf1.cv.bpnet.models.bpnet_lite_model import BpNetLiteModel
from nvidia_tao_tf1.cv.bpnet.models.templates.utils import add_input

NUM_DENSE_LAYERS = 0
NUM_RESHAPE_LAYERS = 0
NUM_DROPOUT_LAYERS = 0


def test_bpnet_lite_model_builder():
    """Test BpNetLite model builder."""
    input_tensor = add_input(name='input', data_format='channels_last')

    ##################################
    # BpNetLiteModel default params
    ##################################
    default_params = {
        'backbone_attributes': {
            'architecture': 'vgg',
            'mtype': 'default',
            'use_bias': False
        },
        'heat_channels': 19,
        'paf_channels': 38,
        'stages': 6,
        'stage1_kernel_sizes': [3, 3, 3],
        'stageT_kernel_sizes': [7, 7, 7, 7, 7],
        'use_self_attention': False,
        'regularization_type': 'l2',
        'kernel_regularization_factor': 5e-4,
        'bias_regularization_factor': 0
    }

    model = BpNetLiteModel(**default_params)
    model.build(input_tensor)
    keras_model = model._keras_model

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 64
    assert count_layers_by_class_name(keras_model,
                                      ['Dense']) == NUM_DENSE_LAYERS
    assert count_layers_by_class_name(keras_model,
                                      ['Reshape']) == NUM_RESHAPE_LAYERS
    assert count_layers_by_class_name(keras_model,
                                      ['Dropout']) == NUM_DROPOUT_LAYERS
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 0
    assert count_layers_by_class_name(
        keras_model, ['Concatenate']) == (default_params['stages'] - 1)
    assert keras_model.count_params() == 30015638

    #################################
    # BpNetLiteModel with 3 stages
    #################################
    model_params = copy.deepcopy(default_params)
    model_params['stages'] = 3
    model = BpNetLiteModel(**model_params)
    model.build(input_tensor)
    keras_model = model._keras_model

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 37
    assert count_layers_by_class_name(keras_model,
                                      ['Dense']) == NUM_DENSE_LAYERS
    assert count_layers_by_class_name(keras_model,
                                      ['Reshape']) == NUM_RESHAPE_LAYERS
    assert count_layers_by_class_name(keras_model,
                                      ['Dropout']) == NUM_DROPOUT_LAYERS
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 0
    assert count_layers_by_class_name(
        keras_model, ['Concatenate']) == (model_params['stages'] - 1)
    assert keras_model.count_params() == 16777835

    #####################################
    # BpNetLiteModel with helnet18 base
    #####################################
    model_params = copy.deepcopy(default_params)
    model_params['backbone_attributes'] = {
        'architecture': 'helnet',
        'mtype': 's8_3rdblock',
        "nlayers": 18,
        'use_batch_norm': False
    }
    model_params['stages'] = 3
    model = BpNetLiteModel(**model_params)
    model.build(input_tensor)
    keras_model = model._keras_model

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 39
    assert count_layers_by_class_name(keras_model,
                                      ['Dense']) == NUM_DENSE_LAYERS
    assert count_layers_by_class_name(keras_model,
                                      ['Reshape']) == NUM_RESHAPE_LAYERS
    assert count_layers_by_class_name(keras_model,
                                      ['Dropout']) == NUM_DROPOUT_LAYERS
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 0
    assert count_layers_by_class_name(
        keras_model, ['Concatenate']) == (model_params['stages'] - 1)
    assert keras_model.count_params() == 12463531
