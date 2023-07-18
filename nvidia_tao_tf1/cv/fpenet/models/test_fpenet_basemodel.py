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
"""Tests for FpeNet BaseModel model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from nvidia_tao_tf1.core.models.templates.utils import count_layers_by_class_name
from nvidia_tao_tf1.cv.fpenet.models.fpenet_basemodel import FpeNetBaseModel


def build_sample_images(name='inputs',
                        data_format='channels_first',
                        channels=1,
                        height=80,
                        width=80):
    """Construct FpeNet model for testing.

    Args:
        name (str): Name of the input tensor. Default value is 'inputs'
        data_format (str): Expected tensor format, either `channels_first` or `channels_last`.
            Default value is `channels_first`.
        channels, height, width (all int): Input image dimentions.
    """

    # Set sample inputs.
    if data_format == 'channels_first':
        shape = (channels, height, width)
    elif data_format == 'channels_last':
        shape = (height, width, channels)
    else:
        raise ValueError(
            'Provide either `channels_first` or `channels_last` for `data_format`.'
        )
    image_face = Input(shape=shape, name=name)
    return image_face


def test_fpenet_model_builder():
    """Test FpeNetBaseModel constructor."""
    image_face = build_sample_images(name='input_face')

    # Test: 'FpeNet_base'
    model_parameters = {
        'beta': 0.01,
        'dropout_rate': 0.5,
        'freeze_Convlayer': None,
        'pretrained_model_path': None,
        'regularizer_type': 'l2',
        'regularizer_weight': 1.0e-05,
        'type': 'FpeNet_base',
        'use_upsampling_layer': False,
        'visualization_parameters': None,
    }

    model = FpeNetBaseModel(model_parameters)
    model.build(image_face, num_keypoints=80)
    keras_model = model._keras_model

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 17
    assert count_layers_by_class_name(keras_model, ['Dense']) == 0
    assert count_layers_by_class_name(keras_model, ['Reshape']) == 0
    assert count_layers_by_class_name(keras_model, ['Dropout']) == 0
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 4
    assert count_layers_by_class_name(keras_model, ['Concatenate']) == 4
    assert count_layers_by_class_name(keras_model, ['Softargmax']) == 1
    assert keras_model.count_params() == 588944

    # Test: 'FpeNet_base_5x5_conv'
    model_parameters = {
        'beta': 0.01,
        'dropout_rate': 0.5,
        'freeze_Convlayer': None,
        'pretrained_model_path': None,
        'regularizer_type': 'l2',
        'regularizer_weight': 1.0e-05,
        'type': 'FpeNet_base_5x5_conv',
        'use_upsampling_layer': False,
        'visualization_parameters': None,
    }

    model = FpeNetBaseModel(model_parameters)
    model.build(image_face, num_keypoints=80)
    keras_model = model._keras_model

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 16
    assert count_layers_by_class_name(keras_model, ['Dense']) == 0
    assert count_layers_by_class_name(keras_model, ['Reshape']) == 0
    assert count_layers_by_class_name(keras_model, ['Dropout']) == 0
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 4
    assert count_layers_by_class_name(keras_model, ['Concatenate']) == 4
    assert count_layers_by_class_name(keras_model, ['Softargmax']) == 1
    assert keras_model.count_params() == 1109072

    # Test: 68 points model
    model_parameters = {
        'beta': 0.01,
        'dropout_rate': 0.5,
        'freeze_Convlayer': None,
        'pretrained_model_path': None,
        'regularizer_type': 'l2',
        'regularizer_weight': 1.0e-05,
        'type': 'FpeNet_base',
        'use_upsampling_layer': False,
        'visualization_parameters': None,
    }

    model = FpeNetBaseModel(model_parameters)
    model.build(image_face, num_keypoints=68)
    keras_model = model._keras_model

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 17
    assert count_layers_by_class_name(keras_model, ['Dense']) == 0
    assert count_layers_by_class_name(keras_model, ['Reshape']) == 0
    assert count_layers_by_class_name(keras_model, ['Dropout']) == 0
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 4
    assert count_layers_by_class_name(keras_model, ['Concatenate']) == 4
    assert count_layers_by_class_name(keras_model, ['Softargmax']) == 1
    assert keras_model.count_params() == 588164

    # Test: Upsampling layer model
    model_parameters = {
        'beta': 0.01,
        'dropout_rate': 0.5,
        'freeze_Convlayer': None,
        'pretrained_model_path': None,
        'regularizer_type': 'l2',
        'regularizer_weight': 1.0e-05,
        'type': 'FpeNet_base',
        'use_upsampling_layer': True,
        'visualization_parameters': None,
    }

    model = FpeNetBaseModel(model_parameters)
    model.build(image_face, num_keypoints=80)
    keras_model = model._keras_model

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 17
    assert count_layers_by_class_name(keras_model, ['Dense']) == 0
    assert count_layers_by_class_name(keras_model, ['Reshape']) == 0
    assert count_layers_by_class_name(keras_model, ['Dropout']) == 0
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 4
    assert count_layers_by_class_name(keras_model, ['Concatenate']) == 4
    assert count_layers_by_class_name(keras_model, ['Softargmax']) == 1
    assert keras_model.count_params() == 523408

    # Test: Varying input image dimension
    image_face = build_sample_images(name='input_face', height=112, width=112)

    model_parameters = {
        'beta': 0.01,
        'dropout_rate': 0.5,
        'freeze_Convlayer': None,
        'pretrained_model_path': None,
        'regularizer_type': 'l2',
        'regularizer_weight': 1.0e-05,
        'type': 'FpeNet_base',
        'use_upsampling_layer': False,
        'visualization_parameters': None,
    }

    model = FpeNetBaseModel(model_parameters)
    model.build(image_face, num_keypoints=80)
    keras_model = model._keras_model

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 17
    assert count_layers_by_class_name(keras_model, ['Dense']) == 0
    assert count_layers_by_class_name(keras_model, ['Reshape']) == 0
    assert count_layers_by_class_name(keras_model, ['Dropout']) == 0
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 4
    assert count_layers_by_class_name(keras_model, ['Concatenate']) == 4
    assert count_layers_by_class_name(keras_model, ['Softargmax']) == 1
    assert keras_model.count_params() == 601232
