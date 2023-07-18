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
"""Tests for FpeNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import remove
from os.path import splitext

from keras.layers import Input
from keras.models import load_model
from keras.models import model_from_json
import pytest

from nvidia_tao_tf1.core.models.templates.utils import count_layers_by_class_name
from nvidia_tao_tf1.cv.fpenet.models.custom.softargmax import Softargmax
from nvidia_tao_tf1.cv.fpenet.models.fpenet import FpeNetModel


def load_keras_model(file_name, remove_file=False):
    """Load a Keras model from an HDF5 file or from a JSON file.

    Args:
        file_name (str): File name of the model file, must end on '.h5', '.hdf5' or '.json'.
        remove_file (bool): Toggles if the file shall be removed after successful loading.
    """
    _, extension = splitext(file_name.lower())
    if extension in ['.h5', '.hdf5']:
        model = load_model(
            file_name,
            custom_objects={'Softargmax': Softargmax},
            compile=False)
    elif extension == '.json':
        with open(file_name, 'r') as json_file:
            json_content = json_file.read()
            model = model_from_json(
                json_content, custom_objects={'Softargmax': Softargmax})
    else:
        raise ValueError(
            'Can only load a model with extensions .h5, .hdf5 or .json, \
            got %s.' % extension)
    if remove_file:
        remove(file_name)
    return model


def save_keras_model(keras_model, base_name):
    """Save a model to JSON and HDF5 format and return their file paths.

    Args:
        keras_model (Model): Model to be saved.
        base_name (str): Base name for the files to be written.
    """
    json_string = keras_model.to_json()
    json_file_name = base_name + '.json'
    with open(json_file_name, 'w') as json_file:
        json_file.write(json_string)

    hdf5_file_name = base_name + '.h5'
    keras_model.save(hdf5_file_name)

    return json_file_name, hdf5_file_name


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


def model_io(keras_model, base_name, remove_file):
    """ Serialize a model to HDF5 and JSON format, an then deserialize it again.
    Args:
        keras_model (Model): The Keras model to be tested.
        base_name (str): Name of the files, without extension.
        remove_file (bool): Toggles if the files shall be removed after successful loading.
    """
    json_file_name, hdf5_file_name = save_keras_model(keras_model, base_name)
    model = load_keras_model(json_file_name, remove_file)
    model = load_keras_model(hdf5_file_name, remove_file)
    return model


@pytest.mark.parametrize('data_format', ['channels_first'])
def test_layer_counts_default(
        data_format,
        save=False,
):
    """Test for correct layer counts of the default version with classical spatial convolutions.

    Args:
        data_format (str): Expected tensor format, either `channels_first` or `channels_last`.
            Default value is `channels_first`.
        save (bool): Toggles it the model should be serialized as JSON and HDF5 format.
    """
    image_face = build_sample_images(
        name='input_face', data_format=data_format)
    blocks_decoder = [[(3, 64), (1, 64)]] * 4
    model = FpeNetModel(
        pooling=True,
        use_batch_norm=False,
        data_format=data_format,
        kernel_regularizer=None,
        bias_regularizer=None,
        activation_type='relu',
        activation_kwargs=None,
        blocks_encoder=None,
        block_trunk=None,
        blocks_decoder=blocks_decoder,
        block_post_decoder=None,
        nkeypoints=80,
        beta=0.1)

    keras_model = model.construct(image_face)
    base_name = keras_model.name

    keras_model.summary()

    assert count_layers_by_class_name(keras_model, ['InputLayer']) == 1
    assert count_layers_by_class_name(keras_model, ['Conv2D']) == 17
    assert count_layers_by_class_name(keras_model, ['Dense']) == 0
    assert count_layers_by_class_name(keras_model, ['Reshape']) == 0
    assert count_layers_by_class_name(keras_model, ['Dropout']) == 0
    assert count_layers_by_class_name(keras_model, ['MaxPooling2D']) == 4
    assert count_layers_by_class_name(keras_model, ['Concatenate']) == 4
    assert count_layers_by_class_name(keras_model, ['Softargmax']) == 1
    assert keras_model.count_params() == 523408
    remove_file = not save
    if not os.path.exists(base_name + '.h5'):
        model_io(keras_model, base_name, remove_file)
