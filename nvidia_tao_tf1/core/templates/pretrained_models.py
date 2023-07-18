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

"""Pull Google Open Images pre-trained models from NVidia GitLab."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from keras.layers import AveragePooling2D, Dense, Flatten, Input
from keras.models import Model
from keras.utils.data_utils import get_file
from nvidia_tao_tf1.core.templates import resnet
from nvidia_tao_tf1.core.templates import vgg

BASE_MODEL_PATH= os.getenv("BASE_MODEL_PATH", "")

def add_dense_head(nclasses, base_model, data_format,
                   kernel_regularizer=None, bias_regularizer=None):
    """Add dense head to the base model."""
    output = base_model.output
    output_shape = base_model.output.get_shape().as_list()
    # use average pooling and flatten to replace global average pooling and add dense head
    output = AveragePooling2D(pool_size=(output_shape[-2], output_shape[-1]),
                              data_format=data_format, padding='valid')(output)
    output = Flatten()(output)
    output = Dense(nclasses, activation='softmax', name='predictions',
                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(output)
    final_model = Model(inputs=base_model.input, outputs=output, name=base_model.name)
    return final_model


def ResNet(nlayers=18,
           input_shape=(3, 224, 224),
           add_head=False,
           nclasses=1000,
           data_format='channels_first',
           kernel_regularizer=None,
           bias_regularizer=None):
    """Build ResNet based on Pretrained weights."""
    if nlayers == 10:
        model_name = 'resnet10.h5'
    elif nlayers == 18:
        model_name = 'resnet18.h5'
    elif nlayers == 34:
        model_name = 'resnet34.h5'
    elif nlayers == 50:
        model_name = 'resnet50.h5'
    else:
        raise NotImplementedError('There is no pre-trained models for this number of layers')

    assert data_format == 'channels_first', \
        "Pretrained weights only available for channels_first models."

    url = BASE_MODEL_PATH + model_name
    input_image = Input(shape=input_shape)
    base_model = resnet.ResNet(nlayers=nlayers,
                               input_tensor=input_image,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               data_format=data_format,
                               use_batch_norm=True,
                               activation_type='relu',
                               all_projections=True,
                               use_pooling=False)
    model_path = get_file(model_name, url, cache_subdir='models')
    base_model.load_weights(model_path)
    if not add_head:
        return base_model

    return add_dense_head(nclasses, base_model, data_format,
                          kernel_regularizer, bias_regularizer)


def VggNet(nlayers=16,
           input_shape=(3, 224, 224),
           add_head=False,
           nclasses=1000,
           data_format='channels_first',
           kernel_regularizer=None,
           bias_regularizer=None):
    """Build VGG based on pretrained weights."""
    if nlayers == 16:
        model_name = 'vgg16.h5'
    elif nlayers == 19:
        model_name = 'vgg19.h5'
    else:
        raise NotImplementedError('There is no pre-trained models for this number of layers')

    assert data_format == 'channels_first', \
        "Pretrained weights only available for channels_first models."

    url = BASE_MODEL_PATH + model_name
    input_image = Input(shape=input_shape)
    base_model = vgg.VggNet(nlayers=nlayers,
                            inputs=input_image,
                            data_format=data_format,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            use_batch_norm=True,
                            activation_type='relu',
                            use_pooling=False)
    model_path = get_file(model_name, url, cache_subdir='models')
    base_model.load_weights(model_path)
    if not add_head:
        return base_model

    return add_dense_head(nclasses, base_model, data_format,
                          kernel_regularizer, bias_regularizer)
