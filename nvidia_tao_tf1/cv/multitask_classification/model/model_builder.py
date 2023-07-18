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

"""IVA Colornet model construction wrapper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import AveragePooling2D, Dense, Flatten
from keras.layers import Input
from keras.models import Model

from nvidia_tao_tf1.core.templates.utils import _leaky_conv, arg_scope
from nvidia_tao_tf1.cv.common.models.backbones import get_backbone


def add_classification_branch(nclasses_dict, base_model, data_format,
                              kernel_regularizer, bias_regularizer):
    """Add classification branches for multitasknet."""
    output = base_model.output
    output_shape = output.get_shape().as_list()
    all_out = []
    for class_task, nclasses in sorted(nclasses_dict.items(), key=lambda x: x[0]):

        # Implement a trick to use _leaky_conv for Conv+BN+ReLU.
        with arg_scope([_leaky_conv],
                       use_batch_norm=True,
                       data_format=data_format,
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       alpha=0,
                       kernel=(3, 3),
                       padding='same',
                       freeze_bn=False,
                       use_bias=False,
                       force_relu=True):

            attribute_output = _leaky_conv(output, 128,
                                           name='multitask_'+class_task+'_conv1')
            attribute_output = _leaky_conv(attribute_output, 64,
                                           name='multitask_'+class_task+'_conv2')

        if data_format == 'channels_first':
            pool_size = (output_shape[-2], output_shape[-1])
        else:
            pool_size = (output_shape[-3], output_shape[-2])

        attribute_output = AveragePooling2D(pool_size=pool_size,
                                            data_format=data_format,
                                            padding='valid',
                                            name='multitask_'+class_task+'_pool')(attribute_output)
        attribute_output = Flatten(name='multitask_'+class_task+'_flatten')(attribute_output)

        attribute_output = Dense(nclasses, activation='softmax',
                                 kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 name=class_task)(attribute_output)
        all_out.append(attribute_output)
    final_model = Model(inputs=base_model.input, outputs=all_out,
                        name='multitask_' + base_model.name)
    return final_model


def get_model(nclasses_dict,
              arch="squeezenet",
              input_shape=(3, 224, 224),
              data_format=None,
              kernel_regularizer=None,
              bias_regularizer=None,
              freeze_blocks=None,
              **kwargs):
    '''Wrapper function to construct a model.'''

    input_image = Input(shape=input_shape)
    base_model = get_backbone(backbone=arch,
                              input_tensor=input_image,
                              data_format=data_format,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              freeze_blocks=freeze_blocks,
                              **kwargs)

    final_model = add_classification_branch(nclasses_dict, base_model, data_format,
                                            kernel_regularizer, bias_regularizer)

    return final_model
