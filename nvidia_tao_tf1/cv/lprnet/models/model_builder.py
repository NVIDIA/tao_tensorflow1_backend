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

"""TLT LPRNet model builder."""

import keras
import six
import tensorflow as tf

from nvidia_tao_tf1.cv.common.models.backbones import get_backbone
from nvidia_tao_tf1.cv.lprnet.models.lprnet_base_model import LPRNetbaseline


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


class get_iva_template_model:
    '''Get keras model from iva template and convert to tf.keras model.'''

    def __init__(self, arch, nlayers,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 freeze_blocks=None,
                 freeze_bn=None):
        '''Initialize iva template model parameters.'''

        self.arch = arch
        self.nlayers = nlayers
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.freeze_blocks = freeze_blocks
        self.freeze_bn = freeze_bn

    def __call__(self, input_tensor, trainable=True):
        '''Generate iva template model.'''

        input_shape = input_tensor.get_shape().as_list()

        if trainable:
            tf.keras.backend.set_learning_phase(1)
        else:
            tf.keras.backend.set_learning_phase(0)

        dummy_input = keras.Input(shape=(input_shape[1], input_shape[2], input_shape[3]),
                                  name="dummy_input")
        dummy_model = get_backbone(dummy_input,
                                   self.arch,
                                   data_format='channels_first',
                                   kernel_regularizer=self.kernel_regularizer,
                                   bias_regularizer=self.bias_regularizer,
                                   freeze_blocks=self.freeze_blocks,
                                   freeze_bn=self.freeze_bn,
                                   nlayers=self.nlayers,
                                   use_batch_norm=True,
                                   use_pooling=False,
                                   use_bias=False,
                                   all_projections=True,
                                   dropout=1e-3,
                                   force_relu=True)

        model_config = dummy_model.to_json()
        real_backbone = tf.keras.models.model_from_json(model_config)

        output_tensor = real_backbone(input_tensor)

        return output_tensor


def LPRNetmodel(input_shape,
                backbone_func,
                trainable,
                n_class,
                model_name,
                hidden_units=512,
                kernel_regularizer=None,
                bias_regularizer=None):
    '''build the LPRNet model and compute time_step.'''

    image_input = tf.keras.Input(shape=input_shape, name="image_input")
    model_input = tf.keras.backend.sum(image_input, axis=1, keepdims=True)

    backbone_feature = backbone_func(model_input, trainable)

    # chw -> whc
    permuted_feature = tf.keras.layers.Permute((3, 2, 1),
                                               name="permute_feature")(backbone_feature)

    permuted_feature_shape = permuted_feature.get_shape().as_list()
    time_step = permuted_feature_shape[1]
    flatten_size = permuted_feature_shape[2]*permuted_feature_shape[3]
    flatten_feature = tf.keras.layers.Reshape((time_step, flatten_size),
                                              name="flatten_feature")(permuted_feature)

    # LSTM input: [batch_size, 24, 12*300] LSTM output: [batch_size, 24, 512]
    lstm_sequence = tf.keras.layers.LSTM(units=hidden_units, return_sequences=True,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer)(flatten_feature)

    logic = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=n_class + 1,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer), name="td_dense")(lstm_sequence)

    prob = tf.keras.layers.Softmax(axis=-1)(logic)  # shape: [batch_size, 24, 36]

    if trainable:
        model = tf.keras.Model(inputs=[image_input],
                               outputs=prob,
                               name=model_name)
        return model, time_step

    # For inference and evaluation
    sequence_id = tf.keras.backend.argmax(prob, axis=-1)  # [batch_size, 24]
    sequence_prob = tf.keras.backend.max(prob, axis=-1)  # [batch_sze, 24]

    model = tf.keras.Model(inputs=[image_input],
                           outputs=[sequence_id, sequence_prob],
                           name=model_name)
    return model, time_step


def build(experiment_spec,
          kernel_regularizer=None,
          bias_regularizer=None):
    '''
    Top builder for the LPRNetmodel.

    return train_model, val_model and time_step
    '''

    input_width = experiment_spec.augmentation_config.output_width
    input_height = experiment_spec.augmentation_config.output_height
    input_channel = experiment_spec.augmentation_config.output_channel

    model_arch = experiment_spec.lpr_config.arch
    n_layers = experiment_spec.lpr_config.nlayers
    freeze_bn = eval_str(experiment_spec.lpr_config.freeze_bn)
    characters_list_file = experiment_spec.dataset_config.characters_list_file
    with open(characters_list_file, "r") as f:
        temp_list = f.readlines()
    classes = [i.strip() for i in temp_list]
    n_class = len(classes)

    input_shape = (input_channel, input_height, input_width)

    if model_arch == "baseline":
        backbone_func = LPRNetbaseline(nlayers=n_layers,
                                       freeze_bn=freeze_bn,
                                       kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer
                                       )
    else:
        # only support freeze blocks in iva template:
        freeze_blocks = eval_str(experiment_spec.lpr_config.freeze_blocks)
        backbone_func = get_iva_template_model(arch=model_arch,
                                               nlayers=n_layers,
                                               kernel_regularizer=kernel_regularizer,
                                               bias_regularizer=bias_regularizer,
                                               freeze_blocks=freeze_blocks,
                                               freeze_bn=freeze_bn
                                               )

    # build train graph
    tf.keras.backend.set_learning_phase(1)
    train_model, time_step = LPRNetmodel(input_shape, backbone_func=backbone_func,
                                         model_name="lpnet_"+model_arch+"_"+str(n_layers),
                                         n_class=n_class,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         trainable=True)

    # build eval graph
    tf.keras.backend.set_learning_phase(0)
    eval_model, _ = LPRNetmodel(input_shape, backbone_func=backbone_func,
                                model_name="lpnet_"+model_arch+"_"+str(n_layers),
                                n_class=n_class,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                trainable=False)

    # set back to training phase
    tf.keras.backend.set_learning_phase(1)

    return train_model, eval_model, time_step
