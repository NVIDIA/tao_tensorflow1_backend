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

"""Helper function to load model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from tensorflow import keras
from nvidia_tao_tf1.cv.common.utils import load_tf_keras_model
from nvidia_tao_tf1.cv.lprnet.loss.wrap_ctc_loss import WrapCTCLoss
from nvidia_tao_tf1.cv.lprnet.models import eval_builder, model_builder
from nvidia_tao_tf1.encoding import encoding

CUSTOM_OBJS = {
}


def load_model(model_path, max_label_length, key=None):
    """Load a model either in .h5 format, .tlt format or .hdf5 format."""

    _, ext = os.path.splitext(model_path)
    if ext == '.hdf5':
        ctc_loss = WrapCTCLoss(max_label_length)
        CUSTOM_OBJS['compute_loss'] = ctc_loss.compute_loss
        # directly load model, add dummy loss since loss is never required.
        model = load_tf_keras_model(model_path, custom_objects=CUSTOM_OBJS)

    elif ext == '.tlt':
        os_handle, temp_file_name = tempfile.mkstemp(suffix='.hdf5')
        os.close(os_handle)

        with open(temp_file_name, 'wb') as temp_file, open(model_path, 'rb') as encoded_file:
            encoding.decode(encoded_file, temp_file, key)
            encoded_file.close()
            temp_file.close()

        # recursive call
        model = load_model(temp_file_name, max_label_length, None)
        os.remove(temp_file_name)

    else:
        raise NotImplementedError("{0} file is not supported!".format(ext))
    return model


def load_eval_model(model_path, max_label_length, key):
    """Load model in evaluation mode."""

    keras.backend.set_learning_phase(0)
    model_eval = load_model(model_path, max_label_length, key)
    model_eval = eval_builder.build(model_eval)
    keras.backend.set_learning_phase(1)

    return model_eval


def load_model_as_pretrain(model_path, experiment_spec, key=None,
                           kernel_regularizer=None,
                           bias_regularizer=None,
                           resume_from_training=False):
    """
    Load a model as pretrained weights.

    If the model is pruned, just return the model.

    Always return two models, first is for training, last is a template with input placeholder.
    """

    model_train, model_eval, time_step = \
        model_builder.build(experiment_spec,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer)
    model_load = load_model(model_path,
                            experiment_spec.lpr_config.max_label_length,
                            key)

    if resume_from_training:
        model_eval = load_eval_model(model_path,
                                     experiment_spec.lpr_config.max_label_length,
                                     key)
        return model_load, model_eval, time_step

    strict_mode = True
    error_layers = []
    loaded_layers = []
    for layer in model_train.layers[1:]:
        try:
            l_return = model_load.get_layer(layer.name)
        except ValueError:
            if layer.name[-3:] != 'qdq' and strict_mode:
                error_layers.append(layer.name)
            # Some layers are not there
            continue
        try:
            wts = l_return.get_weights()
            if len(wts) > 0:
                layer.set_weights(wts)
                loaded_layers.append(layer.name)
        except ValueError:

            if layer.name == "td_dense":
                continue

            if strict_mode:
                # This is a pruned model
                print('The shape of this layer does not match original model:', layer.name)
                print('Loading the model as a pruned model.')
                model_config = model_load.get_config()
                for layer, layer_config in zip(model_load.layers, model_config['layers']):
                    if hasattr(layer, 'kernel_regularizer'):
                        layer_config['config']['kernel_regularizer'] = kernel_regularizer
                reg_model = keras.models.Model.from_config(model_config, custom_objects=CUSTOM_OBJS)
                reg_model.set_weights(model_load.get_weights())

                os_handle, temp_file_name = tempfile.mkstemp(suffix='.hdf5')
                os.close(os_handle)
                reg_model.save(temp_file_name, overwrite=True, include_optimizer=False)

                model_eval = load_eval_model(model_path,
                                             experiment_spec.lpr_config.max_label_length,
                                             key)

                train_model = load_model(temp_file_name,
                                         experiment_spec.lpr_config.max_label_length,
                                         )
                os.remove(temp_file_name)
                return train_model, model_eval, time_step
            error_layers.append(layer.name)
    if len(error_layers) > 0:
        print('Weights for those layers can not be loaded:', error_layers)
        print('STOP trainig now and check the pre-train model if this is not expected!')

    print("Layers that load weights from the pretrained model:", loaded_layers)

    return model_train, model_eval, time_step


def save_model(keras_model, model_path, key, save_format=None):
    """Save a model to either .h5, .tlt or .hdf5 format."""

    _, ext = os.path.splitext(model_path)
    if (save_format is not None) and (save_format != ext):
        # recursive call to save a correct model
        return save_model(keras_model, model_path + save_format, key, None)

    if ext == '.hdf5':
        keras_model.save(model_path, overwrite=True, include_optimizer=True)
    else:
        raise NotImplementedError("{0} file is not supported!".format(ext))
    return model_path
