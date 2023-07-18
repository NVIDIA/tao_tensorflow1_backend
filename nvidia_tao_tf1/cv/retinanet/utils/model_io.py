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

import logging
import os
import tempfile
import keras
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.common.utils import load_keras_model
from nvidia_tao_tf1.cv.retinanet.builders import model_builder
from nvidia_tao_tf1.cv.retinanet.initializers.prior_prob import PriorProbability
from nvidia_tao_tf1.cv.retinanet.layers.anchor_box_layer import RetinaAnchorBoxes
from nvidia_tao_tf1.cv.retinanet.losses.focal_loss import FocalLoss
from nvidia_tao_tf1.encoding import encoding

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)
custom_objs = {'RetinaAnchorBoxes': RetinaAnchorBoxes,
               'PriorProbability': PriorProbability}


def get_model_with_input(model, input_layer):
    """Implement a trick to replace input tensor."""
    _explored_layers = dict()
    for l in model.layers:
        _explored_layers[l.name] = [False, None]
    layers_to_explore = [l for l in model.layers if (type(l) == keras.layers.InputLayer)]
    model_outputs = {}
    # Loop until we reach the last layer.
    while layers_to_explore:
        layer = layers_to_explore.pop(0)
        # Skip layers that may be revisited in the graph to prevent duplicates.
        if not _explored_layers[layer.name][0]:
            # Check if all inbound layers explored for given layer.
            if not all([
                    _explored_layers[l.name][0]
                    for n in layer._inbound_nodes
                    for l in n.inbound_layers
                    ]):
                continue
            outputs = None
            # Visit input layer.
            if type(layer) == keras.layers.InputLayer:
                # skip input layer and use outside input tensors intead.
                _explored_layers[layer.name][0] = True
                _explored_layers[layer.name][1] = None
                layers_to_explore.extend([node.outbound_layer for
                                          node in layer._outbound_nodes])
                continue
            else:
                # Create new layer.
                layer_config = layer.get_config()
                with keras.utils.CustomObjectScope({'PriorProbability': PriorProbability}):
                    new_layer = type(layer).from_config(layer_config)
                # Add to model.
                outputs = []
                for node in layer._inbound_nodes:
                    prev_outputs = []
                    for idx, l in enumerate(node.inbound_layers):
                        if type(l) == keras.layers.InputLayer:
                            prev_outputs.append(input_layer)
                        else:
                            keras_layer = _explored_layers[l.name][1]
                            _tmp_output = keras_layer.get_output_at(node.node_indices[idx])
                            prev_outputs.append(_tmp_output)
                    assert prev_outputs, "Expected non-input layer to have inputs."
                    if len(prev_outputs) == 1:
                        prev_outputs = prev_outputs[0]
                    outputs.append(new_layer(prev_outputs))
                if len(outputs) == 1:
                    outputs = outputs[0]
                weights = layer.get_weights()
                if weights is not None and type(layer) != keras.layers.Dense:
                    try:
                        new_layer.set_weights(weights)
                    except ValueError:
                        print("{} is NOT loaded".format(layer.name))
            outbound_nodes = layer._outbound_nodes
            if not outbound_nodes:
                model_outputs[layer.output.name] = outputs
            layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])
            # Mark current layer as visited and assign output nodes to the layer.
            _explored_layers[layer.name][0] = True
            _explored_layers[layer.name][1] = new_layer
        else:
            continue
    # Create new keras model object from pruned specifications.
    # only use input_image as Model Input.
    output_tensors = [model_outputs[l.name] for l in model.outputs if l.name in model_outputs]
    new_model = keras.models.Model(inputs=input_layer, outputs=output_tensors, name=model.name)

    return new_model


def load_model(model_path, experiment_spec=None, input_tensor=None, key=None):
    """Load a model either in .h5 format, .tlt format or .hdf5 format."""

    _, ext = os.path.splitext(model_path)
    if ext == '.h5':
        # build model and load weights
        assert experiment_spec is not None, "To load weights, spec file must be provided"
        model = model_builder.build(experiment_spec, model_only=True, input_tensor=input_tensor)
        model.load_weights(model_path)

    elif ext == '.hdf5':
        focalloss = FocalLoss(loc_loss_weight=experiment_spec.retinanet_config.loss_loc_weight,
                              alpha=experiment_spec.retinanet_config.focal_loss_alpha,
                              gamma=experiment_spec.retinanet_config.focal_loss_gamma)
        custom_objs['compute_loss'] = focalloss.compute_loss
        # directly load model, add dummy loss since loss is never required.
        if input_tensor is None:
            model = load_keras_model(model_path,
                                     custom_objects=custom_objs)
            bs, im_channel, im_height, im_width = model.layers[0].input_shape[:]
            # make bs implicit for DALI trained model
            if bs is not None:
                new_input = keras.layers.Input(shape=(im_channel, im_height, im_width),
                                               name="Input")
                model = get_model_with_input(model, new_input)
        else:
            input_layer = keras.layers.Input(tensor=input_tensor, name="Input")
            model = load_keras_model(model_path, custom_objects=custom_objs)
            model = get_model_with_input(model, input_layer)

    elif ext == '.tlt':
        os_handle, temp_file_name = tempfile.mkstemp(suffix='.hdf5')
        os.close(os_handle)
        if isinstance(key, str):
            key = key.encode()
        with open(temp_file_name, 'wb') as temp_file, open(expand_path(model_path), 'rb') as encoded_file:
            encoding.decode(encoded_file, temp_file, key)
            encoded_file.close()
            temp_file.close()

        # recursive call
        model = load_model(temp_file_name, experiment_spec, input_tensor, None)
        os.remove(temp_file_name)

    else:
        raise NotImplementedError("{0} file is not supported!".format(ext))
    return model


def load_model_as_pretrain(model_path, load_graph, n_classes,
                           experiment_spec=None, input_tensor=None, key=None,
                           kernel_regularizer=None, resume_training=False):
    """
    Load a model as pretrained weights.

    If the model is pruned, just return the model.

    Returns:
        model_train: model for training
        model_eval: model for eval
        optimizer: None if not resume_training
    """

    model_train, model_eval = model_builder.build(experiment_spec, n_classes,
                                                  kernel_regularizer=kernel_regularizer,
                                                  input_tensor=input_tensor)
    if not model_path:
        logger.info("Building model from spec file...")
        return model_train, model_eval, None
    model_load = load_model(model_path, experiment_spec, None, key)
    if resume_training:
        logger.info("Resume from checkpoint...")
        # using DALI
        if input_tensor is not None:
            input_layer = keras.layers.Input(tensor=input_tensor, name="Input")
            return get_model_with_input(model_load, input_layer), model_load, model_load.optimizer
        return model_load, model_load, model_load.optimizer
    strict_mode = True
    logger.info("Loading model weights...")
    for layer in model_load.layers[1:]:
        # The layer must match up to retinanet layers.
        if layer.name.find('retinanet_') != -1:
            strict_mode = False
        try:
            l_return = model_train.get_layer(layer.name)
        except ValueError:
            # Some layers are not there
            continue
        try:
            l_return.set_weights(layer.get_weights())
        except ValueError:
            if strict_mode:
                logger.info("Pruned model detected...")
                assert load_graph, "Your pretrained model is a pruned graph. \
                    Please use pruned_model_path to set the path."
                # This is a pruned model
                model_config = model_load.get_config()
                for layer, layer_config in zip(model_load.layers, model_config['layers']):
                    if hasattr(layer, 'kernel_regularizer'):
                        layer_config['config']['kernel_regularizer'] = kernel_regularizer
                reg_model = keras.models.Model.from_config(model_config,
                                                           custom_objects=custom_objs)
                reg_model.set_weights(model_load.get_weights())
                # if use dali, replace input
                if input_tensor is not None:
                    input_layer = keras.layers.Input(tensor=input_tensor, name="Input")
                    reg_model = get_model_with_input(reg_model, input_layer)
                # Alternative: save reg_model to tmp file
                # os_handle, temp_file_name = tempfile.mkstemp(suffix='.hdf5')
                # os.close(os_handle)
                # reg_model.save(temp_file_name, overwrite=True, include_optimizer=False)
                # reg_model = load_model(temp_file_name, experiment_spec,
                #                        input_tensor, None)
                return reg_model, model_load, None

    return model_train, model_eval, None


def save_model(keras_model, model_path, key, save_format=None):
    """Save a model to either .h5, .tlt or .hdf5 format."""

    _, ext = os.path.splitext(model_path)
    if (save_format is not None) and (save_format != ext):
        # recursive call to save a correct model
        return save_model(keras_model, model_path + save_format, key, None)

    if ext == '.h5':
        keras_model.save_weights(model_path)

    elif ext == '.hdf5':
        keras_model.save(model_path, overwrite=True, include_optimizer=False)

    else:
        raise NotImplementedError("{0} file is not supported!".format(ext))
    return model_path
