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
"""Patch keras.engine.saving so that we can load pretrained weights for TimeDistributed layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import warnings

import keras
from keras import backend as K
from keras.engine.saving import load_attributes_from_hdf5_group, \
                                preprocess_weights_for_loading
from keras.layers import TimeDistributed
import numpy as np


logger = logging.getLogger(__name__)


def load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=False,
                                         reshape=False):
    """Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: A list of target layers.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file and skip_mismatch=False.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        # If it is a TD layer, try to load weights with the inner layer's weights
        if type(layer) == TimeDistributed:
            if layer.layer.name:
                index.setdefault(layer.layer.name, []).append(layer)
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend,
                reshape=reshape)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:  # noqa pylint: disable = R1724
                    warnings.warn('Skipping loading of weights for '
                                  'layer {}'.format(layer.name) + ' due to mismatch '
                                  'in number of weights ({} vs {}).'.format(
                                      len(symbolic_weights), len(weight_values)))
                    continue
                else:  # noqa pylint: disable = R1724
                    raise ValueError('Layer #' + str(k) +
                                     ' (named "' + layer.name +
                                     '") expects ' +
                                     str(len(symbolic_weights)) +
                                     ' weight(s), but the saved weights' +
                                     ' have ' + str(len(weight_values)) +
                                     ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                symbolic_shape = K.int_shape(symbolic_weights[i])
                if symbolic_shape != weight_values[i].shape:
                    if skip_mismatch:  # noqa pylint: disable = R1724
                        warnings.warn('Skipping loading of weights for '
                                      'layer {}'.format(layer.name) + ' due to '
                                      'mismatch in shape ({} vs {}).'.format(
                                          symbolic_weights[i].shape,
                                          weight_values[i].shape))
                        continue
                    else:  # noqa pylint: disable = R1724
                        raise ValueError('Layer #' + str(k) +
                                         ' (named "' + layer.name +
                                         '"), weight ' +
                                         str(symbolic_weights[i]) +
                                         ' has shape {}'.format(symbolic_shape) +
                                         ', but the saved weight has shape ' +
                                         str(weight_values[i].shape) + '.')
                else:
                    weight_value_tuples.append((symbolic_weights[i],
                                                weight_values[i]))

    K.batch_set_value(weight_value_tuples)


def _patch(f):
    """Apply the patches to the function."""
    name = f.__name__
    logger.debug('Patching %s' % name)
    keras.engine.saving.__setattr__(name, f)


def patch():
    '''Patch this function.'''
    _patch(load_weights_from_hdf5_group_by_name)
