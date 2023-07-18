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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import numpy as np

from nvidia_tao_tf1.core.models.import_keras import keras as keras_fn
from nvidia_tao_tf1.core.models.templates.conv_gru_2d import ConvGRU2D
from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.utils import add_deconv_head
import nvidia_tao_tf1.core.pruning
from nvidia_tao_tf1.core.pruning.utils import find_prunable_parent
from nvidia_tao_tf1.core.templates.alexnet import AlexNet
from nvidia_tao_tf1.core.templates.helnet import HelNet
from nvidia_tao_tf1.core.templates.resnet import ResNet
import nvidia_tao_tf1.core.templates.utils
import nvidia_tao_tf1.core.utils

import pytest

keras = keras_fn()


class TestPruning(object):
    """Main class for pruning tests."""

    @staticmethod
    def create_alexnet(input_shape, data_format):
        return AlexNet(
            input_shape, add_head=True, data_format=data_format, nclasses=10, hidden_fc_neurons=256)

    def check_weights(self, pruned_model, granularity, min_num_filters, filter_counts):
        for layer in pruned_model.layers:
            weights = layer.get_weights()
            if type(layer) in [
                    keras.layers.Conv2D,
                    keras.layers.DepthwiseConv2D,
                    keras.layers.Conv2DTranspose,
                    keras.layers.Dense,
                    QuantizedConv2D
            ]:
                # When unpacking weights, the last element is always a scale factor, while the
                # remaining blobs behaves like a conv layer.
                if type(layer) == QuantizedConv2D:
                    # when the QuantizedConv2D node is generated from using
                    # modulus.models.quantized_keras_model.create_quantized_keras_model
                    # the scale factor is not intialized and weightsp[-1] is not a float
                    weights = weights[:-1]
                if len(weights) == 1:
                    kernels = weights[0]
                    biases = None
                elif len(weights) == 2:
                    kernels, biases = weights
                else:
                    raise ValueError("Unhandled number of weights: %d" % len(weights))
            if type(layer) == keras.models.Model:
                self.check_weights(layer,
                                   granularity,
                                   min_num_filters,
                                   filter_counts.pop(layer.name)
                                   )
            elif type(layer) == keras.layers.Conv2DTranspose:
                # we're not pruning these layers
                filter_count = filter_counts[layer.name]
                n_kept = kernels.shape[-2]
                assert n_kept == filter_count['total']
                if biases is not None:
                    assert n_kept == biases.shape[-1]
            elif type(layer) in [
                    keras.layers.BatchNormalization,
                    QDQ
            ]:
                # this should just propagate previous pruning
                filter_count = filter_counts[layer.name]
                to_prune = filter_count['to_prune']
                # apply granularity and min
                to_prune = min(to_prune - to_prune % granularity,
                               filter_count['total'] - min_num_filters)
                to_keep = filter_count['total'] - to_prune
                if type(layer) == keras.layers.BatchNormalization:
                    assert all([len(w) == to_keep for w in weights])
                else:
                    assert all([type(w) == np.float32 for w in weights])
            elif type(layer) == keras.layers.DepthwiseConv2D:
                # handle depthwiseconv2d specially.
                n_kept = kernels.shape[-2]
                if biases is not None:
                    assert n_kept == biases.shape[-1]
                filter_count = filter_counts[layer.name]
                if filter_count['total'] > min_num_filters:
                    assert n_kept >= min_num_filters
                    n_pruned = filter_count['total'] - n_kept
                    to_prune = filter_count['to_prune']
                    assert n_pruned == min(to_prune - to_prune % granularity,
                                           filter_count['total'] - min_num_filters)
            elif type(layer) == ConvGRU2D:
                # Check all weights (kernels and biases) share the same output dimensionality.
                n_kept = [w.shape[-1] for w in weights]
                assert len(set(n_kept)) == 1
                n_kept = n_kept[0]
                # Check that weights operating on the state variable have correct input depth.
                # U_r, U_z, U_h.
                for state_weight in weights[3:6]:
                    assert state_weight.shape[-2] == n_kept
                # Make sure we kept the min amount of filters.
                filter_count = filter_counts[layer.name]
                if filter_count['total'] > min_num_filters:
                    assert n_kept >= min_num_filters
                    n_pruned = filter_count['total'] - n_kept
                    to_prune = filter_count['to_prune']
                    # Make sure the number of pruned filters matches
                    # the expected granularity.
                    assert n_pruned == min(to_prune - to_prune % granularity,
                                           filter_count['total'] - min_num_filters)
            elif weights:
                # Checking weights for a conv2d layer.
                n_kept = kernels.shape[-1]
                if biases is not None:
                    # Make sure we pruned kernels and biases identically.
                    assert n_kept == biases.shape[-1]
                filter_count = filter_counts[layer.name]
                # Make sure we kept the min amount of filters.
                if filter_count['total'] > min_num_filters:
                    assert n_kept >= min_num_filters
                    n_pruned = filter_count['total'] - n_kept
                    to_prune = filter_count['to_prune']
                    # Make sure the number of pruned filters matches
                    # the expected granularity.
                    assert n_pruned <= min(to_prune - to_prune % granularity,
                                           filter_count['total'] - min_num_filters)

    def common(self,
               model,
               method,
               normalizer,
               criterion,
               granularity,
               min_num_filters,
               threshold,
               excluded_layers=None,
               check_output_on_input_shape=None,
               layer_config_overrides=None,
               equalization_criterion='union'):
        """Common denominator for most pruning tests.

        This method sets weights such that half of the neurons should be pruned
        considering the specified threshold but ignoring granularity, the min number
        of filters to retain and excluded layers.

        This method then proceeds to pruning the model and checks whether the expected
        number of neurons has been pruned.

        Args:
            model: the model to prune.
            method (str): pruning method.
            normalizer (str): type of normalizer to use when pruning.
            criterion (str): type of criterion to use when pruning.
            granularity (int): granularity by which to prune filters.
            min_num_filters (int): min number of filters to retain when pruning.
            threshold (float): pruning threshold.
            excluded_layers (list): list of layers to exclude when pruning.
            check_output_on_input_shape (tuple): shape to use to verify inference (output shape
                                                 and activations), or ``None`` to skip inference
                                                 checks. For multiple inputs, this can also be
                                                 passed as a list of tuples.
            layer_config_overrides (dict): A dictionary of key-value pairs used for overriding
                layer configuration. Use cases include changing regularizers after pruning.
            equalization_criterion (str): Criteria to equalize the stats of inputs to an element
                wise op layer. Options are [arithmetic_mean, geometric_mean, union, intersection].
        """
        if excluded_layers is None:
            excluded_layers = []
        assert criterion == 'L2'
        # Targeted average norm of the filters to keep (actual weights will be
        # randomly picked from a narrow uniform distribution).
        keep_norm = threshold * 4.

        if check_output_on_input_shape is not None:
            # This test only works on activations for which f(0)=0, for example:
            # "tanh", "relu", "linear".
            for layer in model.layers:
                if (layer.name not in excluded_layers and hasattr(layer, 'activation') and
                        layer.activation.__name__ not in ['linear', 'relu', 'tanh']):
                    raise ValueError("Found unsupported activation  in layer "
                                     "named %s with type %s and activation type %s" %
                                     (layer.name, type(layer), layer.activation.__name__))
            if equalization_criterion in ['intersection', 'geometric_mean']:
                raise ValueError("Unsupported merge layer equalization criterion for"
                                 "pruning output check: %s." % equalization_criterion)
            # Set the norm of neurons to prune to zero so we can match the unpruned
            # model output with the pruned model output.
            prune_norm = 0.
        else:
            # Just make neurons small enough to be pruned.
            prune_norm = threshold / 4.

        filter_counts = {}
        filter_counts = self.set_weights(model, method, normalizer, criterion, granularity,
                                         min_num_filters, keep_norm, prune_norm, excluded_layers,
                                         threshold, equalization_criterion, filter_counts)

        if check_output_on_input_shape is not None:
            batch_size = 2
            if isinstance(check_output_on_input_shape, list):
                # Multiple-input case.
                batch_data = []
                for shape in check_output_on_input_shape:
                    batch_shape = (batch_size,) + shape
                    batch_data.append(np.random.random_sample(batch_shape))
            else:
                # single-input case.
                batch_shape = (batch_size,) + check_output_on_input_shape
                batch_data = np.random.random_sample(batch_shape)

            output_before = model.predict(batch_data)
            shape_before = output_before.shape

        pruned_model = nvidia_tao_tf1.core.pruning.prune(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            excluded_layers,
            equalization_criterion=equalization_criterion,
            layer_config_overrides=layer_config_overrides
        )

        self.check_weights(pruned_model, granularity, min_num_filters, filter_counts)

        if check_output_on_input_shape is not None:
            output_after = pruned_model.predict(batch_data)
            shape_after = output_after.shape
            assert shape_before == shape_after
            assert np.allclose(output_before, output_after, rtol=1e-03, atol=1e-05)

        return pruned_model

    @staticmethod
    def get_uniform(shape, mean, boundary=0.1):
        """Return a uniform distributed sample with a randomized sign.

        Returns U(mean*(1-boundary), mean*(1+boundary)) with a random sign.

        Args:
            shape (list): shape of distribution to return.
            mean (float): float of distribution to return.
            boundary (float): relative multiplier to set range boundaries.
        """
        x = np.random.uniform(low=mean * (1 - boundary), high=mean * (1 + boundary), size=shape)
        x *= np.sign(np.random.normal(size=shape))

        return x

    def set_weights(self, model, method, normalizer, criterion, granularity, min_num_filters,
                    keep_norm, prune_norm, excluded_layers, threshold, equalization_criterion,
                    filter_counts):
        # Pass 1 : Visit only prunable layers
        for layer in model.layers:
            weights = layer.get_weights()
            norms = []
            prune_indices = []
            keep_indices = []
            if type(layer) in [
                    keras.layers.Conv2D,
                    keras.layers.DepthwiseConv2D,
                    keras.layers.Conv2DTranspose,
                    keras.layers.Dense,
                    QuantizedConv2D,
            ]:
                scale_factor = None
                if type(layer) == QuantizedConv2D:
                    scale_factor = weights[-1]
                    weights = weights[:-1]
                if len(weights) == 1:
                    kernels = weights[0]
                    biases = None
                elif len(weights) == 2:
                    kernels, biases = weights
                else:
                    raise ValueError("Unhandled number of weights: %d" % len(weights))
            if type(layer) == keras.models.Model:
                filter_counts = self.set_weights(
                    layer, method, normalizer, criterion, granularity, min_num_filters, keep_norm,
                    prune_norm, excluded_layers, threshold, equalization_criterion, filter_counts)
            elif type(layer) == keras.layers.Conv2DTranspose:
                # expected kernel shape is (kernel_width, kernel_height, output_fmaps, input_fmaps)
                n_filters = kernels.shape[-2]
                # we are not pruning these layers
                filter_counts[layer.name] = {
                    'to_keep': n_filters,
                    'to_prune': 0,
                    'total': n_filters,
                    'keep_indices': range(n_filters),
                    'prune_indices': prune_indices,
                    'norms': np.asarray(norms)
                }
            elif type(layer) in [
                keras.layers.BatchNormalization,
                QDQ
            ]:
                # Account for weights in the layer, but pass through during first pass
                # waiting all prunable and element wise layers to be explored.
                pass
            elif (
                type(layer) == keras.layers.Conv2D or
                type(layer) == QuantizedConv2D
            ):
                n_prune = 0
                n_keep = 0
                n_params_per_kernel = kernels[:, :, :, 0].size
                keep_norm_ = math.sqrt(keep_norm**2 / n_params_per_kernel)
                prune_norm_ = math.sqrt(prune_norm**2 / n_params_per_kernel)
                for i in range(kernels.shape[-1]):
                    # Randomly keep or remove filters.
                    if np.random.uniform() > 0.5 or layer.name in excluded_layers:
                        # Keep that one.
                        kernels[:, :, :, i] = self.get_uniform(kernels.shape[:3], keep_norm_)
                        if biases is not None:
                            biases[i] = keep_norm_
                        n_keep += 1
                        keep_indices.append(i)
                        norms.append(keep_norm)
                    else:
                        # Prune that one.
                        kernels[:, :, :, i] = self.get_uniform(kernels.shape[:3], prune_norm_)
                        if biases is not None:
                            biases[i] = prune_norm_
                        norms.append(prune_norm)
                        n_prune += 1
                        prune_indices.append(i)
                if biases is not None:
                    apply_weights = (kernels, biases)
                else:
                    apply_weights = (kernels,)

                # Packing weights for the QuantizedConv2D layer.
                if scale_factor:
                    apply_weights += (scale_factor,)

                layer.set_weights(apply_weights)
                filter_counts[layer.name] = {
                    'layer_name': layer.name,
                    'to_keep': n_keep,
                    'to_prune': n_prune,
                    'total': n_keep + n_prune,
                    'keep_indices': keep_indices,
                    'prune_indices': prune_indices,
                    'norms': np.asarray(norms)
                }
            elif type(layer) == keras.layers.DepthwiseConv2D:
                n_prune = 0
                n_keep = 0
                n_params_per_kernel = kernels[:, :, 0, 0].size
                keep_norm_ = math.sqrt(keep_norm ** 2 / n_params_per_kernel)
                prune_norm_ = math.sqrt(prune_norm ** 2 / n_params_per_kernel)
                for i in range(kernels.shape[-2]):
                    # Randomly keep or remove filters.
                    if np.random.uniform() > 0.5 or layer.name in excluded_layers:
                        # Keep that one.
                        kernels[:, :, i, 0] = self.get_uniform(kernels.shape[:2], keep_norm_)
                        if biases is not None:
                            biases[i] = keep_norm_
                        n_keep += 1
                        keep_indices.append(i)
                        norms.append(keep_norm)
                    else:
                        # Prune that one.
                        kernels[:, :, i, 0] = self.get_uniform(kernels.shape[:2], prune_norm_)
                        if biases is not None:
                            biases[i] = prune_norm_
                        norms.append(prune_norm)
                        n_prune += 1
                        prune_indices.append(i)
                if biases is not None:
                    layer.set_weights((kernels, biases))
                else:
                    layer.set_weights((kernels,))
                filter_counts[layer.name] = {'layer_name': layer.name,
                                             'to_keep': n_keep,
                                             'to_prune': n_prune,
                                             'total': n_keep + n_prune,
                                             'keep_indices': keep_indices,
                                             'prune_indices': prune_indices,
                                             'norms': np.asarray(norms)}
            elif type(layer) == keras.layers.Dense:
                n_prune = 0
                n_keep = 0
                n_params_per_kernel = kernels.shape[0]
                keep_norm_ = math.sqrt(keep_norm**2 / n_params_per_kernel)
                prune_norm_ = math.sqrt(prune_norm**2 / n_params_per_kernel)

                for i in range(kernels.shape[1]):
                    # Randomly keep or remove filters.
                    if np.random.uniform() > 0.5 or layer.name in excluded_layers:
                        # Keep that one.
                        kernels[:, i] = self.get_uniform(kernels.shape[:1], keep_norm_)
                        n_keep += 1
                        if biases is not None:
                            biases[i] = keep_norm_
                        keep_indices.append(i)
                        norms.append(keep_norm_)
                    else:
                        # Prune that one.
                        kernels[:, i] = self.get_uniform(kernels.shape[:1], prune_norm_)
                        if biases is not None:
                            biases[i] = prune_norm_
                        n_prune += 1
                        prune_indices.append(i)
                        norms.append(prune_norm_)
                if biases is not None:
                    layer.set_weights((kernels, biases))
                else:
                    layer.set_weights((kernels,))
                filter_counts[layer.name] = {
                    'to_keep': n_keep,
                    'to_prune': n_prune,
                    'total': n_keep + n_prune,
                    'keep_indices': keep_indices,
                    'prune_indices': prune_indices,
                    'norms': np.asarray(norms)
                }
            elif type(layer) == ConvGRU2D:
                n_prune = 0
                n_keep = 0
                # Weights are W_z, W_r, W_h, U_z, U_r, U_h, b_z, b_r, b_h.
                n_params_per_kernel = weights[0][:, :, :, 0].size
                keep_norm_ = math.sqrt(keep_norm**2 / n_params_per_kernel)
                prune_norm_ = math.sqrt(prune_norm**2 / n_params_per_kernel)
                for i in range(weights[0].shape[-1]):
                    # Randomly keep or remove filters.
                    if np.random.uniform() > 0.5 or layer.name in excluded_layers:
                        # Keep that one. Kernels W_z, W_r, W_h, U_z, U_r, U_h.
                        for kernels in weights[:6]:
                            kernels[:, :, :, i] = self.get_uniform(kernels.shape[:3], keep_norm_)
                        # Biases b_z, b_r, b_h.
                        for biases in weights[6:]:
                            biases[i] = keep_norm_
                        n_keep += 1
                        keep_indices.append(i)
                        norms.append(keep_norm)
                    else:
                        # Prune that one. Kernels W_z, W_r, W_h, U_z, U_r, U_h.
                        for kernels in weights[:6]:
                            kernels[:, :, :, i] = self.get_uniform(kernels.shape[:3], prune_norm_)
                        # Biases b_z, b_r, b_h.
                        for biases in weights[6:]:
                            biases[i] = prune_norm_
                        n_prune += 1
                        prune_indices.append(i)
                        norms.append(prune_norm)

                layer.set_weights(weights)

                filter_counts[layer.name] = {
                    'layer_name': layer.name,
                    'to_keep': n_keep,
                    'to_prune': n_prune,
                    'total': n_keep + n_prune,
                    'keep_indices': keep_indices,
                    'prune_indices': prune_indices,
                    'norms': np.asarray(norms)
                }
            elif weights:
                raise RuntimeError("Unknown layer type=%s has weights" % type(layer))
        # Equalizing inputs for layers with element wise operations.
        filter_counts = self._equalize_inputs(model, filter_counts, granularity, min_num_filters,
                                              threshold, equalization_criterion, excluded_layers)
        # Pass two: This time visit batchnorm layers.
        for layer in model.layers:
            if type(layer) in [
                keras.layers.BatchNormalization,
                QDQ
            ]:
                # We are just propagating the previous layer.
                previous_layer = []
                inbound_nodes = layer._inbound_nodes
                # For some reason, tf.keras does not always put things in a list.
                if not isinstance(inbound_nodes, list):
                    inbound_nodes = [inbound_nodes]
                for n in inbound_nodes:
                    _inbound_layers = n.inbound_layers
                    # For some reason, tf.keras does not always put things in a list.
                    if not isinstance(_inbound_layers, list):
                        _inbound_layers = [_inbound_layers]
                    for l in _inbound_layers:
                        previous_layer.append(l.name)

                filter_counts[layer.name] = filter_counts[previous_layer[0]]
            if type(layer) == keras.layers.DepthwiseConv2D:
                dw_parents = []
                dw_parents = find_prunable_parent(dw_parents, layer, True)
                filter_counts = self._match_dw_indices(dw_parents[0], layer, filter_counts,
                                                       min_num_filters, granularity, threshold,
                                                       equalization_criterion, excluded_layers)
        return filter_counts

    def _equalize_inputs(self,
                         model,
                         filter_counts,
                         granularity,
                         min_num_filters,
                         threshold,
                         equalization_criterion,
                         excluded_layers=None):
        layer_types = {type(l) for l in model.layers}
        if keras.models.Model in layer_types:
            if layer_types != set([keras.layers.InputLayer, keras.models.Model]):
                raise NotImplementedError("Model encapsulation is only supported if outer model"
                                          "only consists of input layers.")
            model_layer = [l for l in model.layers if (type(l) == keras.models.Model)]
            if len(model_layer) > 1:
                raise NotImplementedError("Model encapsulation is only supported if outer model"
                                          "only includes one inner model")
            return self._equalize_inputs(model_layer[0], filter_counts, granularity,
                                         min_num_filters, equalization_criterion, excluded_layers)
        # Iterating though model layers.
        for layer in model.layers:
            if type(layer) in [
                    keras.layers.Add, keras.layers.Subtract, keras.layers.Multiply,
                    keras.layers.Average, keras.layers.Maximum
            ]:
                eltwise_prunable_inputs = []
                eltwise_prunable_inputs = find_prunable_parent(eltwise_prunable_inputs, layer)
                # Remove broadcast operation layers from mapping
                for l in eltwise_prunable_inputs:
                    if l.filters == 1:
                        eltwise_prunable_inputs.pop(eltwise_prunable_inputs.index(l))

                # Do not update/match filter indices for eltwise layer inputs if they included
                # in exclude layers.
                # if not any(i.name in excluded_layers for i in eltwise_prunable_inputs):
                if len(eltwise_prunable_inputs) > 1:
                    filter_counts = self._match_indices(
                        eltwise_prunable_inputs, filter_counts, min_num_filters, granularity, layer,
                        threshold, equalization_criterion, excluded_layers)
        return filter_counts

    def _match_indices(self, eltwise_prunable_inputs, filter_counts, min_num_filters, granularity,
                       layer, threshold, equalization_criterion, excluded_layers):
        # Compute retainable filters.
        output_depth = eltwise_prunable_inputs[0].filters
        if any(l.name in excluded_layers for l in eltwise_prunable_inputs):
            matched_retained_idx = range(output_depth)
        else:
            cumulative_stat = np.array([])
            for idx, l in enumerate(eltwise_prunable_inputs, 1):
                layerwise_stat = filter_counts[l.name]['norms']
                if not np.asarray(cumulative_stat).size:
                    cumulative_stat = layerwise_stat
                elif equalization_criterion == 'union':
                    cumulative_stat = np.maximum(layerwise_stat, cumulative_stat)
                elif equalization_criterion == 'intersection':
                    cumulative_stat = np.minimum(layerwise_stat, cumulative_stat)
                elif equalization_criterion == "arithmetic_mean":
                    cumulative_stat = (cumulative_stat * (idx - 1) + layerwise_stat) / float(idx)
                elif equalization_criterion == "geometric_mean":
                    cumulative_stat = np.power(
                        np.multiply(np.power(cumulative_stat, idx - 1), layerwise_stat),
                        float(1 / idx))
                else:
                    raise NotImplementedError("Unknown equalization criterion: {}"
                                              .format(equalization_criterion))
            output_idx = np.where(cumulative_stat > threshold)[0]
            num_retained = len(output_idx)
            min_num_filters = min(min_num_filters, output_depth)
            num_retained = max(min_num_filters, num_retained)
            if num_retained % granularity > 0:
                num_retained += granularity - (num_retained % granularity)
            num_retained = min(num_retained, output_depth)
            sorted_idx = np.argsort(-cumulative_stat)
            matched_retained_idx = np.sort(sorted_idx[:num_retained])

        # Set filter counts for updated layers
        for l in eltwise_prunable_inputs:
            filter_counts[l.name]['keep_indices'] = matched_retained_idx
            filter_counts[l.name]['prune_indices'] = np.setdiff1d(matched_retained_idx,
                                                                  range(output_depth))
            filter_counts[l.name]['to_keep'] = len(matched_retained_idx)
            filter_counts[l.name]['to_prune'] = output_depth - len(matched_retained_idx)
            filter_counts[l.name]['total'] = output_depth
        return filter_counts

    def _match_dw_indices(self, parent_layer, layer, filter_counts,
                          min_num_filters, granularity, threshold,
                          equalization_criterion, excluded_layers):
        # Compute retainable filters for DepthwiseConv2D layer.
        dw_layers = [parent_layer, layer]
        output_depth = parent_layer.filters
        if any(l.name in excluded_layers for l in dw_layers):
            matched_retained_idx = range(output_depth)
        else:
            cumulative_stat = np.array([])
            for idx, l in enumerate(dw_layers, 1):
                layerwise_stat = filter_counts[l.name]['norms']
                if not np.asarray(cumulative_stat).size:
                    cumulative_stat = layerwise_stat
                elif equalization_criterion == 'union':
                    cumulative_stat = np.maximum(layerwise_stat, cumulative_stat)
                elif equalization_criterion == 'intersection':
                    cumulative_stat = np.minimum(layerwise_stat, cumulative_stat)
                elif equalization_criterion == "arithmetic_mean":
                    cumulative_stat = (cumulative_stat * (idx - 1) + layerwise_stat) / float(idx)
                elif equalization_criterion == "geometric_mean":
                    cumulative_stat = np.power(np.multiply(np.power(cumulative_stat, idx - 1),
                                                           layerwise_stat), float(1 / idx))
                else:
                    raise NotImplementedError("Unknown equalization criterion: {}"
                                              .format(equalization_criterion))
            output_idx = np.where(cumulative_stat > threshold)[0]
            num_retained = len(output_idx)
            min_num_filters = min(min_num_filters, output_depth)
            num_retained = max(min_num_filters, num_retained)
            if num_retained % granularity > 0:
                num_retained += granularity - (num_retained % granularity)
            num_retained = min(num_retained, output_depth)
            sorted_idx = np.argsort(-cumulative_stat)
            matched_retained_idx = np.sort(sorted_idx[:num_retained])

        # Set filter counts for updated layers
        for l in dw_layers:
            filter_counts[l.name]['keep_indices'] = matched_retained_idx
            filter_counts[l.name]['prune_indices'] = np.setdiff1d(matched_retained_idx,
                                                                  range(output_depth))
            filter_counts[l.name]['to_keep'] = len(matched_retained_idx)
            filter_counts[l.name]['to_prune'] = output_depth - len(matched_retained_idx)
            filter_counts[l.name]['total'] = output_depth
        return filter_counts

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "equalization_criterion, elmtwise_op, method, normalizer, criterion,"
        "granularity, min_num_filters, threshold, dont_prune_elmtwise_input", [
            (HelNet, 6, 'channels_first', False, (3, 128, 64), "union",
             keras.layers.Add, 'min_weight', 'off', 'L2', 2, 8, 0.5, False),
            (ResNet, 10, 'channels_first', True, (3, 128, 64), "union",
             keras.layers.Add, 'min_weight', 'off', 'L2', 2, 8, 0.5, True),
            (HelNet, 6, 'channels_first', False, (3, 128, 64), "geometric_mean",
             keras.layers.Multiply, 'min_weight', 'off', 'L2', 2, 8, 0.5, True),
            (ResNet, 10, 'channels_first', False, (3, 128, 64), "union",
             keras.layers.Subtract, 'min_weight', 'off', 'L2', 2, 8, 0.5, False),
        ])
    def test_broadcast_ops(self, model, nlayers, data_format, use_batch_norm, input_shape,
                           equalization_criterion, elmtwise_op, normalizer, method, criterion,
                           granularity, min_num_filters, threshold, dont_prune_elmtwise_input):
        """Test broadcast element-wise operations."""
        inputs = keras.layers.Input(shape=input_shape)
        if model == ResNet:  # pylint: disable=W0143
            model = model(
                nlayers,
                inputs,
                use_batch_norm=use_batch_norm,
                data_format=data_format,
                all_projections=True)
        else:
            model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Add conv layer.
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='linear',
            name='elmtwise_input_1')(x)

        if elmtwise_op != keras.layers.Subtract:
            x2 = keras.layers.Conv2D(
                filters=1,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding='same',
                data_format=data_format,
                dilation_rate=(1, 1),
                activation='linear',
                name="broadcast_input")(x)

        # Add branch.
        x1 = keras.layers.Conv2D(
            filters=24,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='linear',
            name='conv2d_x1')(x)

        x1 = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='linear',
            name='elmtwise_input_2')(x1)

        # Add skip connection. Broadcast operations are not supported for subtract layers.
        if elmtwise_op != keras.layers.Subtract:
            x = elmtwise_op()([x1, x, x2])
        else:
            x = elmtwise_op()([x1, x])

        # Add extra layer on top.
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='sigmoid',
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        excluded_layers = ['conv2d_output']
        if dont_prune_elmtwise_input:
            excluded_layers.extend(['elmtwise_input_1'])

        if equalization_criterion in ['intersection', 'geometric_mean']:
            # Disable the output tests, as these criteria are not even supposed to work.
            # Technically, geometric_mean might work when the merge is a multiplication,
            # but since the setting is global, it is better not support it.
            check_output_on_input_shape = None
        else:
            check_output_on_input_shape = input_shape

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {
            'excluded_layers': excluded_layers,
            'check_output_on_input_shape': check_output_on_input_shape,
            'equalization_criterion': equalization_criterion
        }
        # Pruning and check for pruned weights.
        self.common(*args, **kwargs)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape, method,"
        "normalizer, criterion, granularity, min_num_filters, threshold,"
        "excluded_layers", [
            (HelNet, 6, 'channels_first', False,
             (3, 256, 256), 'min_weight', 'off', 'L2', 4, 8, 0.5, ['block_4a_conv_1']),
            ('alexnet', None, 'channels_last', True,
             (256, 256, 3), 'min_weight', 'off', 'L2', 4, 16, 0.25, ['head_fc8']),
        ])
    def test_dont_prune_layer(self, model, nlayers, data_format, use_batch_norm, input_shape,
                              normalizer, method, criterion, granularity, min_num_filters,
                              threshold, excluded_layers):
        """Test that we don't prune layers that are explicitly excluded."""
        inputs = keras.layers.Input(shape=input_shape)

        if model == HelNet:  # pylint: disable=W0143
            model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        elif model == 'alexnet':
            model = self.create_alexnet(input_shape, data_format)

        self.common(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            excluded_layers=excluded_layers,
            check_output_on_input_shape=input_shape)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "equalization_criterion, elmtwise_op, method, normalizer, criterion,"
        "granularity, min_num_filters, threshold, dont_prune_elmtwise_input,"
        "enable_qat", [
            (HelNet, 6, 'channels_first', False, (3, 128, 64), "union",
             keras.layers.Add, 'min_weight', 'off', 'L2', 2, 8, 0.5, False, False),
            (HelNet, 6, 'channels_first', True, (3, 128, 64), "intersection",
             keras.layers.Add, 'min_weight', 'off', 'L2', 2, 8, 0.5, True, True),
            (ResNet, 10, 'channels_last', False, (128, 64, 3), "arithmetic_mean",
             keras.layers.Average, 'min_weight', 'off', 'L2', 2, 8, 0.5, False, False),
            (HelNet, 6, 'channels_first', False, (3, 128, 64), "geometric_mean",
             keras.layers.Multiply, 'min_weight', 'off', 'L2', 2, 8, 0.5, True, True),
            (ResNet, 18, 'channels_last', True, (128, 64, 3), "union",
             keras.layers.Subtract, 'min_weight', 'off', 'L2', 2, 8, 0.5, False, False),
        ])
    def test_elmtwise_ops(self, model, nlayers, data_format, use_batch_norm, input_shape,
                          equalization_criterion, elmtwise_op, normalizer, method, criterion,
                          granularity, min_num_filters, threshold, dont_prune_elmtwise_input,
                          enable_qat):
        """Test element-wise operations."""
        if os.getenv("TF_KERAS") is not None and enable_qat:
            pytest.skip("Quantized Conv2D not supported by pruning app for tf.keras yet.")

        inputs = keras.layers.Input(shape=input_shape)
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Add conv layer.
        if enable_qat:
            x = QuantizedConv2D(
                filters=16,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding='same',
                data_format=data_format,
                dilation_rate=(1, 1),
                activation='linear',
                name='elmtwise_input_1')(x)

            # adding QDQ node to compute moving average
            # scale factors.
            x = QDQ(name="elmtwise_input_1_qdq")(x)
        else:
            x = keras.layers.Conv2D(
                filters=16,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding='same',
                data_format=data_format,
                dilation_rate=(1, 1),
                activation='linear',
                name='elmtwise_input_1')(x)

        # Add branch.
        if enable_qat:
            x1 = QuantizedConv2D(
                filters=24,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding='same',
                data_format=data_format,
                dilation_rate=(1, 1),
                activation='linear',
                name='conv2d_x1')(x)

            # adding QDQ node to compute moving average
            # scale factors.
            x1 = QDQ(name="conv2d_x1_qdq")(x1)
        else:
            x1 = keras.layers.Conv2D(
                filters=24,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding='same',
                data_format=data_format,
                dilation_rate=(1, 1),
                activation='linear',
                name='conv2d_x1')(x)

        if enable_qat:
            x1 = QuantizedConv2D(
                filters=16,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding='same',
                data_format=data_format,
                dilation_rate=(1, 1),
                activation='linear',
                name='elmtwise_input_2')(x1)

            # adding QDQ node to compute moving average
            # scale factors.
            x1 = QDQ(name="elmtwise_input_2_qdq")(x1)
        else:
            x1 = keras.layers.Conv2D(
                filters=16,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding='same',
                data_format=data_format,
                dilation_rate=(1, 1),
                activation='linear',
                name='elmtwise_input_2')(x1)

        # Add skip connection.
        x = elmtwise_op()([x1, x])

        # Add extra layer on top.
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='sigmoid',
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        excluded_layers = ['conv2d_output']
        if dont_prune_elmtwise_input:
            excluded_layers.extend(['elmtwise_input_1', 'elmtwise_input_2'])

        if equalization_criterion in ['intersection', 'geometric_mean']:
            # Disable the output tests, as these criteria are not even supposed to work.
            check_output_on_input_shape = None
        else:
            check_output_on_input_shape = input_shape

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {
            'equalization_criterion': equalization_criterion,
            'excluded_layers': excluded_layers,
            'check_output_on_input_shape': check_output_on_input_shape
        }

        self.common(*args, **kwargs)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "equalization_criterion, method, normalizer, criterion,"
        "granularity, min_num_filters, threshold", [
            (HelNet, 6, 'channels_first', False,
             (3, 128, 64), "union", 'min_weight', 'off', 'L2', 1, 8, 0.5),
        ])
    def test_conv_gru_2d(self, model, nlayers, data_format, use_batch_norm, input_shape,
                         equalization_criterion, normalizer, method, criterion,
                         granularity, min_num_filters, threshold):
        """Test convolutional GRU layer."""
        if os.getenv("TF_KERAS") is not None:
            pytest.skip("ConvGRU2D not supported by pruning app for tf.keras yet.")

        inputs = keras.layers.Input(shape=input_shape)
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Add conv layer.
        x = ConvGRU2D(model_sequence_length_in_frames=2,
                      input_sequence_length_in_frames=2,
                      state_scaling=1.0,
                      input_shape=[None, None, 8, 4],
                      initial_state_shape=[None, 64, 8, 4],
                      spatial_kernel_height=3,
                      spatial_kernel_width=3)(x)

        # Add extra layer on top.
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='linear',
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        excluded_layers = ['conv2d_output']

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {
            'equalization_criterion': equalization_criterion,
            'excluded_layers': excluded_layers,
            'check_output_on_input_shape': input_shape
        }

        self.common(*args, **kwargs)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape, method,"
                             "normalizer, criterion, granularity, min_num_filters, threshold", [
                                 (HelNet, 6, 'channels_first', False,
                                  (3, 128, 256), 'min_weight', 'max', 'L2', 8, 16, 1.),
                                 (HelNet, 12, 'channels_last', True,
                                  (256, 256, 3), 'min_weight', 'off', 'L2', 8, 16, 1e3),
                             ])
    def test_min_weight(self, model, nlayers, data_format, use_batch_norm, input_shape, normalizer,
                        method, criterion, granularity, min_num_filters, threshold):
        """Test that we retain min_num_filters.

        This also tests the lower bound on thresholds.
        """
        inputs = keras.layers.Input(shape=input_shape)

        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)

        pruned_model = nvidia_tao_tf1.core.pruning.prune(
            model, method, normalizer, criterion, granularity,
            min_num_filters, threshold
        )

        weights = pruned_model.get_weights()
        assert all([w.shape[-1] == min_num_filters for w in weights])

    @pytest.mark.parametrize("data_format, input_shape, method,"
                             "normalizer, criterion, granularity, min_num_filters, threshold", [
                                 ('channels_first',
                                  (3, 64, 96), 'min_weight', 'off', 'L2', 4, 8, 0.5),
                                 (None, (3, 256, 256), 'min_weight', 'off', 'L2', 4, 8, 0.5),
                             ])
    def test_flatten(self, data_format, input_shape, normalizer, method, criterion, granularity,
                     min_num_filters, threshold):
        """Test that we can prune 'flatten' layer."""
        inputs = keras.layers.Input(shape=input_shape)

        if data_format is not None:
            x = keras.layers.Conv2D(
                filters=32,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding='same',
                data_format=data_format)(inputs)
        else:
            # Test pruning of flatten layer with unknown format (the API will
            # verify that the previous layer was unpruned).
            x = inputs
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(10, activation='linear', name='dense_output')(x)

        model = keras.models.Model(inputs=inputs, outputs=x)

        self.common(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            excluded_layers=['dense_output'],
            check_output_on_input_shape=input_shape)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape, method,"
                             "normalizer, criterion, granularity, min_num_filters, threshold", [
                                 (HelNet, 6, 'channels_first', False,
                                  (3, 256, 256), 'min_weight', 'off', 'L2', 4, 8, 0.5),
                                 (HelNet, 12, 'channels_last', True,
                                  (256, 256, 3), 'min_weight', 'max', 'L2', 8, 16, 0.5),
                                 ('alexnet', None, 'channels_first', True,
                                  (3, 223, 223), 'min_weight', 'max', 'L2', 8, 16, 0.5),
                                 ('alexnet', None, 'channels_last', True,
                                  (256, 256, 3), 'min_weight', 'off', 'L2', 4, 16, 0.25),
                             ])
    def test_granularity(self, model, nlayers, data_format, use_batch_norm, input_shape, normalizer,
                         method, criterion, granularity, min_num_filters, threshold):
        """Test that we prune n*granularity filters."""
        inputs = keras.layers.Input(shape=input_shape)

        if model == HelNet:  # pylint: disable=W0143
            model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        elif model == 'alexnet':
            model = self.create_alexnet(input_shape, data_format)

        batch_shape = (1,) + input_shape
        pruned_model = self.common(model, method, normalizer, criterion, granularity,
                                   min_num_filters, threshold)

        model = keras.models.Model(inputs=inputs, outputs=pruned_model(inputs), name=model.name)
        model.predict(np.zeros(batch_shape))

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape_1,"
                             "input_shape_2, method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold", [
                                 (HelNet, 6, 'channels_first', False, (3, 128, 64),
                                  (3, 64, 32), 'min_weight', 'off', 'L2', 2, 4, 0.5),
                             ])
    def test_mimo(self, model, nlayers, data_format, use_batch_norm, input_shape_1, input_shape_2,
                  normalizer, method, criterion, granularity, min_num_filters, threshold):
        """Test the pruning of models with multiple inputs and multiple outputs."""
        input_1 = keras.layers.Input(shape=input_shape_1)
        model = model(nlayers, input_1, use_batch_norm=use_batch_norm, data_format=data_format)
        x_1 = model.outputs[0]

        input_2 = keras.layers.Input(shape=input_shape_2)
        x_2 = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(8, 8), padding='same',
            data_format=data_format)(input_2)

        inputs = [input_1, input_2]

        # Merge.
        x = keras.layers.Concatenate(axis=1, name='output')([x_1, x_2])

        # Add two branches on top.
        out_1 = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='sigmoid',
            name='output_1')(x)

        out_2 = keras.layers.Conv2D(
            filters=24,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='sigmoid',
            name='output_2')(x)

        # Provide outputs in reverse creation order to verify fix in output ordering.
        outputs = [out_2, out_1]

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=outputs)

        batch_shape_1 = (8,) + input_shape_1
        batch_shape_2 = (8,) + input_shape_2
        batch = [np.zeros(batch_shape_1), np.zeros(batch_shape_2)]
        shapes_before = [out.shape for out in model.predict(batch)]

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {'excluded_layers': ['output_1', 'output_2']}

        pruned_model = self.common(*args, **kwargs)
        pruned_model = keras.models.Model(
            inputs=inputs, outputs=pruned_model(inputs), name=model.name)
        shapes_after = [out.shape for out in pruned_model.predict(batch)]
        assert shapes_before == shapes_after

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape_1,"
                             "input_shape_2, method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold", [
                                 (HelNet, 6, 'channels_first', False, (3, 128, 64),
                                  (3, 64, 32), 'min_weight', 'off', 'L2', 2, 4, 0.5),
                             ])
    def test_multiple_inputs(self, model, nlayers, data_format, use_batch_norm, input_shape_1,
                             input_shape_2, normalizer, method, criterion, granularity,
                             min_num_filters, threshold):
        """Test the pruning of models with multiple inputs."""
        input_1 = keras.layers.Input(shape=input_shape_1)
        model = model(nlayers, input_1, use_batch_norm=use_batch_norm, data_format=data_format)
        out_1 = model.outputs[0]

        input_2 = keras.layers.Input(shape=input_shape_2)
        out_2 = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(8, 8), padding='same',
            data_format=data_format)(input_2)

        # Feed inputs in reverse creation order to verify fix in input order.
        inputs = [input_2, input_1]
        input_shapes = [input_shape_2, input_shape_1]

        # Merge.
        x = keras.layers.Concatenate(axis=1, name='output')([out_1, out_2])

        # Add extra layer on top.
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='tanh',
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {'excluded_layers': ['conv2d_output'], 'check_output_on_input_shape': input_shapes}

        self.common(*args, **kwargs)

    @pytest.mark.parametrize(
        "data_format, input_shape, method,"
        "normalizer, criterion, granularity, min_num_filters, threshold", [
            ('channels_first', (3, 128, 256), 'min_weight', 'max', 'L2', 8, 16, 0.5),
            ('channels_last', (256, 256, 3), 'min_weight', 'off', 'L2', 8, 16, 0.5),
        ])
    def test_no_bias_in_conv_layer(self, data_format, input_shape, normalizer, method, criterion,
                                   granularity, min_num_filters, threshold):
        """Test that we can prune conv layers with no bias terms."""
        inputs = keras.layers.Input(shape=input_shape)

        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='relu',
            use_bias=False,
            kernel_initializer='glorot_uniform',
            name='conv2d_1')(inputs)

        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='linear',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        self.common(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            excluded_layers=['conv2d_output'],
            check_output_on_input_shape=input_shape)

    @pytest.mark.parametrize("data_format, input_shape, method,"
                             "normalizer, criterion, granularity, min_num_filters, threshold", [
                                 ('channels_first',
                                  (3, 128, 256), 'min_weight', 'max', 'L2', 8, 16, 0.5),
                             ])
    def test_no_bias_in_conv_transpose_layer(self, data_format, input_shape, normalizer, method,
                                             criterion, granularity, min_num_filters, threshold):
        """Test that we can prune conv layers with no bias terms."""
        inputs = keras.layers.Input(shape=input_shape)

        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='tanh',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name='conv2d')(inputs)

        x = keras.layers.Conv2DTranspose(
            filters=8,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            data_format=data_format,
            use_bias=False,
            name='conv2d_transpose')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        self.common(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            check_output_on_input_shape=input_shape)

    @pytest.mark.parametrize(
        "input_shape, data_format, method,"
        "normalizer, criterion, granularity, min_num_filters, threshold", [
            ((3, 16, 16), 'channels_first', 'min_weight', 'max', 'L2', 2, 2, 0.5),
        ])
    def test_no_bias_in_dense_layer(self, input_shape, data_format, normalizer, method, criterion,
                                    granularity, min_num_filters, threshold):
        """Test that we can prune dense layers with no bias terms."""
        inputs = keras.layers.Input(shape=input_shape)

        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(2, 2),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name='conv2d_1')(inputs)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(32, activation='relu', use_bias=False)(x)
        x = keras.layers.Dense(16, activation='tanh', use_bias=True)(x)
        x = keras.layers.Dense(10, activation='linear', name='dense_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        self.common(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            excluded_layers=['dense_output'],
            check_output_on_input_shape=input_shape)

    # @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape, method,"
    #                          "normalizer, criterion, granularity, min_num_filters, threshold", [
    #                              (HelNet, 6, 'channels_first', False,
    #                               (3, 256, 256), 'min_weight', 'off', 'L2', 4, 8, 0.5),
    #                              (HelNet, 6, 'channels_last', True,
    #                               (256, 256, 3), 'min_weight', 'max', 'L2', 8, 16, 0.5),
    #                          ])
    # def test_recurse(self, model, nlayers, data_format, use_batch_norm, input_shape, normalizer,
    #                  method, criterion, granularity, min_num_filters, threshold):
    #     """Test that we recurse through 'Model' layers."""
    #     inner_inputs = keras.layers.Input(shape=input_shape)
    #     outer_inputs = keras.layers.Input(shape=input_shape)
    #     inner_model = model(
    #         nlayers, inner_inputs, use_batch_norm=use_batch_norm, data_format=data_format)
    #     outer_model = keras.models.Model(inputs=outer_inputs, outputs=inner_model(outer_inputs))
    #     self.common(outer_model, method, normalizer, criterion, granularity, min_num_filters,
    #                 threshold)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "method, normalizer, criterion, granularity, min_num_filters,"
        "threshold, equalization_criterion, all_projections",
        [(ResNet, 18, 'channels_first', False,
          (3, 960, 544), 'min_weight', 'off', 'L2', 2, 8, 0.5, "union", False),
         (ResNet, 18, 'channels_first', False,
          (3, 960, 544), 'min_weight', 'off', 'L2', 2, 8, 0.5, "union", True),
         (ResNet, 10, 'channels_first', True,
          (3, 960, 544), 'min_weight', 'off', 'L2', 2, 8, 0.5, "arithmetic_mean", True),
         (ResNet, 10, 'channels_first', True,
          (3, 960, 544), 'min_weight', 'off', 'L2', 2, 8, 0.5, "union", False)])
    def test_resnets(self, model, nlayers, data_format, use_batch_norm, input_shape, normalizer,
                     method, criterion, granularity, min_num_filters, threshold,
                     equalization_criterion, all_projections):
        """Test partial pruning for MSRA resnet templates."""
        # Set up Resnet model.
        inputs = keras.layers.Input(shape=input_shape)
        model = model(
            nlayers,
            inputs,
            use_batch_norm=use_batch_norm,
            data_format=data_format,
            all_projections=all_projections)
        x = model.outputs[0]

        # Hooking up to fully connected layer for 10 classes.
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(256, name='inner_fc', activation='relu')(x)
        x = keras.layers.Dense(10, name='output_fc', activation='relu')(x)

        # Setting up a model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        # Define elementwise input layers alone as exclude layers.
        excluded_layers = ['output_fc']

        # Prune model and check weights.
        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {
            'equalization_criterion': equalization_criterion,
            'excluded_layers': excluded_layers,
            'check_output_on_input_shape': input_shape
        }

        self.common(*args, **kwargs)

    @pytest.mark.parametrize("data_format, input_shape,"
                             "method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold", [
                                 ('channels_first',
                                  (3, 128, 64), 'min_weight', 'off', 'L2', 2, 4, 0.5),
                             ])
    def test_shared_layer(self, data_format, input_shape, normalizer, method, criterion,
                          granularity, min_num_filters, threshold):
        """Test the pruning of models with shared layers."""
        input_1 = keras.layers.Input(shape=input_shape)
        input_2 = keras.layers.Input(shape=input_shape)
        input_3 = keras.layers.Input(shape=input_shape)
        inputs = [input_1, input_2, input_3]

        # This layer will be applied to three different inputs.
        conv_layer = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(2, 2), padding='same', data_format=data_format)

        conv_layer_output_1 = conv_layer(input_1)
        conv_layer_output_2 = conv_layer(input_2)
        conv_layer_output_3 = conv_layer(input_3)

        # Merge.
        x = keras.layers.Concatenate(
            axis=1, name='concat')([conv_layer_output_1, conv_layer_output_2, conv_layer_output_3])
        x = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(8, 8), padding='same',
            data_format=data_format)(x)

        # Add named output layer.
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        input_shapes = [input_shape, input_shape, input_shape]

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {'excluded_layers': ['conv2d_output'], 'check_output_on_input_shape': input_shapes}

        self.common(*args, **kwargs)

    @pytest.mark.parametrize("data_format, input_shape,"
                             "method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold", [
                                 ('channels_first',
                                  (3, 128, 64), 'min_weight', 'off', 'L2', 2, 4, 0.5),
                             ])
    def test_shared_layer2(self, data_format, input_shape, normalizer, method, criterion,
                           granularity, min_num_filters, threshold):
        """Test the pruning of models with shared layers."""
        input_1 = keras.layers.Input(shape=input_shape)
        input_2 = keras.layers.Input(shape=input_shape)
        input_3 = keras.layers.Input(shape=input_shape)
        inputs = [input_1, input_2, input_3]

        # This layer will be applied to three different inputs.
        c1 = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(1, 1), padding='same', data_format=data_format)
        # This layer will be applied to three different inputs.
        c2 = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(1, 1), padding='same', data_format=data_format)
        # This layer will be applied to three different inputs.
        c3 = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(1, 1), padding='same', data_format=data_format)
        # This layer will be applied to three different inputs.
        conv_layer = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(2, 2), padding='same', data_format=data_format)

        conv_layer_output_1 = conv_layer(c1(input_1))
        conv_layer_output_2 = conv_layer(c2(input_2))
        conv_layer_output_3 = conv_layer(c3(input_3))

        # Merge.
        x = keras.layers.Concatenate(
            axis=1, name='concat')([conv_layer_output_1, conv_layer_output_2, conv_layer_output_3])
        x = keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1], strides=(8, 8), padding='same',
            data_format=data_format)(x)

        # Add named output layer.
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {'excluded_layers': ['conv2d_output']}

        self.common(*args, **kwargs)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape, method,"
                             "normalizer, criterion, granularity, min_num_filters, threshold", [
                                 (HelNet, 6, 'channels_first', False,
                                  (3, 256, 256), 'toto', 'off', 'L2', 4, 8, 0.5),
                                 (HelNet, 6, 'channels_first', True,
                                  (3, 256, 256), 'min_weight', 'toto', 'L2', 4, 8, 0.5),
                                 (HelNet, 6, 'channels_first', False,
                                  (3, 256, 256), 'min_weight', 'max', 'toto', 4, 8, 0.5),
                             ])
    def test_unknown_params(self, model, nlayers, data_format, use_batch_norm, input_shape,
                            normalizer, method, criterion, granularity, min_num_filters, threshold):
        """Test that we prune n*granularity filters."""
        inputs = keras.layers.Input(shape=input_shape)

        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)

        with pytest.raises(NotImplementedError):
            nvidia_tao_tf1.core.pruning.prune(
                model, method, normalizer, criterion, granularity,
                min_num_filters, threshold
            )

    def test_unsupported_layer(self):
        """Test that we drop an error on an unsupported layer."""
        inputs = keras.layers.Input(shape=(3, 8, 4, 2))
        # 3D layers are not currently supported.
        x = keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format="channels_first")(
            inputs
        )
        model = keras.models.Model(inputs, x)
        with pytest.raises(NotImplementedError):
            nvidia_tao_tf1.core.pruning.prune(
                model, "min_weight",
                "off", "L2", 8, 1,
                0.01
            )

    # def test_unsupported_recurse(self):
    #     """Test that the API fails 'gracefully' on unsupported types of recursion."""
    #     # Create the inner model.
    #     inputs = keras.layers.Input((1,))
    #     x = keras.layers.core.Dense(1, activation='linear')(inputs)
    #     outputs = keras.layers.core.Dense(2, activation='softmax')(x)
    #     inner_model = keras.models.Model(inputs, outputs)

    #     # Create an outer model with a layer before the inner model.
    #     inputs = keras.layers.Input((1,))
    #     x = keras.layers.core.Dense(1, activation='linear')(inputs)
    #     outputs = inner_model(x)
    #     outer_model_layer_before = keras.models.Model(inputs, outputs)
    #     with pytest.raises(NotImplementedError):
    #         nvidia_tao_tf1.core.pruning.prune(
    #           outer_model_layer_before,
    #           'min_weight',
    #           'off', 'L2', 8,
    #           1, 0.01)

    #     # Create an outer model with a layer on top of the inner model.
    #     inputs = keras.layers.Input((1,))
    #     x = inner_model(inputs)
    #     outputs = keras.layers.core.Dense(2, activation='softmax')(x)
    #     outer_model_layer_after = keras.models.Model(inputs, outputs)
    #     with pytest.raises(NotImplementedError):
    #         nvidia_tao_tf1.core.pruning.prune(
    #             outer_model_layer_after,
    #             'min_weight', 'off',
    #             'L2', 8, 1, 0.01)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape,"
                             "num_classes, method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold",
                             [(HelNet, 6, 'channels_first', False,
                               (3, 128, 64), 4, 'min_weight', 'off', 'L2', 2, 8, 0.5),
                              (HelNet, 6, 'channels_first', False,
                               (3, 128, 64), 4, 'min_weight', 'off', 'L2', 4, 8, 0.5)])
    def test_with_averagepooling2d(self, model, nlayers, data_format, use_batch_norm, input_shape,
                                   num_classes, method, normalizer, criterion, granularity,
                                   min_num_filters, threshold):
        """Test with AveragePooling2D."""
        inputs = keras.layers.Input(shape=input_shape)

        # Defining the model defined in the test case.
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Adding AveragePooling2D node.
        x = keras.layers.AveragePooling2D(
            pool_size=(2, 2), data_format=data_format, padding='same')(x)

        # Adding a dense head of num classes.
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(num_classes, name='output_fc', activation='relu')(x)
        model = keras.models.Model(inputs=inputs, outputs=x)

        # Exclude final fc layer from pruning.
        excluded_layers = ['output_fc']

        # Prune model and check weights.
        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {'excluded_layers': excluded_layers, 'check_output_on_input_shape': input_shape}
        self.common(*args, **kwargs)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape,"
                             "num_classes, method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold, exclude_permute_inputs",
                             [(HelNet, 6, 'channels_first', False,
                               (3, 128, 64), 4, 'min_weight', 'off', 'L2', 2, 8, 0.5, False),
                              (HelNet, 6, 'channels_first', False,
                               (3, 128, 64), 4, 'min_weight', 'off', 'L2', 4, 8, 0.5, True)])
    def test_with_permute_layer(self, model, nlayers, data_format, use_batch_norm, input_shape,
                                num_classes, method, normalizer, criterion, granularity,
                                min_num_filters, threshold, exclude_permute_inputs):
        """Test with Permute layer."""
        inputs = keras.layers.Input(shape=input_shape)

        # Defining the model defined in the test case.
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Adding Permute Node.
        x = keras.layers.Permute((1, 3, 2))(x)

        # Adding a dense head of num classes.
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(num_classes, name='output_fc', activation='relu')(x)
        model = keras.models.Model(inputs=inputs, outputs=x)
        model.summary()

        # Exclude final fc layer from pruning.
        excluded_layers = ['output_fc']
        if exclude_permute_inputs:
            excluded_layers.append("block_4a_conv_1")

        # Prune model and check weights.
        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {'excluded_layers': excluded_layers, 'check_output_on_input_shape': input_shape}

        # Catch error if permute inputs are not excluded.
        if not exclude_permute_inputs:
            with pytest.raises(NotImplementedError):
                self.common(*args, **kwargs)
        else:
            self.common(*args, **kwargs)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "num_classes, method, normalizer, criterion, granularity,"
        "min_num_filters, threshold, zeropadding_dims",
        [(HelNet, 6, 'channels_first', False,
          (3, 128, 64), 4, 'min_weight', 'off', 'L2', 2, 8, 0.5, 3),
         (HelNet, 6, 'channels_first', False, (3, 128, 64), 4, 'min_weight', 'off', 'L2', 4, 8, 0.5,
          (3, 2)), (HelNet, 6, 'channels_last', False,
                    (128, 64, 3), 4, 'min_weight', 'off', 'L2', 4, 8, 0.5, ((3, 2), (3, 2)))])
    def test_with_zeropadding2D_layer(self, model, nlayers, data_format, use_batch_norm,
                                      input_shape, num_classes, method, normalizer, criterion,
                                      granularity, min_num_filters, threshold, zeropadding_dims):
        """Test with ZeroPadding2D."""
        inputs = keras.layers.Input(shape=input_shape)

        # Defining the model defined in the test case.
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Adding ZeroPadding2D Node.
        x = keras.layers.ZeroPadding2D(
            padding=zeropadding_dims, data_format=data_format)(x)

        # Adding a dense head of num classes.
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(num_classes, name='output_fc', activation='relu')(x)
        model = keras.models.Model(inputs=inputs, outputs=x)

        # Exclude final fc layer from pruning.
        excluded_layers = ['output_fc']

        # Prune model and check weights.
        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {'excluded_layers': excluded_layers, 'check_output_on_input_shape': input_shape}
        self.common(*args, **kwargs)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape, method,"
                             "normalizer, criterion, granularity, min_num_filters, threshold", [
                                 (HelNet, 6, 'channels_first', False,
                                  (3, 256, 256), 'min_weight', 'off', 'L2', 4, 8, 0.5),
                                 (HelNet, 6, 'channels_last', True,
                                  (256, 256, 3), 'min_weight', 'max', 'L2', 8, 16, 0.5),
                             ])
    def test_with_conv_transpose_head(self, model, nlayers, data_format, use_batch_norm,
                                      input_shape, normalizer, method, criterion, granularity,
                                      min_num_filters, threshold):
        """Test that we prune n*granularity filters."""
        inputs = keras.layers.Input(shape=input_shape)
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)

        model = add_deconv_head(
            model=model,
            inputs=inputs,
            nmaps=1,
            upsampling=2,
            data_format=data_format,
            activation_type='relu')

        self.common(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            check_output_on_input_shape=input_shape)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape, method,"
        "normalizer, criterion, granularity, min_num_filters, threshold,"
        "excluded_layers",
        [(HelNet, 6, 'channels_first', False,
          (3, 256, 256), 'min_weight', 'off', 'L2', 4, 8, 0.5, ['block_4a_conv_1'])])
    def test_reprune(self, model, nlayers, data_format, use_batch_norm, input_shape, normalizer,
                     method, criterion, granularity, min_num_filters, threshold, excluded_layers):
        """Test that we can reprune a model.

        Args:
            model: the model template to use.
            nlayers (int): number of layers to build template of.
            data_format (str): one of 'channels_first' or 'channels_last'.
            use_batch_norm (bool): whether to use batchnorm.
            input_shape (tuple of ints): input shape.
            method (str): pruning method.
            normalizer (str): type of normalizer to use when pruning.
            criterion (str): type of criterion to use when pruning.
            granularity (int): granularity by which to prune filters.
            min_num_filters (int): min number of filters to retain when pruning.
            threshold (float): pruning threshold.
            excluded_layers (list): list of layers to exclude when pruning."""
        inputs = keras.layers.Input(shape=input_shape)
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)

        pruned_model = self.common(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            excluded_layers=excluded_layers,
            check_output_on_input_shape=input_shape)

        # Apply pruned model to our inputs. When cloning a model, Keras internally
        # recreates all layers - this is different from applying the model to
        # another input, which creates another model but does not create new layers
        # (thus the layers are shared between models, which means layers have multiple
        # outbound nodes, making forward parsing ill-defined).
        # Below we are cloning the model and instructing Keras to use placeholders
        # for the new inputs (if we provide the same input layer as in the original
        # model, Keras will - wrongly? - re-create a new layer with the same name and
        # complain that two layers of the model have the same name!).
        pruned_model_applied = keras.models.clone_model(pruned_model)
        pruned_model_applied.set_weights(pruned_model.get_weights())

        # Note: at this stage a typical workflow would fine-tune the pruned model.

        # Now prune the model again and verify the output shape.
        self.common(
            pruned_model_applied,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            excluded_layers=excluded_layers,
            check_output_on_input_shape=input_shape)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape,"
                             "num_classes, method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold, concat_axis",
                             [(HelNet, 6, 'channels_first', False,
                               (3, 128, 64), 4, 'min_weight', 'off', 'L2', 2, 8, 0.5, 1),
                              (HelNet, 6, 'channels_first', False,
                               (3, 128, 64), 4, 'min_weight', 'off', 'L2', 4, 8, 0.5, 2)])
    def test_with_branches(self, model, nlayers, data_format, use_batch_norm, input_shape,
                           num_classes, normalizer, method, criterion, granularity, min_num_filters,
                           threshold, concat_axis):
        """Test concatenation head."""
        inputs = keras.layers.Input(shape=input_shape)
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Add 1st branch.
        cov_channels = 1
        x1 = keras.layers.Conv2D(
            filters=num_classes * cov_channels,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='linear',
            name='conv2d_x1')(x)

        # Add 2nd branch.
        bbox_channels = 4 if concat_axis == 1 else 1
        x2 = keras.layers.Conv2D(
            filters=num_classes * bbox_channels,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='relu',
            name='conv2d_x2')(x)

        # Merge.
        x = keras.layers.Concatenate(axis=concat_axis, name='output')([x1, x2])

        # Add extra layer on top.
        x = keras.layers.Conv2D(
            filters=num_classes * (bbox_channels + cov_channels),
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='sigmoid',
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {
            'excluded_layers': ['conv2d_x1', 'conv2d_x2', 'conv2d_output'],
            'check_output_on_input_shape': input_shape
        }

        if concat_axis == 1:  # Only channels_first is supported by this test.
            self.common(*args, **kwargs)
        else:
            with pytest.raises(ValueError):
                self.common(*args, **kwargs)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape,"
                             "num_classes, method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold", [
                                 (HelNet, 6, 'channels_first', False,
                                  (3, 128, 64), 4, 'min_weight', 'off', 'L2', 2, 8, 0.5),
                             ])
    def test_with_concat_before_reshape(self, model, nlayers, data_format, use_batch_norm,
                                        input_shape, num_classes, normalizer, method, criterion,
                                        granularity, min_num_filters, threshold):
        """Test pruning in presence of concat layer following a reshape."""
        inputs = keras.layers.Input(shape=input_shape)
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Add 1st branch.
        cov_channels = 1
        x1 = keras.layers.Conv2D(
            filters=num_classes * cov_channels,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='linear',
            name='conv2d_x1')(x)
        x1 = keras.layers.Reshape((num_classes, cov_channels, int(x.shape[-2]),
                                   int(x.shape[-1])))(x1)

        # Add 2nd branch.
        bbox_channels = 4
        x2 = keras.layers.Conv2D(
            filters=num_classes * bbox_channels,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='relu',
            name='conv2d_x2')(x)
        x2 = keras.layers.Reshape((num_classes, bbox_channels, int(x.shape[-2]),
                                   int(x.shape[-1])))(x2)

        # Merge.
        x = keras.layers.Concatenate(axis=2, name='output')([x1, x2])

        x = keras.layers.Reshape((num_classes * (bbox_channels + cov_channels), int(x.shape[-2]),
                                  int(x.shape[-1])))(x)

        x = keras.layers.Conv2D(
            filters=8,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='relu',
            name='conv2d_output')(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        kwargs = {
            'excluded_layers': ['conv2d_x1', 'conv2d_x2', 'conv2d_output'],
            'check_output_on_input_shape': input_shape
        }

        self.common(*args, **kwargs)

    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape,"
                             "num_classes, method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold, prune_before_reshape", [
                                 (HelNet, 6, 'channels_first', False,
                                  (3, 128, 64), 8, 'min_weight', 'off', 'L2', 2, 2, 0.5, True),
                                 (HelNet, 6, 'channels_first', False,
                                  (3, 128, 64), 8, 'min_weight', 'off', 'L2', 2, 2, 0.5, False),
                             ])
    def test_with_reshape(self, model, nlayers, data_format, use_batch_norm, input_shape,
                          num_classes, normalizer, method, criterion, granularity, min_num_filters,
                          threshold, prune_before_reshape):
        """Test pruning of reshape layer."""
        inputs = keras.layers.Input(shape=input_shape)
        model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
        x = model.outputs[0]

        # Add conv layer
        cov_channels = 2
        x = keras.layers.Conv2D(
            filters=num_classes * cov_channels,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='linear',
            name='conv2d_x')(x)

        # Add reshape.
        x = keras.layers.Reshape((num_classes, cov_channels, int(x.shape[-2]), int(x.shape[-1])))(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)
        model.summary()
        args = [model, method, normalizer, criterion, granularity, min_num_filters, threshold]
        if not prune_before_reshape:
            kwargs = {'excluded_layers': ['conv2d_x'], 'check_output_on_input_shape': input_shape}
            self.common(*args, **kwargs)
        else:
            with pytest.raises(NotImplementedError):
                self.common(*args)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, input_shape,"
        "method, normalizer, criterion, granularity,"
        "min_num_filters, threshold ", [
            (HelNet, 6, 'channels_first', (3, 128, 64), 'min_weight', 'off', 'L2', 2, 2, 0.5),
            (HelNet, 6, 'channels_last', (64, 64, 3), 'min_weight', 'off', 'L2', 2, 2, 0.5),
        ])
    def test_with_softmax(self, model, nlayers, data_format, input_shape, normalizer, method,
                          criterion, granularity, min_num_filters, threshold):
        """Test pruning in presence of softmax layer."""
        inputs = keras.layers.Input(shape=input_shape)
        model = model(nlayers, inputs, data_format=data_format)
        x = model.outputs[0]

        x = keras.layers.Conv2D(
            filters=8,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            dilation_rate=(1, 1),
            activation='sigmoid',
            name='conv2d_output')(x)

        # Add softmax layer
        if data_format == 'channels_first':
            softmax_axis = 1
        elif data_format == 'channels_last':
            softmax_axis = -1
        else:
            raise ValueError("Unknown data format: %s" % data_format)
        x = keras.layers.Softmax(axis=softmax_axis)(x)

        # Create model.
        model = keras.models.Model(inputs=inputs, outputs=x)

        # Prune and check activations.
        self.common(
            model,
            method,
            normalizer,
            criterion,
            granularity,
            min_num_filters,
            threshold,
            excluded_layers='conv2d_output',
            check_output_on_input_shape=input_shape)

    @pytest.mark.parametrize("data_format, input_shape,"
                             " method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold ", [
                                 ('channels_first', (3, 128, 64),
                                  'min_weight', 'off', 'L2', 2, 2, 0.5),
                                 ('channels_last', (64, 64, 3),
                                  'min_weight', 'off', 'L2', 2, 2, 0.5),
                             ])
    def test_with_depthwise_conv_layer(self, data_format, input_shape, normalizer, method,
                                       criterion, granularity, min_num_filters, threshold):
        """Test pruning in presence of DepthwiseConv2D layer."""
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Conv2D(32,
                                kernel_size=3,
                                strides=(1, 1),
                                padding='valid',
                                name='conv1',
                                data_format=data_format)(inputs)
        x = keras.layers.DepthwiseConv2D((3, 3),
                                         padding='valid',
                                         strides=1,
                                         depth_multiplier=1,
                                         name='conv_dw_1',
                                         data_format=data_format)(x)
        model = keras.models.Model(inputs=inputs, outputs=x)

        self.common(model, method, normalizer, criterion,
                    granularity, min_num_filters, threshold)

    @pytest.mark.parametrize("data_format, input_shape,"
                             " method, normalizer, criterion, granularity,"
                             "min_num_filters, threshold, depth_multiplier ", [
                                 ('channels_first', (3, 128, 64),
                                  'min_weight', 'off', 'L2', 2, 2, 0.5, 2),
                                 ('channels_last', (64, 64, 3),
                                  'min_weight', 'off', 'L2', 2, 2, 0.5, 3),
                             ])
    def test_depth_multiplier_not_one(self, data_format, input_shape, normalizer, method,
                                      criterion, granularity, min_num_filters, threshold,
                                      depth_multiplier):
        """Test pruning in presence of DepthwiseConv2D layer."""
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Conv2D(32,
                                kernel_size=3,
                                strides=(1, 1),
                                padding='valid',
                                name='conv1',
                                data_format=data_format)(inputs)
        x = keras.layers.DepthwiseConv2D((3, 3),
                                         padding='valid',
                                         strides=1,
                                         depth_multiplier=depth_multiplier,
                                         name='conv_dw_1',
                                         data_format=data_format)(x)
        model = keras.models.Model(inputs=inputs, outputs=x)
        # Will raise ValueError during explore stage when depth_multiplier is not 1.
        with pytest.raises(ValueError):
            self.common(model, method, normalizer, criterion,
                        granularity, min_num_filters, threshold)

    def test_overrides(self):
        """Test that layer config overrides work."""
        input_shape = (10,)
        inputs = keras.layers.Input(shape=input_shape)

        x = keras.layers.Dense(3, activation='linear', name='dense_output')(inputs)

        model = keras.models.Model(inputs=inputs, outputs=x)

        layer_config_overrides = {
            'bias_regularizer': keras.regularizers.l1(0.01),
            'kernel_regularizer': keras.regularizers.l1(0.01),
            'trainable': False
        }

        pruned_model = self.common(
            model,
            'min_weight',
            'off',
            'L2',
            4,
            8,
            0.5,
            excluded_layers=['dense_output'],
            check_output_on_input_shape=input_shape,
            layer_config_overrides=layer_config_overrides)

        # Verify that the overrides got applied.
        for layer in pruned_model.layers:
            # Overrides don't apply to input layers.
            if isinstance(layer, keras.layers.InputLayer):
                continue
            for key in layer_config_overrides:
                assert getattr(layer, key) == layer_config_overrides[key]

    # @pytest.mark.parametrize(
    #     "output_fn, model, nlayers, data_format, use_batch_norm, input_shape,"
    #     "method, normalizer, criterion, granularity, min_num_filters, threshold", [
    #         ('default', HelNet, 6, 'channels_first', False,
    #          (3, 256, 256), 'min_weight', 'off', 'L2', 4, 8, 0.5),
    #         ('toto.pruned', HelNet, 6, 'channels_last', True,
    #          (256, 256, 3), 'min_weight', 'max', 'L2', 8, 16, 0.5),
    #     ])
    # @pytest.mark.script_launch_mode('subprocess')
    # def test_prune_app(self, script_runner, tmpdir, output_fn, model, nlayers, data_format,
    #                    use_batch_norm, input_shape, normalizer, method, criterion, granularity,
    #                    min_num_filters, threshold):
    #     """Test the prune application.

    #     Just make sure a pruned model file is generated.
    #     """
    #     model_filename = os.path.join(str(tmpdir), 'model.h5')

    #     if output_fn == 'default':
    #         extra_args = []
    #         suffix = '.pruned'
    #         pruned_model_filename = os.path.join(str(tmpdir), 'model.h5' + suffix)
    #     else:
    #         pruned_model_filename = os.path.join(str(tmpdir), output_fn)
    #         extra_args = ['--output', pruned_model_filename]

    #     inputs = keras.layers.Input(shape=input_shape)
    #     model = model(nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format)
    #     model.save(model_filename)

    #     env = os.environ.copy()
    #     env['CUDA_VISIBLE_DEVICES'] = ''
    #     script = 'app.py'
    #     # Path adjustment for bazel tests
    #     root_dir = os.path.dirname(os.path.abspath(__file__))
    #     if os.path.exists(os.path.join(root_dir, script)):
    #         script = os.path.join(root_dir, script)
    #     ret = script_runner.run(script, model_filename, env=env, *extra_args)

    #     assert ret.success, "Process returned error: %s error trace: %s" % (ret.success, ret.stderr)
    #     assert os.path.isfile(pruned_model_filename)
