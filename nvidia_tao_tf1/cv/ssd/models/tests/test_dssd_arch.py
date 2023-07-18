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
"""test dssd arch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import numpy as np
import pytest
import tensorflow as tf
from nvidia_tao_tf1.cv.ssd.models.dssd_arch import attach_pred_layers, generate_dssd_layers
from nvidia_tao_tf1.cv.ssd.models.patch_keras import patch as ssd_keras_patch

# Patching SSD Keras.
ssd_keras_patch()


def build_dummy_model(tensor_shape):
    data_format = 'channels_first' if tensor_shape[0] == 3 else 'channels_last'
    layers = [keras.layers.Input(shape=tensor_shape)]
    for _ in range(5):
        new_layer = keras.layers.Conv2D(512,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        data_format=data_format,
                                        use_bias=False)(layers[-1])
        layers.append(new_layer)
    return layers, data_format


def test_dssd_layers_shape():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    test_input_shape = [(3, 512, 512), (512, 51, 3), (1, 3, 3), (5, 3, 3), (64, 51, 3), (3, 11, 64)]
    for idx, input_shape in enumerate(test_input_shape):
        layers, data_format = build_dummy_model(input_shape)
        new_layers = generate_dssd_layers(layers, data_format)
        model = keras.models.Model(inputs=layers[0], outputs=new_layers)
        pred = model.predict(np.random.rand(3, *input_shape))
        for i in range(6):
            assert pred[i][0].shape == layers[i].shape[1:]

        if idx < 3:
            continue
        # test pred module
        no_pred_layers = attach_pred_layers(new_layers, 0, data_format)
        no_pred_model = keras.models.Model(inputs=layers[0], outputs=no_pred_layers)

        # same model
        assert no_pred_model.count_params() == model.count_params()
        pred = no_pred_model.predict(np.random.rand(3, *input_shape))
        for i in range(6):
            assert pred[i][0].shape == layers[i].shape[1:]

        for pred_channel in [1, 8, 9999]:
            with pytest.raises(AssertionError):
                attach_pred_layers(new_layers, pred_channel, data_format)

        for pred_channel in [256, 512, 1024]:
            pred_layers = attach_pred_layers(new_layers, pred_channel, data_format)
            pred_model = keras.models.Model(inputs=layers[0], outputs=pred_layers)
            pred = pred_model.predict(np.random.rand(3, *input_shape))
            for i in range(6):
                if data_format == 'channels_first':
                    assert pred[i][0].shape[1:] == layers[i].shape[2:]
                    assert pred[i][0].shape[0] == pred_channel
                else:
                    assert pred[i][0].shape[:-1] == layers[i].shape[1:-1]
                    assert pred[i][0].shape[-1] == pred_channel
