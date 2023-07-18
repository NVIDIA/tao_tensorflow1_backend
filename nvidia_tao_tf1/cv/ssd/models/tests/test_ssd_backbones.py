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
"""test ssd base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.ssd.models.patch_keras import patch as ssd_keras_patch
from nvidia_tao_tf1.cv.ssd.models.ssd_backbones import get_raw_ssd_model

# Patching SSD Keras.
ssd_keras_patch()


def test_ssd_arch():
    '''To make sure all the SSD model works.'''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    input_shape = (3, 33, 32)
    archs = [('resnet', x) for x in [10, 18, 34, 50, 101]] + [('darknet', 19), ('darknet', 53)] + \
        [('vgg', 19), ('vgg', 16)] + [('squeezenet', 0), ('googlenet', 0)] + \
        [('mobilenet_v1', 0), ('mobilenet_v2', 0)]
    input_tensor = keras.layers.Input(shape=input_shape)
    for arch, nlayers in archs:
        out_layers = get_raw_ssd_model(input_tensor, arch, nlayers)
        pred_model = keras.models.Model(inputs=input_tensor, outputs=out_layers)
        pred_model.predict(np.random.rand(1, *input_shape))
