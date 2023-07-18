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
"""test wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.ssd.models.base_model import get_base_model
from nvidia_tao_tf1.cv.ssd.models.patch_keras import patch as ssd_keras_patch

# Patching SSD Keras.
ssd_keras_patch()


def test_base_model():
    '''
    Test wrapper.

    No need for exaustive test as it's done in test_ssd_backbones.
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    arch_configs = [[(3, 33, 128), 'resnet', 18, True, 256],
                    [(3, 33, 128), 'resnet', 18, False, 0],
                    [(3, 33, 128), 'darknet', 53, True, 1024],
                    [(3, 33, 128), 'resnet', 10, True, 256],
                    [(3, 33, 128), 'mobilenet_v2', 18, True, 256],
                    [(3, 33, 128), 'resnet', 50, True, 256],
                    [(3, 33, 128), 'resnet', 101, True, 1024]]

    for arch_config in arch_configs:
        input_tensor = keras.layers.Input(shape=arch_config[0])
        ssd_dssd = get_base_model(input_tensor, *arch_config[1:])
        pred_model = keras.models.Model(inputs=input_tensor, outputs=ssd_dssd)
        pred_model.predict(np.random.rand(1, *arch_config[0]))
