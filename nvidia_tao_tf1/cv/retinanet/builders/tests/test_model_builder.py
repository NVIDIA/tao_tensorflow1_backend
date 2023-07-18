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

"""Test model builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from nvidia_tao_tf1.cv.retinanet.builders import model_builder
from nvidia_tao_tf1.cv.retinanet.utils.spec_loader import load_experiment_spec


def test_model_builder():
    K.set_learning_phase(0)
    experiment_spec = load_experiment_spec(merge_from_default=True)
    cls_mapping = experiment_spec.dataset_config.target_class_mapping
    classes = sorted({str(x) for x in cls_mapping.values()})
    model_train, _ = model_builder.build(experiment_spec, len(classes) + 1, input_tensor=None)
    assert model_train.get_layer('retinanet_predictions').output_shape[-2:] == (49104, 16)
