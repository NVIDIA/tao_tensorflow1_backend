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

'''build model for training.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _load_pretrain_weights(pretrain_model, train_model):
    """Load weights in pretrain model to model."""
    strict_mode = True
    for layer in train_model.layers[1:]:
        # The layer must match up to yolo layers.
        if layer.name.find('yolo_') != -1:
            strict_mode = False
        try:
            l_return = pretrain_model.get_layer(layer.name)
        except ValueError:
            if strict_mode and layer.name[-3:] != 'qdq' and len(layer.get_weights()) != 0:
                raise ValueError(layer.name + ' not found in pretrained model.')
            # Ignore QDQ
            continue
        try:
            layer.set_weights(l_return.get_weights())
        except ValueError:
            if strict_mode:
                raise ValueError(layer.name + ' has incorrect shape in pretrained model.')
            continue
