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

"""MaskRCNN custom layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras


class BoxInput(keras.layers.InputLayer):
    '''A custom Keras layer for bbox label input.'''

    def __init__(self,
                 input_shape=None,
                 batch_size=None,
                 dtype=None,
                 input_tensor=None,
                 sparse=False,
                 name="box_input",
                 ragged=False,
                 **kwargs):
        '''Init function.'''
        super(BoxInput, self).__init__(input_shape=input_shape,
                                       batch_size=batch_size,
                                       dtype=dtype,
                                       input_tensor=input_tensor,
                                       sparse=sparse,
                                       name=name,
                                       ragged=ragged,
                                       **kwargs)
