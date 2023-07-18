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
"""TFReshape layer in FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.layers import Layer


class TFReshape(Layer):
    '''New Reshape Layer to mimic TF reshape op.'''

    def __init__(self, target_shape, **kwargs):
        """Init function.

        Args:
            target_shape(tuple): the target shape of this layer.
        """
        assert not (None in target_shape), 'Target shape should be all defined.'
        minus_one_num = sum(s for s in target_shape if s == -1)
        if minus_one_num > 1:
            raise ValueError('Can have at most one -1 in target shape.')
        self.target_shape = list(target_shape)
        super(TFReshape, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """compute_output_shape.

        Args:
            input_shape(tuple): the shape of the input tensor.
        Returns:
            the target shape.
        """
        return tuple(self.target_shape)

    def call(self, x, mask=None):
        """call.

        Args:
            x(list): the list of input tensors.
        Returns:
            The output tensor with the target shape.
        """
        return K.reshape(x, self.target_shape)

    def get_config(self):
        """Get config for this layer."""
        config = {'target_shape': self.target_shape}
        base_config = super(TFReshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
