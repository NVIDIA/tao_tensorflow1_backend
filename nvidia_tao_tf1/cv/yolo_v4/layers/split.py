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

"""Split layer in YOLO v4 tiny."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer


class Split(Layer):
    '''Keras Split layer for doing tf.split.'''

    def __init__(self, groups, group_id, **kwargs):
        """Initialize the Split layer.

        Args:
            groups(int): Number of groups of channels.
            group_id(int): The ID of the output group of channels.
        """
        self.groups = groups
        self.group_id = group_id
        super(Split, self).__init__(**kwargs)

    def build(self, input_shape):
        """Setup some internal parameters."""
        self.nb_channels = input_shape[1]
        assert self.nb_channels % self.groups == 0, (
            "Number of channels is not a multiple of number of groups!"
        )

    def compute_output_shape(self, input_shape):
        """compute_output_shape.

        Args:
            input_shape(tuple): the shape of the input tensor.
        Returns:
            The output tensor shape: (N, C // g, h, w).
        """
        batch_size = input_shape[0]
        h = input_shape[2]
        w = input_shape[3]
        return (batch_size, self.nb_channels // self.groups, h, w)

    def call(self, x, mask=None):
        """Call this layer with inputs."""
        group_size = self.nb_channels // self.groups
        return x[:, group_size * self.group_id : group_size * (self.group_id + 1), :, :]

    def get_config(self):
        """Get config for this layer."""
        config = {'groups': self.groups, "group_id": self.group_id}
        base_config = super(Split, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
