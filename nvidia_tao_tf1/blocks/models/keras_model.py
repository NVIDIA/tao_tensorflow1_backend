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
"""Base Keras Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

from nvidia_tao_tf1.blocks.models.model import Model


class KerasModel(Model):
    """Keras Model class."""

    def __init__(self, weights_path=None, **kwargs):
        """__init__ method."""
        super(KerasModel, self).__init__(**kwargs)
        self._keras_model = self._outputs = None
        self._weights_path = weights_path

    @property
    def keras_model(self):
        """Return the Keras model built.

        Returns:
            keras.Model: Keras model built through the `build` function call.
        """
        return self._keras_model

    @abstractmethod
    def build(self, inputs):
        """
        Create a Keras model.

        Args:
            inputs (tf.Tensor): input tensor to the model.
        """
        raise NotImplementedError("KerasModel build not implemented")

    def __call__(self, inputs):
        """__call__ method.

        Enables calling KerasModel in a functional style similar to Keras.
        Calls `build(inputs)` if the Keras model has not been build.
        Loads weights to model if pretrained_weights_path is specified.

        Args:
            inputs (tf.Tensor): input tensor to the model.

        Returns:
            list of tf.Tensor: output tensors from the model.
        """
        if not self._keras_model:
            outputs = self.build(inputs)
            if self._weights_path:
                self._keras_model.load_weights(self._weights_path, by_name=True)
            return outputs[0] if len(outputs) == 1 else outputs
        return self._keras_model(inputs)

    def save_model(self, model_file_path):
        """Save the model to disk.

        Args:
            model_file_path: full path to save model.
        """
        self._keras_model.save(model_file_path)
