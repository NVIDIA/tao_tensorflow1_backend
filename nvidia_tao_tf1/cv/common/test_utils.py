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

"""Simple script to test routines written in utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import keras
from keras import backend as K

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.templates.googlenet import GoogLeNet
from nvidia_tao_tf1.core.templates.resnet import ResNet
from nvidia_tao_tf1.core.templates.vgg import VggNet
from nvidia_tao_tf1.cv.common.utils import decode_to_keras, encode_from_keras


class TestUtils(object):
    """Class to test utils.py."""

    def _set_tf_session(self):
        # Restricting the number of GPU's to be used.
        gpu_id = str(0)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = gpu_id
        K.set_session(tf.Session(config=config))

    def _get_googlenet(self, nlayers, input_shape, all_projections=False,
                       use_batch_norm=False, use_pooling=False, data_format='channels_first'):
        inputs = keras.layers.Input(shape=input_shape)
        return GoogLeNet(inputs=inputs,
                         use_batch_norm=use_batch_norm,
                         data_format=data_format)

    def _get_resnet(self, nlayers, input_shape, all_projections=True,
                    use_batch_norm=False, use_pooling=False, data_format='channels_first'):
        inputs = keras.layers.Input(shape=input_shape)
        return ResNet(nlayers, inputs, all_projections=all_projections,
                      use_batch_norm=use_batch_norm,
                      use_pooling=use_pooling,
                      data_format=data_format)

    def _get_vgg(self, nlayers, input_shape, all_projections=False,
                 use_batch_norm=False, use_pooling=False, data_format='channels_first'):
        inputs = keras.layers.Input(shape=input_shape)
        return VggNet(nlayers, inputs,
                      use_batch_norm=use_batch_norm,
                      use_pooling=use_pooling,
                      data_format=data_format)

    def _create_model(self, model_template, nlayers=None, use_pooling=False, use_batch_norm=False,
                      input_shape=(3, 224, 224), data_format="channels_first",
                      instantiation_mode='detector'):
        self._set_tf_session()
        # Constructing a dictionary for feature extractor templates.
        self.model_choose = {"ResNet": self._get_resnet,
                             "VggNet": self._get_vgg,
                             "GoogLeNet": self._get_googlenet}

        # Choosing the feature extractor template.
        if model_template in self.model_choose.keys():
            keras_model = self.model_choose[model_template](nlayers, input_shape,
                                                            use_batch_norm=use_batch_norm,
                                                            use_pooling=use_pooling,
                                                            data_format=data_format)
        else:
            raise NotImplementedError('Unsupported model template: {}'.format(model_template))

        # Hooking the model to the respective outputs.
        x = keras_model.outputs[0]

        # Appending the outputs based on the instantiation mode.
        if instantiation_mode == "classifier":
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(10, activation='softmax',
                                   name='output')(x)
            outputs = [x]
        elif instantiation_mode == 'detector':
            x1 = keras.layers.Conv2D(filters=1,
                                     kernel_size=[1, 1],
                                     strides=(1, 1),
                                     padding='same',
                                     data_format=data_format,
                                     dilation_rate=(1, 1),
                                     name='conv2d_cov')(x)
            x2 = keras.layers.Conv2D(filters=4,
                                     kernel_size=[1, 1],
                                     strides=(1, 1),
                                     padding='same',
                                     data_format=data_format,
                                     dilation_rate=(1, 1),
                                     name='conv2d_bbox')(x)
            outputs = [x1, x2]
        else:
            raise NotImplementedError("Unknown instantiation mode: {}".format(instantiation_mode))

        # Generate final keras model.
        keras_model = keras.models.Model(inputs=keras_model.inputs, outputs=outputs)

        return keras_model

    @pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
    @pytest.mark.parametrize("model, nlayers, data_format, use_batch_norm, input_shape,"
                             "use_pooling, instantiation_mode, enc_key",
                             [
                              ("ResNet", 10, 'channels_first', False, (3, 128, 64), True,
                               'classifier', 'th0@!#$(@*#$)is'),
                              ("ResNet", 18, 'channels_first', True, (3, 128, 64), False,
                               'detector', '0@!#$(@*#$)'),
                              ("VggNet", 16, 'channels_last', False, (3, 128, 64), False,
                               'classifier', '0@!#$(@*#$)')
                             ])
    def test_encode_decode(self, model, nlayers, data_format, use_batch_norm,
                           input_shape, use_pooling, instantiation_mode, enc_key):
        """Simple function to test encode and decode wrappers.

        Args:
            model (str): Name of the model template.
            nlayers (int): Number of layers for Resnet-xx and Vgg-xx models. This parameter
                may be set to None for fixed templates.
            data_format (str): Setting keras backend data format to 'channels_first' and
                'channels_last'.
            use_batch_norm (bool): Flag to enable or disable batch norm.
            input_shape (tuple): Shape of the keras model input.
            use_pooling (bool): Flag to enable or disable pooling.
            instantiation_mode (str): The mode to define the models. Two types of instantiations
                are allowed.
                1. classification: Here the output is a single dense layer of nclasses = 10
                2. detection: Here there are two outputs.
                    a. output_cov: number of filters = 1
                    b. output_bbox: number of filters = 4
            enc_key (str): Key for encrpytion and decryption.
        """
        enc_key = str.encode(enc_key)
        os_handle, output_file_name = tempfile.mkstemp()
        os.close(os_handle)

        # Defining the keras model for testing.
        keras_model = self._create_model(model, nlayers,
                                         use_pooling=use_pooling,
                                         input_shape=input_shape,
                                         use_batch_norm=use_batch_norm,
                                         data_format=data_format,
                                         instantiation_mode=instantiation_mode)

        # Encrypting the model.
        encode_from_keras(keras_model, output_file_name, enc_key)

        # Decrypting the above encrypted model.
        decoded_model = decode_to_keras(output_file_name, enc_key)

        # Extracting the data format parameter to detect input shape.
        data_format = decoded_model.layers[1].data_format

        # Computing shape of input tensor.
        image_shape = decoded_model.layers[0].input_shape[1:4]

        # Create an input image.
        test_image = np.random.randn(image_shape[0], image_shape[1], image_shape[2])
        test_image.shape = (1, ) + test_image.shape

        # Infer on both original and decrypted model.
        decoded_model_output = decoded_model.predict(test_image, batch_size=1)
        original_model_output = keras_model.predict(test_image, batch_size=1)

        # Check for equality of inferences.
        assert len(decoded_model_output) == len(original_model_output)
        for output in range(len(decoded_model_output)):
            assert np.array_equal(
                decoded_model_output[output],
                original_model_output[output]
            )
