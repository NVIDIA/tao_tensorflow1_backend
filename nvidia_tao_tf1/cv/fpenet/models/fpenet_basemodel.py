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
"""FpeNetBaseModel class that takes care of constructing, training and validating a model."""

import gc
import logging
import keras
from keras.layers import Input
from nvidia_tao_tf1.blocks.models import KerasModel
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import encode_from_keras, model_io
from nvidia_tao_tf1.cv.fpenet.models.custom.softargmax import Softargmax
from nvidia_tao_tf1.cv.fpenet.models.fpenet import FpeNetModel

logger = logging.getLogger(__name__)


class FpeNetBaseModel(KerasModel):
    """FpeNetBaseModel class.

    FpeNetBaseModel contains functions for constructing and manipulating a Keras based FpeNet model,
    building training and validation graphs for the model, and visualizing predictions.
    """

    @save_args
    def __init__(self,
                 model_parameters,
                 visualization_parameters=None,
                 **kwargs):
        """__init__ method.

        Args:
            model_parameters (dict): A dictionary of all the parameters required in initialization.
            visualization_parameters (dict): A dictionary of info required by visualization.
        """
        super(FpeNetBaseModel, self).__init__(**kwargs)
        self._data_format = 'channels_first'

        self._regularizer_type = model_parameters.get('regularizer_type')
        self._regularizer_weight = model_parameters.get('regularizer_weight')
        self._pretrained_model_path = model_parameters.get('pretrained_model_path')
        self._model_type = model_parameters.get('type')
        self._use_upsampling_layer = model_parameters.get('use_upsampling_layer')

        self._input_shape = None
        self._output_shape = None
        self._keras_model = None

        if 'beta' in model_parameters.keys():
            self._beta = model_parameters.get('beta')
        else:
            raise ValueError('beta value not provided.')

        self._kernel_regularizer = None
        self._bias_regularizer = None

        # Set regularizer.
        if self._regularizer_type.lower() == 'l1':
            self._kernel_regularizer = keras.regularizers.l1(
                self._regularizer_weight)
            self._bias_regularizer = keras.regularizers.l1(
                self._regularizer_weight)
        elif self._regularizer_type.lower() == 'l2':
            self._kernel_regularizer = keras.regularizers.l2(
                self._regularizer_weight)
            self._bias_regularizer = keras.regularizers.l2(
                self._regularizer_weight)
        else:
            raise ValueError('%s regularizer type not supported.' % self._regularizer_type)

    def build(self, input_images, enc_key=None, num_keypoints=80):
        """Build a FpeNet Keras model.

        Args:
            input_images: Input images to FpeNet model.
        """
        input_tensor = Input(tensor=input_images, name="input_face_images")
        logging.info("model type is: " + self._model_type)

        if self._keras_model is not None:

            predictions = {}
            predictions['landmarks'] = self._keras_model.outputs[0]
            predictions['confidence'] = self._keras_model.outputs[1]
            predictions['heat_map'] = self._keras_model.outputs[2]
            return predictions

        if self._model_type == 'FpeNet_base':
            model = FpeNetModel(
                data_format=self._data_format,
                blocks_decoder=[[(3, 64), (1, 64)]] * 4,
                nkeypoints=num_keypoints,
                additional_conv_layer=True,
                use_upsampling_layer=self._use_upsampling_layer,
                beta=self._beta,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
        elif self._model_type == 'FpeNet_base_5x5_conv':
            model = FpeNetModel(
                data_format=self._data_format,
                blocks_decoder=[[(5, 64), (1, 64)]] * 4,
                nkeypoints=num_keypoints,
                additional_conv_layer=False,
                use_upsampling_layer=self._use_upsampling_layer,
                beta=self._beta,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
        elif self._model_type == 'FpeNet_release':
            model = FpeNetModel(
                data_format=self._data_format,
                blocks_decoder=[[(3, 64), (1, 64)]] * 4,
                nkeypoints=num_keypoints,
                additional_conv_layer=True,
                use_upsampling_layer=True,
                beta=self._beta,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
        elif self._model_type == 'FpeNet_public':
            model = FpeNetModel(
                data_format=self._data_format,
                blocks_decoder=[[(3, 64), (1, 64)]] * 4,
                nkeypoints=num_keypoints,
                additional_conv_layer=False,
                use_upsampling_layer=False,
                beta=self._beta,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
        else:
            raise NotImplementedError(
                'A FpeNet model of type %s is not implemented.' %
                self._model_type)

        model = model.construct(input_tensor)

        # If you have weights you've trained previously, you can load them into this model.
        if self._pretrained_model_path is not None:
            loaded_model = model_io(self._pretrained_model_path,
                                    enc_key=enc_key,
                                    custom_objects={"Softargmax": Softargmax})
            loaded_model_layers = [layer.name for layer in loaded_model.layers]
            logger.info("Loading weights from pretrained model file. {}".format(
                self._pretrained_model_path))
            for layer in model.layers:
                if layer.name in loaded_model_layers:
                    pretrained_layer = loaded_model.get_layer(layer.name)
                    weights_pretrained = pretrained_layer.get_weights()
                    model_layer = model.get_layer(layer.name)
                    try:
                        model_layer.set_weights(weights_pretrained)
                    except ValueError:
                        continue
            del loaded_model
            gc.collect()
        else:
            logger.info('This model will be trained from scratch.')

        self._keras_model = model

        predictions = {}
        predictions['landmarks'] = self._keras_model.outputs[0]
        predictions['confidence'] = self._keras_model.outputs[1]
        predictions['heat_map'] = self._keras_model.outputs[2]

        return predictions

    def load_model(self, model_file):
        """Load a previously saved Keras model.

        Args:
            model_file: Keras Model file name.
        """
        self._keras_model = keras.models.load_model(model_file, compile=False)
        self._input_shape = self._keras_model.input_shape
        self._output_shape = self._keras_model.output_shape

    @property
    def keras_model(self):
        """Get Keras model as a class property."""
        return self._keras_model

    @keras_model.setter
    def keras_model(self, model):
        """Set to a new model."""
        self._keras_model = model

    def save_model(self, file_name, enc_key=None):
        """Save the model to disk.

        Args:
            file_name (str): Model file name.
            enc_key (str): Key string for encryption.
        Raises:
            ValueError if postprocessing_config is None but save_metadata is True.
        """
        self.keras_model.save(file_name, overwrite=True)
