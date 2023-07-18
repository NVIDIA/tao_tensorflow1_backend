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

"""Base Unet model class that takes care of constructing feature extractor."""

import logging
import keras
from keras.layers import Input
from keras.models import Model
from keras.models import model_from_json
import tensorflow as tf

from nvidia_tao_tf1.core.models.quantize_keras_model import create_quantized_keras_model
from nvidia_tao_tf1.cv.unet.model.utilities import get_num_unique_train_ids
from nvidia_tao_tf1.cv.unet.proto.model_config_pb2 import ModelConfig


logger = logging.getLogger(__name__)


class UnetModel(object):
    """UNet class.

    UNet class contains functionality for constructing Unet with all backbones.
    """

    def __init__(self, num_layers, use_pooling, use_batch_norm, dropout_rate,
                 activation_config, target_class_names,
                 freeze_pretrained_layers, allow_loaded_model_modification,
                 template='resnet', all_projections=True, freeze_blocks=None,
                 freeze_bn=False, load_graph=True, initializer=None, seed=None,
                 enable_qat=False):
        """Init function.

        Args:
            num_layers (int): Number of layers for scalable feature extractors.
            use_pooling (bool): Whether to add pooling layers to the feature extractor.
            use_batch_norm (bool): Whether to add batch norm layers.
            dropout_rate (float): Fraction of the input units to drop. 0.0 means dropout is
                not used.
            target_class_names (list): A list of target class names.
            freeze_pretrained_layers (bool): Prevent updates to pretrained layers' parameters.
            allow_loaded_model_modification (bool): Allow loaded model modification.
            template (str): Model template to use for feature extractor.
            freeze_bn (bool): The boolean to freeze BN or not.
            load_graph (bool): The boolean to laod graph for phase 1.
            initializer (str): layer initializer
            enable_qat (bool): flag to enable QAT during training
            seed (int): Experiment seed for initialization

        """
        self.num_layers = num_layers
        self.use_pooling = use_pooling
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.template = template
        self.keras_model = None
        self.keras_training_model = None
        self.target_class_names = target_class_names
        self.num_target_classes = get_num_unique_train_ids(self.target_class_names)
        if activation_config == "sigmoid":
            self.num_target_classes = 1
        self.activation_config = activation_config
        self.freeze_pretrained_layers = freeze_pretrained_layers
        self.freeze_blocks = freeze_blocks
        self.freeze_bn = freeze_bn
        self.allow_loaded_model_modification = allow_loaded_model_modification
        self.constructed = False
        self.all_projections = all_projections
        self.inference_model = None
        self.load_graph = load_graph
        self.data_format = "channels_first"
        self.seed = seed
        self.initializer = self.get_initializer(initializer)
        self.enable_qat = enable_qat
        self.num_params = None

    def construct_model(self, input_shape, kernel_regularizer=None,
                        bias_regularizer=None, pretrained_weights_file=None,
                        enc_key=None, mode=None, features=None,
                        export=False, model_json=None, construct_qat=False,
                        custom_objs=None, qat_on_pruned=False):
        """Function to construct the encoder/ decoder.

        Args:
            input_shape (tuple / list / TensorShape):
                model input shape without batch dimension (C, H, W).
            kernel_regularizer (keras.regularizers.Regularizer instance):
                Regularizer to be applied to convolution kernels.
            bias_regularizer (keras.regularizers.Regularizer instance):
                Regularizer to be applied to biases.
            pretrained_weights_file (str): An optional model weights file to be loaded.
            enc_key (str): The key to encrypt model.
            model_json (str): path of pruned model graph.
            qat_on_pruned (bool): This is to indicate if we are applying qat on pruned
                model. That way the orig model will be loaded from pruned json and
                not constructed.
            construct_qat (bool): To indicate if qat grpah is being constructed
            custom_objs (dictionary): The custom layers present in the model

        Raises:
            NotImplementedError: If pretrained_weights_file is not None.
        """
        if self.load_graph or (self.enable_qat and not construct_qat):
            # Load the pruned graph
            with open(model_json, 'r') as json_file:
                loaded_model_json = json_file.read()
            model_unet_tmp = model_from_json(loaded_model_json,
                                             custom_objects=custom_objs)
            # Adding softmax explicitly
            if export:
                inputs = model_unet_tmp.input
                out = model_unet_tmp.output
                out = self.get_activation_for_export(out)
                model_unet = Model(inputs=[inputs], outputs=[out])
            else:
                model_unet = model_unet_tmp
            if qat_on_pruned:
                # QAT on top of pruned model for training
                model_unet = create_quantized_keras_model(model_unet)
        else:
            if "vanilla" in self.template:
                model_unet = self.construct_encoder_decoder_model(input_shape, mode,
                                                                  features, export)
            else:
                encoder_model = self.construct_feature_extractor(
                    input_shape=input_shape, data_format=self.data_format,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    features=features, use_pooling=True)
                model_unet = self.construct_decoder_model(encoder_model, export=export)
            if self.enable_qat:
                model_unet = create_quantized_keras_model(model_unet)

        if construct_qat:
            # Construction phase of QAT
            model_json = model_unet.to_json()
            keras.backend.clear_session()
            return model_json

        self.keras_model = model_unet
        self.num_params = model_unet.count_params()
        self.constructed = True

        return None

    def get_initializer(self, initializer):
        """Function to assign the kernel initializer."""

        if initializer == ModelConfig.KernelInitializer.HE_UNIFORM:
            return tf.compat.v1.initializers.he_uniform(seed=self.seed)
        if initializer == ModelConfig.KernelInitializer.HE_NORMAL:
            return tf.compat.v1.initializers.he_normal(seed=self.seed)
        return 'glorot_uniform'

    def construct_feature_extractor(self, input_shape, data_format,
                                    kernel_regularizer=None,
                                    bias_regularizer=None, features=None,
                                    use_pooling=False):
        """Generate a keras feature extractor model.

        Args:
            input_shape: model input shape (N,C,H,W). N is ignored.
            data_format: Order of the dimensions (C, H, W).
            kernel_regularizer: Keras regularizer to be applied to convolution
                                kernels.
            bias_regularizer: Keras regularizer to be applied to biases.
            features: Fed by the estimator during training.
            use_pooling: Bool to use pooling.
        Raises:
            AssertionError: If the model is already constructed.
        Returns:
            model (keras.model): The model for feature extraction.
        """
        assert not self.constructed, "Model already constructed."
        # Define entry points to the model.
        assert len(input_shape) == 3
        self.input_num_channels = int(input_shape[0])
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        if features:
            inputs = Input(tensor=features, shape=input_shape)
        else:
            inputs = Input(shape=(self.input_num_channels, self.input_height,
                                  self.input_width))

        # Set up positional arguments and key word arguments to instantiate
        # feature extractor templates.
        args = [self.num_layers, inputs]

        kwargs = {'use_batch_norm': self.use_batch_norm,
                  'kernel_regularizer': kernel_regularizer,
                  'bias_regularizer': bias_regularizer,
                  'freeze_blocks': self.freeze_blocks,
                  'freeze_bn': self.freeze_bn,
                  'use_pooling': use_pooling
                  }
        if self.template == "shufflenet":
            kwargs['input_shape'] = input_shape
        model = self.get_base_model(args, kwargs)
        # Feature extractor output shape.
        if self.template != "shufflenet":
            # Feature extractor output shape.
            self.output_height = model.output_shape[2]
            self.output_width = model.output_shape[3]

        return model

    def get_activation_for_export(self, out):
        """Function to get the activation of last layer.

        Args:
            out (keras layer): Keras layer output
        """
        if self.activation_config == "sigmoid":
            out = keras.layers.Activation('sigmoid', name="sigmoid")(out)
        else:
            out = keras.layers.Permute((2, 3, 1))(out)
            out = keras.layers.Softmax(axis=-1)(out)
        return out
