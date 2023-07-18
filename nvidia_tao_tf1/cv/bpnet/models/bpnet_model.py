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
"""BpNetModel model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import re

import keras
from keras.initializers import constant
from keras.initializers import glorot_uniform
from keras.initializers import random_normal
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Multiply
from keras.models import Model

import tensorflow as tf

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.blocks.models.keras_model import KerasModel
from nvidia_tao_tf1.core.models.templates.utils import get_batchnorm_axis
from nvidia_tao_tf1.core.templates.resnet import ResNet
from nvidia_tao_tf1.cv.bpnet.models.templates.helnet import HelNet
from nvidia_tao_tf1.cv.bpnet.models.templates.vgg import VggNet
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import encode_from_keras

logger = logging.getLogger(__name__)


class BpNetModel(KerasModel):
    """BpNet model definition."""

    @tao_core.coreobject.save_args
    def __init__(self,
                 backbone_attributes,
                 stages=6,
                 heat_channels=19,
                 paf_channels=38,
                 use_self_attention=False,
                 data_format='channels_last',
                 use_bias=True,
                 regularization_type='l1',
                 kernel_regularization_factor=1e-9,
                 bias_regularization_factor=1e-9,
                 kernel_initializer='random_normal',
                 **kwargs):
        """Initialize the model.

        Args:
            backbone (str): vgg, helnet
            stages (int): Number of stages of refinement in the network
            data_format (str): Channel ordering
            regularization_type (str): 'l1', 'l2' or 'l1_l2'.
            regularization_factor (float): regularization weight.
        """
        super(BpNetModel, self).__init__(**kwargs)
        self._backbone_attributes = backbone_attributes
        self._data_format = data_format
        self._stages = stages
        self._paf_stages = stages
        self._cmap_stages = stages
        self._heat_channels = heat_channels
        self._paf_channels = paf_channels
        self._use_self_attention = use_self_attention
        self._use_bias = use_bias
        if kernel_initializer == 'xavier':
            self._kernel_initializer = glorot_uniform()
        else:
            self._kernel_initializer = random_normal(stddev=0.01)
        self._bias_initializer = constant(0.0)
        self._regularization_type = regularization_type
        self._kernel_regularization_factor = kernel_regularization_factor
        self._bias_regularization_factor = bias_regularization_factor

        self._set_regularizer()

    # TODO: move to utils
    def _set_regularizer(self):
        """Return regularization function."""
        if self._regularization_type == 'l1':
            kernel_regularizer = keras.regularizers.l1(
                self._kernel_regularization_factor)
            bias_regularizer = keras.regularizers.l1(
                self._bias_regularization_factor)
        elif self._regularization_type == 'l2':
            kernel_regularizer = keras.regularizers.l2(
                self._kernel_regularization_factor)
            bias_regularizer = keras.regularizers.l2(
                self._bias_regularization_factor)
        elif self._regularization_type == 'l1_l2':
            kernel_regularizer = keras.regularizers.l1_l2(
                self._kernel_regularization_factor)
            bias_regularizer = keras.regularizers.l1_l2(
                self._bias_regularization_factor)
        else:
            raise NotImplementedError(
                "Regularization type: {} is not supported.".format(
                    self._regularization_type))

        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

    def regularization_losses(self):
        """Get the regularization losses.

        Returns:
            Scalar (tensor fp32) with the model dependent (regularization) losses.
        """
        return tf.reduce_sum(self.keras_model.losses)

    # TODO (sakthivels): move to layers.py
    def _maxpool(self, input_tensor, name, kernel_size=(2, 2), stride=2):
        """Add MaxPool layer to the network.

        Args:
            input_tensor (Tensor): An input tensor object.
            kernel_size (int): Size of the kernel.
            stride (int): Size of the stride.
            name (str): Name of the maxpool layer.

        Returns:
            tensor (Tensor): The output Tensor object after construction.
        """
        return MaxPooling2D(kernel_size, (stride, stride),
                            padding='same',
                            name=name,
                            data_format=self._data_format)(input_tensor)

    # TODO (sakthivels): move to layers.py (and remove regaulizers
    #       and initalizers. use update_regularizers in utils.py )
    def _conv2d_block(self,
                      input_tensor,
                      num_filters,
                      kernel_size,
                      name,
                      stride=1,
                      activation_type=None,
                      use_bn=False):
        """Construct a convolution layer to the network.

        Args:
            input_tensor (Tensor): An input tensor object.
            num_filters (int): Number of filters.
            kernel_size (int): Size of the kernel.
            stride (int): Size of the stride.
            name (str): Name of the conv block.

        Returns:
            tensor (Tensor): The output Tensor object after construction.
        """
        conv_layer = Conv2D(num_filters,
                            kernel_size=kernel_size,
                            strides=(stride, stride),
                            padding='same',
                            data_format=self._data_format,
                            use_bias=self._use_bias,
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            kernel_regularizer=self._kernel_regularizer,
                            bias_regularizer=self._bias_regularizer,
                            name=name)

        tensor = conv_layer(input_tensor)

        if use_bn:
            tensor = BatchNormalization(axis=get_batchnorm_axis(
                self._data_format),
                                        name=name + "/BN")(tensor)
        if activation_type is not None:
            tensor = Activation(activation_type,
                                name=name + "/" + activation_type)(tensor)

        return tensor

    # TODO (sakthivels): move to base_models and add get_customvgg to cut off at layer 10
    def _build_vgg_backbone(self, x):
        """Build a VGG backbone network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            x (Tensor): Output Tensor (feature map)
        """

        # Block 1
        x = self._conv2d_block(x,
                               64, (3, 3),
                               name='block1_conv1',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               64, (3, 3),
                               name='block1_conv2',
                               activation_type='relu')
        x = self._maxpool(x, name='block1_pool')

        # Block 2
        x = self._conv2d_block(x,
                               128, (3, 3),
                               name='block2_conv1',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               128, (3, 3),
                               name='block2_conv2',
                               activation_type='relu')
        x = self._maxpool(x, name='block2_pool')

        # Block 3
        x = self._conv2d_block(x,
                               256, (3, 3),
                               name='block3_conv1',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               256, (3, 3),
                               name='block3_conv2',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               256, (3, 3),
                               name='block3_conv3',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               256, (3, 3),
                               name='block3_conv4',
                               activation_type='relu')
        x = self._maxpool(x, name='block3_pool')

        # Block 4
        x = self._conv2d_block(x,
                               512, (3, 3),
                               name='block4_conv1',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               512, (3, 3),
                               name='block4_conv2',
                               activation_type='relu')

        # Non-VGG layers
        x = self._conv2d_block(x,
                               256, (3, 3),
                               name='interface/conv4_3',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               128, (3, 3),
                               name='interface/conv4_4',
                               activation_type='relu')

        return x

    # TODO (sakthivels): add as a seperate head (initial_stages)
    def _build_stage1(self,
                      x,
                      out_channels,
                      num_channels=128,
                      scope_name='stage1/'):
        """Build the first stage of body pose estimation network.

        Args:
            x (Tensor): Input tensor.
            out_channels (int): Number of final output channels
                (depends on number of parts and branch)
            scope_name (str): Scope name for the stage (ex. stage1/heat_branch)

        Returns:
            x (Tensor): Output Tensor
        """

        x = self._conv2d_block(x,
                               num_channels, (3, 3),
                               name=scope_name + 'conv1',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               num_channels, (3, 3),
                               name=scope_name + 'conv2',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               num_channels, (3, 3),
                               name=scope_name + 'conv3',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               512, (1, 1),
                               name=scope_name + 'conv4',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               out_channels, (1, 1),
                               name=scope_name + 'out')

        return x

    # TODO (sakthivels): add as a seperate head (refinement_stages)
    def _build_stageT(self,
                      x,
                      out_channels,
                      scope_name,
                      branch_type,
                      num_channels=128,
                      kernel_size=(7, 7),
                      is_final_block=False):
        """Build the first stage of body pose estimation network.

        Args:
            x (Tensor): Input tensor.
            out_channels (int): Number of final output channels
                (depends on number of parts and branch)
            scope_name (str): Scope name for the stage
                (ex. stage2/heat_branch, stage3/paf_branch etc.)

        Returns:
            x (Tensor): Output Tensor
        """

        x = self._conv2d_block(x,
                               num_channels,
                               kernel_size,
                               name=scope_name + 'conv1',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               num_channels,
                               kernel_size,
                               name=scope_name + 'conv2',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               num_channels,
                               kernel_size,
                               name=scope_name + 'conv3',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               num_channels,
                               kernel_size,
                               name=scope_name + 'conv4',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               num_channels,
                               kernel_size,
                               name=scope_name + 'conv5',
                               activation_type='relu')
        x = self._conv2d_block(x,
                               num_channels, (1, 1),
                               name=scope_name + 'conv6',
                               activation_type='relu')

        # Self attention block
        # TODO (sakthivels): add as a seperate head (self_attention_head)
        if self._use_self_attention:
            if branch_type == "paf":
                activation_type = 'tanh'
            else:
                activation_type = 'sigmoid'
            out_att = self._conv2d_block(x,
                                         num_channels, (3, 3),
                                         name=scope_name + 'attention_conv',
                                         activation_type=activation_type)

            # apply attention maps
            x = Multiply()([x, out_att])

        # Final layer naming
        if is_final_block:
            last_layer_name = "{}_{}".format(branch_type, 'out')
        else:
            last_layer_name = "{}{}".format(scope_name, 'out')

        x = self._conv2d_block(x,
                               out_channels, (1, 1),
                               name=last_layer_name)

        return x

    # TODO (sakthivels): build model by adding blocks to base in loop (in config as list)
    def build(self, input_image):
        """Create a Keras model to perform body pose estimation.

        Args:
            inputs (4D tensor): the input images.
        """

        heat_outputs = []
        paf_outputs = []

        # Define the inputs to the network
        input_layer = Input(tensor=input_image,
                            shape=(None, None, 3),
                            name='input_1')

        # Add Backbone network
        # VGG backbone
        if self._backbone_attributes["architecture"] == 'vgg':
            self._backbone_attributes["nlayers"] = self._backbone_attributes.get("nlayers", 19)
            assert self._backbone_attributes["nlayers"] == 19, "Only VGG19 is supported currently."
            use_bias = self._backbone_attributes["use_bias"]
            model = VggNet(self._backbone_attributes["nlayers"],
                           input_layer,
                           use_batch_norm=True,
                           data_format=self._data_format,
                           use_pooling=False,
                           use_bias=use_bias,
                           use_modified_vgg=True,
                           kernel_regularizer=self._kernel_regularizer,
                           bias_regularizer=self._bias_regularizer)
            feat = model.outputs[0]
        # Helnet backbone
        elif self._backbone_attributes["architecture"] == 'helnet':
            model = HelNet(
                self._backbone_attributes["nlayers"],
                input_layer,
                self._backbone_attributes["mtype"],
                use_last_block=False,
                use_batch_norm=self._backbone_attributes["use_batch_norm"],
                data_format=self._data_format)
            feat = model.outputs[0]
        # Resnet backbone
        elif self._backbone_attributes["architecture"] == 'resnet':
            model = ResNet(
                self._backbone_attributes["nlayers"],
                input_layer,
                use_batch_norm=self._backbone_attributes["use_batch_norm"],
                data_format=self._data_format)
            feat = model.outputs[0]
        # Else raise error
        else:
            raise NotImplementedError(
                "Backbone network: {} is not supported.".format(
                    self._backbone_attributes["architecture"]))

        # If enabled, add a convolution with 128 kernels,
        # in essence, to reduce the backbone feat map size
        feat = self._conv2d_block(feat,
                                  128, (3, 3),
                                  name='channel_reduction_conv',
                                  activation_type='relu')

        # Add Stage 1 network
        paf_outputs.append(
            self._build_stage1(feat,
                               self._paf_channels,
                               num_channels=128,
                               scope_name="stage1/paf_branch/"))
        heat_outputs.append(
            self._build_stage1(feat,
                               self._heat_channels,
                               num_channels=128,
                               scope_name="stage1/heat_branch/"))

        # Add Stages >= 2
        for stage_idx in range(2, self._stages + 1):
            x = Concatenate()([feat, heat_outputs[-1], paf_outputs[-1]])
            paf_outputs.append(
                self._build_stageT(x, self._paf_channels,
                                   "stage{}/paf_branch/".format(stage_idx),
                                   "paf", is_final_block=(stage_idx == self._stages)))
            heat_outputs.append(
                self._build_stageT(x, self._heat_channels,
                                   "stage{}/heat_branch/".format(stage_idx),
                                   "heatmap", is_final_block=(stage_idx == self._stages)))

        model = Model(inputs=input_layer, outputs=heat_outputs + paf_outputs)

        self._keras_model = model

        return self._keras_model.outputs

    def get_lr_multipiers(self):
        """Get the Learning rate multipliers for different stages of the model."""

        # setup lr multipliers for conv layers
        lr_mult = dict()
        for layer in self._keras_model.layers:
            if isinstance(layer, Conv2D):
                # stage = 1
                if re.match("stage1.*", layer.name):
                    kernel_name = layer.weights[0].name.split(':')[0]
                    lr_mult[kernel_name] = 1
                    if len(layer.weights) > 1:
                        bias_name = layer.weights[1].name.split(':')[0]
                        lr_mult[bias_name] = 2
                # stage > 1
                elif re.match("stage.*", layer.name):
                    kernel_name = layer.weights[0].name.split(':')[0]
                    lr_mult[kernel_name] = 4
                    if len(layer.weights) > 1:
                        bias_name = layer.weights[1].name.split(':')[0]
                        lr_mult[bias_name] = 8
                # output nodes
                elif re.match(".*out", layer.name):
                    kernel_name = layer.weights[0].name.split(':')[0]
                    lr_mult[kernel_name] = 4
                    if len(layer.weights) > 1:
                        bias_name = layer.weights[1].name.split(':')[0]
                        lr_mult[bias_name] = 8
                # vgg
                else:
                    # Commented for TLT branch
                    # logger.info("Layer matched as backbone layer: {}".format(layer.name))
                    kernel_name = layer.weights[0].name.split(':')[0]
                    lr_mult[kernel_name] = 1
                    if len(layer.weights) > 1:
                        bias_name = layer.weights[1].name.split(':')[0]
                        lr_mult[bias_name] = 2
        return lr_mult

    def save_model(self, file_name, enc_key=None):
        """Save the model to disk.

        Args:
            file_name (str): Model file name.
            enc_key (str): Key string for encryption.
        Raises:
            ValueError if postprocessing_config is None but save_metadata is True.
        """
        self.keras_model.save(file_name, overwrite=True)
