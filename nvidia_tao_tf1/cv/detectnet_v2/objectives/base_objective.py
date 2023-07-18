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

"""Base class / API definition of DNN objectives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from keras.layers import Conv2D
from keras.layers import Reshape
import six


class BaseObjective(six.with_metaclass(ABCMeta, object)):
    """Objective base class defining the interface to objectives and common methods.

    Objectives implement the following functionalities:
    - Rasterization (labels -> tensors)
    - Objective transforms (label domain <-> DNN output domain)
    - DNN output head creation
    - Cost function
    - Objective-specific visualization
    - Spatial transformation of objectives (applying spatial transformation
      matrices to predicted tensors)
    """

    @abstractmethod
    def __init__(self, input_layer_name, output_height, output_width):
        """Interface to initializing an Objective and the base initializer.

        Contains the common implementation, concrete classes need to call this.

        Args:
            input_layer_name (string): Name of the input layer of the Objective head.
                If None the last layer of the model will be used.
            output_height, output_width: Shape of the DNN output tensor.
        """
        self.num_channels = None
        self.gradient_flag = None
        self.activation = None
        self.learnable = True
        self.input_layer_name = input_layer_name
        self.template = None
        self.output_height = output_height
        self.output_width = output_width

    def dnn_head(self, num_classes, data_format, kernel_regularizer,
                 bias_regularizer):
        """Function for adding a head to DNN that outputs the prediction tensors.

        Applies the predictor head to a tensor, syntax:
        output = objective.dnn_head(...)(input_tensor)

        Args:
            num_classes: (int) Number of classes.
            data_format: (string) e.g. 'channels_first'.
            kernel_regularizer: Keras regularizer to be applied to convolution kernels.
            bias_regularizer: Keras regularizer to be applied to biases.

        Returns:
            Function for adding the predictor head.
        """
        # TODO: @vpraveen update the naming if mulitstide model is implemented.
        conv = Conv2D(filters=num_classes*self.num_channels,
                      kernel_size=[1, 1],
                      strides=(1, 1),
                      padding='same',
                      data_format=data_format,
                      dilation_rate=(1, 1),
                      activation=self.activation,
                      use_bias=True,
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      activity_regularizer=None,
                      kernel_constraint=None,
                      bias_constraint=None,
                      name="output_" + self.name)

        return conv

    def reshape_output(self, x, num_classes):
        """Reshape class index to its own dimension.

        Args:
            x: The output tensor as produced by self.dnn_head(), shape
                (num_classes*num_channels, H, W).
            num_classes: (int) Number of classes.

        Returns:
            Output tensor with shape (num_classes, num_channels, H, W).
        """
        shape = (num_classes, self.num_channels,
                 self.output_height, self.output_width)
        reshaped_x = Reshape(shape)(x)

        return reshaped_x

    def cost(self, y_true, y_pred, target_class, loss_mask=None):
        """Interface for creating the scalar cost for the Objective.

        Non-learnable objectives do not need to implement this method.

        Args:
            y_true: GT tensor dictionary
            y_pred: Prediction tensor dictionary
            target_class: (TargetClass) for which to create the cost
            loss_mask: (tf.Tensor) Loss mask to multiply the cost by.

        Returns:
            cost: TF scalar.
        """
        pass

    @abstractmethod
    def target_gradient(self, ground_truth_label):
        """Interface for creating target gradient config for rasterizer.

        This function is called separately for each bounding box target.
        The gradients are represented by tuples of coefficients c=(slope_x, slope_y, offset).
        This enables the rasterizer to rasterize a linear gradient whose value at pixel
        (x, y) is x * slope_x + y * slope_y + offset.
        The gradient may be multiplied by the coverage values, if the gradient flag is
        set accordingly.

        Args:
            ground_truth_label: dictionary of label attributes

        Returns:
            The gradients' coefficients.
        """
        pass

    def predictions_to_absolute(self, prediction):
        """Interface / pass through for converting predictions to absolute values.

        This function is called for each DNN output prediction tensor. The function
        transforms back the predictions to the absolute (dataset domain) values. For
        instance for bounding boxes the function converts grid-cell center relative
        coords to absolute coords.

        The base-class implementation returns the input prediction unmodified.

        Args:
            prediction (tensor): shape (batch, class, self.num_channels, height, width)

        Returns:
            transformed prediction (tensor)
        """
        return prediction

    def transform_predictions(self, prediction, matrices=None):
        """Interface / pass through for transforming predictions spatially.

        This may be used for example to undo spatial augmentation effect on
        the bounding box, depth, etc predictions.

        The base-class implementation returns the input prediction unmodified.

        Args:
            prediction (tensor): shape (batch, class, self.num_channels, height, width)
            matrices: A tensor of 3x3 transformation matrices, shape (batch, 3, 3).

        Returns:
            transformed prediction (tensor)
        """
        return prediction
