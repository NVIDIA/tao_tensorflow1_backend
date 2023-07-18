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
"""Modulus utilities for model templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import re
import tempfile

from nvidia_tao_tf1.core.decorators.arg_scope import add_arg_scope
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.utils import get_uid

if os.environ.get("TF_KERAS"):
    from tensorflow import keras
else:
    import keras

logger = logging.getLogger(__name__)

bn_axis_map = {"channels_last": 3, "channels_first": 1}


class SUBBLOCK_IDS(object):
    """A operator to get index of subblock, overload [] operation."""

    def __getitem__(self, key):
        """
        Generate a subblock ID and return.

        Args:
            key (int): an index used to generate the subblock ID.
        """
        cur = key
        subblock_id = ""
        while cur >= 0:
            ch = chr(ord("a") + cur % 26)
            subblock_id = ch + subblock_id
            cur = cur // 26 - 1

        return subblock_id


def get_batchnorm_axis(data_format):
    """Convert a data_format string to the correct index in a 4 dimensional tensor.

    Args:
        data_format (str): either 'channels_last' or 'channels_first'.
    Returns:
        int: the axis corresponding to the `data_format`.
    """
    return bn_axis_map[data_format]


def add_dense_head(model, inputs, nclasses, activation):
    """
    Create a model that stacks a dense head on top of a another model. It is also flattened.

    Args:
        model (Model): the model on top of which the head should be created.
        inputs (tensor): the inputs (tensor) to the previously supplied model.
        nclasses (int): the amount of outputs of the dense map
        activation (string): activation function to use e.g. 'softmax' or 'linear'.

    Returns:
        Model: A model with the head stacked on top of the `model` input.
    """
    x = model.outputs[0]
    head_name = "head_fc%d" % (nclasses)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(nclasses, activation=activation, name=head_name)(x)
    model = keras.models.Model(
        inputs=inputs, outputs=x, name="%s_fc%d" % (model.name, nclasses)
    )
    return model


@add_arg_scope
def conv2D_bn_activation(
    x,
    use_batch_norm,
    filters,
    kernel_size,
    strides=(1, 1),
    activation_type="relu",
    activation_kwargs=None,
    data_format=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    layer_name=None,
    use_bias=True,
    quantize=False,
    bitwidth=8,
):
    """
    Add a conv layer, followed by batch normalization and activation.

    Args:
        x (tensor): the inputs (tensor) to the convolution layer.
        use_batch_norm (bool): use batch norm.
        filters (int): the number of filters.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        activation_type (str): activation function name, e.g., 'relu'.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        data_format (str): either 'channels_last' or 'channels_first'.
        kernel_regularizer (`regularizer`): regularizer for the kernels.
        bias_regularizer (`regularizer`): regularizer for the biases.
        layer_name(str): layer name prefix.
        use_bias(bool): whether or not use bias in convolutional layer.
        quantize (bool): A boolean flag to determine whether to use quantized conv2d or not.
        bitwidth (integer): quantization bitwidth.

    Returns:
        x (tensor): the output tensor of the convolution layer.
    """
    if layer_name is not None:
        layer_name = "%s_m%d" % (layer_name, filters)

    if quantize:
        x = QuantizedConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            data_format=data_format,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=layer_name,
            use_bias=use_bias,
            bitwidth=bitwidth,
        )(x)
    else:
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            data_format=data_format,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=layer_name,
            use_bias=use_bias,
        )(x)
    if use_batch_norm:
        if layer_name is not None:
            layer_name += "_bn"
        x = keras.layers.BatchNormalization(
            axis=get_batchnorm_axis(data_format), name=layer_name
        )(x)
    if activation_type:
        activation_kwargs = activation_kwargs or {}
        x = add_activation(activation_type, **activation_kwargs)(x)
    return x


@add_arg_scope
def deconv2D_bn_activation(
    x,
    use_batch_norm,
    filters,
    kernel_size,
    strides=(1, 1),
    activation_type="relu",
    activation_kwargs=None,
    data_format=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    layer_name=None,
    use_bias=True,
):
    """
    Add a deconv layer, followed by batch normalization and activation.

    Args:
        x (tensor): the inputs (tensor) to the convolution layer.
        use_batch_norm (bool): use batch norm.
        filters (int): the number of filters.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        activation_type (str): activation function name, e.g., 'relu'.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        data_format (str): either 'channels_last' or 'channels_first'.
        kernel_regularizer (`regularizer`): regularizer for the kernels.
        bias_regularizer (`regularizer`): regularizer for the biases.
        layer_name(str): layer name prefix.
        use_bias(bool): whether or not use bias in convolutional layer.

    Returns:
        x (tensor): the output tensor of the convolution layer.
    """
    if layer_name is not None:
        layer_name = "%s_m%d" % (layer_name, filters)

    x = keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        data_format=data_format,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        name=layer_name,
        use_bias=use_bias,
    )(x)
    if use_batch_norm:
        if layer_name is not None:
            layer_name += "_bn"
        x = keras.layers.BatchNormalization(
            axis=get_batchnorm_axis(data_format), name=layer_name
        )(x)
    if activation_type:
        activation_kwargs = activation_kwargs or {}
        x = add_activation(activation_type, **activation_kwargs)(x)
    return x


def add_conv_layer(
    model,
    inputs,
    use_batch_norm,
    filters,
    kernel_size,
    strides=(1, 1),
    activation_type="relu",
    activation_kwargs=None,
    data_format=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    layer_name=None,
    use_bias=True,
    quantize=False,
    bitwidth=8,
):
    """
    Add a conv layer to a model.

    Args:
        model (tensor): the model on top of which the head should be created.
        inputs (tensor): the inputs (tensor) to the previously supplied model.
        use_batch_norm (bool): use batch norm.
        filters (int): the number of filters.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        activation_type (str): activation function name, e.g., 'relu'.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        data_format (str): either 'channels_last' or 'channels_first'.
        kernel_regularizer (`regularizer`): regularizer for the kernels.
        bias_regularizer (`regularizer`): regularizer for the biases.
        layer_name(str): layer name prefix.
        use_bias(bool): whether use bias in convolutional layer.
        quantize (bool): A boolean flag to determine whether to use quantized conv2d or not.
        bitwidth (integer): quantization bitwidth.

    Returns:
        Model: A model with a conv layer stacked on top of the `model` input.
    """
    if data_format is None:
        data_format = keras.backend.image_data_format()
    x = model.outputs[0]
    if layer_name is not None:
        layer_name = "%s_m%d" % (layer_name, filters)

    x = conv2D_bn_activation(
        x,
        use_batch_norm=use_batch_norm,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation_type=activation_type,
        activation_kwargs=activation_kwargs,
        data_format=data_format,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        layer_name=layer_name,
        use_bias=use_bias,
        quantize=quantize,
        bitwidth=bitwidth,
    )
    model = keras.models.Model(inputs=inputs, outputs=x, name="%s_conv" % (model.name))

    return model


def add_conv_head(
    model,
    inputs,
    nmaps,
    kernel_size,
    strides,
    activation_type="sigmoid",
    activation_kwargs=None,
    data_format=None,
    quantize=False,
    bitwidth=8,
):
    """
    Create a model that stacks a convolutional head on top of another model.

    Args:
        model (tensor): the model on top of which the head should be created.
        inputs (tensor): the inputs (tensor) to the previously supplied model.
        nmaps (int): the amount of maps (output filters) the convolution should have.
        kernel_size (int, int): the size of the kernel for this layer.
        strides (int, int): the stride for this layer.
        activation_type (str): the activation function after this layer.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        data_format (str): either 'channels_last' or 'channels_first'.
        quantize (bool): A boolean flag to determine whether to use quantized conv2d or not.
        bitwidth (integer): quantization bitwidth.

    Returns:
        Model: A model with the head stacked on top of the `model` input.
    """
    return add_conv_layer(
        model,
        inputs,
        use_batch_norm=False,
        filters=nmaps,
        kernel_size=kernel_size,
        strides=strides,
        activation_type=activation_type,
        activation_kwargs=activation_kwargs,
        data_format=data_format,
        kernel_regularizer=None,
        bias_regularizer=None,
        layer_name="head_conv",
        quantize=quantize,
        bitwidth=bitwidth,
    )


def add_deconv_layer(
    model,
    inputs,
    use_batch_norm,
    filters,
    upsampling,
    activation_type="relu",
    activation_kwargs=None,
    data_format=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    layer_name=None,
    padding="same",
):
    """
    Add a deconv layer.

    Args:
        model (tensor): the model on top of which the head should be created.
        inputs (tensor): the inputs (tensor) to the previously supplied model.
        use_batch_norm (bool): use batch norm.
        filters (int): the number of filters.
        upsampling (int): the amount of upsampling the transpose convolution should do.
        activation_type (str): activation function name, e.g., 'relu'.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        data_format (str): either 'channels_last' or 'channels_first'.
        kernel_regularizer (`regularizer`): regularizer for the kernels.
        bias_regularizer (`regularizer`): regularizer for the biases.
        layer_name (str): layer_name prefix.

    Returns:
        Model: A model with a deconv layer stacked on top of the `model` input.
    """
    if data_format is None:
        data_format = keras.backend.image_data_format()
    x = model.outputs[0]
    if layer_name is not None:
        layer_name = "%s_m%d_d%d" % (layer_name, filters, upsampling)
    x = keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(upsampling, upsampling),
        strides=(upsampling, upsampling),
        padding=padding,
        data_format=data_format,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        name=layer_name,
    )(x)

    if use_batch_norm:
        if layer_name is not None:
            layer_name += "_bn"
        x = keras.layers.BatchNormalization(
            axis=get_batchnorm_axis(data_format), name=layer_name
        )(x)
    if activation_type:
        activation_kwargs = activation_kwargs or {}
        x = add_activation(activation_type, **activation_kwargs)(x)
    model = keras.models.Model(
        inputs=inputs, outputs=x, name="%s_d%d" % (model.name, upsampling)
    )

    return model


def add_deconv_head(
    model,
    inputs,
    nmaps,
    upsampling,
    activation_type="sigmoid",
    activation_kwargs=None,
    data_format=None,
    padding="same",
):
    """
    Create a model that stacks a deconvolutional (transpose conv) head on top of another model.

    Args:
        model (tensor): the model on top of which the head should be created.
        inputs (tensor): the inputs (tensor) to the previously supplied model.
        nmaps (int): the amount of maps (output filters) the transpose convolution should
            have.
        upsampling (int): the amount of upsampling the transpose convolution should do.
        activation_type (str): activation function name, e.g., 'softmax'.
        activation_kwargs (dict): Additional activation keyword arguments to be fed to
            the add_activation function.
        data_format (str): either 'channels_last' or 'channels_first'.

    Returns:
        Model: A model with the head stacked on top of the `model` input.
    """
    return add_deconv_layer(
        model,
        inputs,
        use_batch_norm=False,
        filters=nmaps,
        upsampling=upsampling,
        activation_type=activation_type,
        activation_kwargs=activation_kwargs,
        data_format=data_format,
        kernel_regularizer=None,
        bias_regularizer=None,
        layer_name="head_deconv",
        padding=padding,
    )


def add_activation(activation_type, **kwargs):
    """
    Create an activation layer based on activation type and additional arguments.

    Note that the needed kwargs depend on the activation type.

    Args:
        activation_type (str): String to indicate activation type.
        kwargs (dict): Additional keyword arguments depending on the activation type.

    Returns:
        activation_layer (a subclass of keras.layers.Layer): The layer type
            depends on activation_type.
    """
    if activation_type == "relu-n":
        max_value = kwargs["max_value"]
        activation_layer = keras.layers.ReLU(max_value=max_value)

    elif activation_type == "lrelu":
        alpha = kwargs["alpha"]
        activation_layer = keras.layers.LeakyReLU(alpha=alpha)
    elif activation_type == "elu":
        alpha = kwargs["alpha"]
        activation_layer = keras.layers.ELU(alpha=alpha)

    else:
        activation_layer = keras.layers.Activation(activation_type, **kwargs)

    return activation_layer


def count_layers_by_class_name(model, class_names):
    """Count the number of layers in a model (recursively) having any of the given class_names."""
    n_layers = 0
    for layer in model.layers:
        if layer.__class__.__name__ in class_names:
            n_layers += 1
        if isinstance(layer, keras.models.Model):
            # The layer is a model: recurse.
            n_layers += count_layers_by_class_name(layer, class_names)
    return n_layers


def clone_model(model, inputs=None, copy_weights=False):
    """
    Clone a model and optionally replace the inputs.

    Args:
        model (Model): The model to clone.
        inputs (list of tensors): The tensor to apply the new model to. If None, the model will
            be returned with placeholders.
        copy_weights (bool): Flag that determines whether the old model's weights should be
            copied into the new one.

    Returns:
        new_model (Model): updated model.
    """
    if inputs is not None:
        # Get all the input placeholders.
        input_placeholders = [
            i
            for i in range(len(model.layers))
            if ("is_placeholder" in dir(model.layers[i]))
            and (model.layers[i].is_placeholder is True)
        ]

        if len(inputs) != len(input_placeholders):
            raise ValueError(
                "Number of model inputs does not match number of given inputs."
            )

        # Rename the input placeholders to avoid name clashes when cloning.
        for placeholder in input_placeholders:
            model.layers[placeholder].name = "input_placeholder_%d" % placeholder
        new_model = keras.models.clone_model(model, inputs)

        # Update the node references in the graph.
        for placeholder in input_placeholders:
            to_remove = [l.name for l in new_model.layers].index(
                "input_placeholder_%d" % placeholder
            )
            to_connect = [
                len(n.inbound_layers)
                for n in new_model.layers[to_remove]._inbound_nodes
            ].index(1)
            new_model.layers[to_remove + 1]._inbound_nodes = []
            new_model.layers[to_remove + 1]._inbound_nodes = [
                new_model.layers[to_remove]._inbound_nodes[to_connect]
            ]

            new_model.layers.remove(new_model.layers[to_remove])

    else:
        new_model = keras.models.clone_model(model)

    if copy_weights:
        new_model.set_weights(model.get_weights())

    return new_model


def update_config(model, config, name_pattern=None, custom_objects=None):
    """
    Update the configuration of an existing model.

    In order to update the configuration of only certain layers,
    a name pattern (regular expression) may be provided.

    Args:
        model (Model): the model to update the regularizers of.
        config (dict): dictionary of layer attributes to update.
        name_pattern (str): pattern to match layers against. Those that
            do not match will not be updated.
        custom_objects (dict): dictionary mapping names (strings) to custom
            classes or functions to be considered during deserialization.
    """
    # Loop through all layers and update those that have a matching config attribute.
    for layer in model.layers:
        if name_pattern is None or re.match(name_pattern, layer.name):
            for name, value in config.items():
                if hasattr(layer, name):
                    setattr(layer, name, value)

    with tempfile.NamedTemporaryFile(delete=True) as f:
        model.save(f.name)
        new_model = keras.models.load_model(
            f.name, custom_objects=custom_objects, compile=False
        )

    return new_model


def update_regularizers(model, kernel_regularizer, bias_regularizer, name_pattern=None):
    """
    Update the weight decay regularizers of an existing model.

    Note that the input tensors to apply the new model to must be different
    from those of the original model. This is because when Keras
    clones a model it retains the original input layer and adds an extra one
    on top.

    In order to update the regularizers of only certain layers,
    a name pattern (regular expression) may be provided.

    Args:
        model (Model): the model to update the regularizers of.
        kernel_regularizer (object): regularizer to apply to kernels.
        bias_regularizer (object): regularizer to apply to biases.
        name_pattern (str): pattern to match layers against. Those that
            do not match will not be updated.
    """
    config = {
        "bias_regularizer": bias_regularizer,
        "kernel_regularizer": kernel_regularizer,
    }
    return update_config(model, config, name_pattern)


def performance_test_model(inputs, data_format=None, activation_type="relu"):
    """Construct a model with 1x1 max pooling with stride 16 for performance diagnostics.

    Args:
        inputs (tensor): The input tensor `x`.
        data_format (string): Either 'channels_last' (NHWC) or 'channels_first' (NCHW).
        activation_type (string): Activation type to use.
    Returns:
        Model: the output model after applying 1x1 max pooling with stride 16 to the input `x`.
    """
    if data_format is None:
        data_format = keras.backend.image_data_format()

    # Create HelNet-0 model which does max pooling with stride 16.
    x = keras.layers.MaxPooling2D(
        pool_size=(1, 1), strides=(16, 16), padding="same", data_format=data_format
    )(inputs)
    x = keras.layers.Activation(activation_type)(x)
    model_name = "helnet0_s16"
    model = keras.models.Model(inputs=inputs, outputs=x, name=model_name)

    return model


class CNNBlock(object):
    """A functor for creating a block of layers."""

    @add_arg_scope
    def __init__(
        self,
        use_batch_norm,
        use_shortcuts,
        data_format,
        kernel_regularizer,
        bias_regularizer,
        repeat,
        stride,
        subblocks,
        index=None,
        activation_type="relu",
        activation_kwargs=None,
        dilation_rate=(1, 1),
        all_projections=False,
        use_bias=True,
        name_prefix=None,
        quantize=False,
        bitwidth=8,
    ):
        """
        Initialization of the block functor object.

        Args:
            use_batch_norm (bool): whether batchnorm should be added after each convolution.
            use_shortcuts (bool): whether shortcuts should be used. A typical ResNet by definition
                uses shortcuts, but these can be toggled off to use the same ResNet topology without
                the shortcuts.
            data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
            kernel_regularizer (float): regularizer to apply to kernels.
            bias_regularizer (float): regularizer to apply to biases.
            repeat (int): repeat number.
            stride (int): The filter stride to be applied only to the first subblock (typically used
                for downsampling). Strides are set to 1 for all layers beyond the first subblock.
            subblocks (list of tuples): A list of tuples defining settings for each consecutive
                convolution. Example:
                    `[(3, 64), (3, 64)]`
                The two items in each tuple represents the kernel size and the amount of filters in
                a convolution, respectively. The convolutions are added in the order of the list.
            index (int): the index of the block to be created.
            activation_type (str): activation function type.
            activation_kwargs (dict): Additional activation keyword arguments to be fed to
                the add_activation function.
            dilation_rate (int or (int, int)): An integer or tuple/list of n integers, specifying
                the dilation rate to use for dilated convolution.
            all_projections (bool): A boolean flag to determinte whether all shortcut connections
                should be implemented as projection layers to facilitate full pruning or not.
            use_bias (bool): whether the layer uses a bias vector.
            name_prefix (str): Prefix the name with this value.
            quantize (bool): A boolean flag to determine whether to use quantized conv2d or not.
            bitwidth (integer): quantization bitwidth.
        """
        self.use_batch_norm = use_batch_norm
        self.use_shortcuts = use_shortcuts
        self.all_projections = all_projections
        self.data_format = data_format
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation_type = activation_type
        self.activation_kwargs = activation_kwargs or {}
        self.dilation_rate = dilation_rate
        self.repeat = repeat
        self.stride = stride
        self.use_bias = use_bias
        self.subblocks = subblocks
        self.subblock_ids = SUBBLOCK_IDS()
        self.quantize = quantize
        self.bitwidth = bitwidth
        if index is not None:
            self.name = "block_%d" % index
        else:
            self.name = "block_%d" % (get_uid("block") + 1)
        if name_prefix is not None:
            self.name = name_prefix + "_" + self.name

    def __call__(self, x):
        """Build the block.

        Args:
            x (tensor): input tensor.

        Returns:
            tensor: the output tensor after applying the block on top of input `x`.
        """
        for i in range(self.repeat):
            name = "%s%s_" % (self.name, self.subblock_ids[i])
            if i == 0:
                # Set the stride only on the first layer.
                stride = self.stride
                dimension_changed = True
            else:
                stride = 1
                dimension_changed = False

            x = self._subblocks(x, stride, dimension_changed, name_prefix=name)

        return x

    def _subblocks(self, x, stride, dimension_changed, name_prefix=None):
        """
        Stack several convolutions in a specific sequence given by a list of subblocks.

        Args:
            x (tensor): the input tensor.
            stride (int): The filter stride to be applied only to the first subblock (typically used
                for downsampling). Strides are set to 1 for all layers beyond the first subblock.
            dimension_changed (bool): This indicates whether the dimension has been changed for this
                block. If this is true, then we need to account for the change, or else we will be
                unable to re-add the shortcut tensor due to incompatible dimensions. This can be
                solved by applying a (1x1) convolution [1]. (The paper also notes the possibility of
                zero-padding the shortcut tensor to match any larger output dimension, but this is
                not implemented.)
            name_prefix (str): name prefix for all the layers created in this function.

        Returns:
            tensor: the output tensor after applying the ResNet block on top of input `x`.
        """
        bn_axis = get_batchnorm_axis(self.data_format)

        shortcut = x
        nblocks = len(self.subblocks)
        for i in range(nblocks):
            kernel_size, filters = self.subblocks[i]
            if i == 0:
                strides = (stride, stride)
            else:
                strides = (1, 1)
            # Keras doesn't support dilation_rate != 1 if stride != 1.
            dilation_rate = self.dilation_rate
            if strides != (1, 1) and dilation_rate != (1, 1):
                dilation_rate = (1, 1)
                logger.warning(
                    "Dilation rate {} is incompatible with stride {}. "
                    "Setting dilation rate to {} for layer {}conv_{}.".format(
                        self.dilation_rate, strides, dilation_rate, name_prefix, i + 1
                    )
                )
            if self.quantize:
                x = QuantizedConv2D(
                    filters,
                    (kernel_size, kernel_size),
                    strides=strides,
                    padding="same",
                    dilation_rate=dilation_rate,
                    data_format=self.data_format,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    bitwidth=self.bitwidth,
                    name="%sconv_%d" % (name_prefix, i + 1),
                )(x)
            else:
                x = keras.layers.Conv2D(
                    filters,
                    (kernel_size, kernel_size),
                    strides=strides,
                    padding="same",
                    dilation_rate=dilation_rate,
                    data_format=self.data_format,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="%sconv_%d" % (name_prefix, i + 1),
                )(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(
                    axis=bn_axis, name="%sbn_%d" % (name_prefix, i + 1)
                )(x)
            if i != nblocks - 1:  # All except last conv in block.
                x = add_activation(self.activation_type, **self.activation_kwargs)(x)

        if self.use_shortcuts:
            if self.all_projections:
                # Implementing shortcut connections as 1x1 projection layers irrespective of
                # dimension change.
                if self.quantize:
                    shortcut = QuantizedConv2D(
                        filters,
                        (1, 1),
                        strides=(stride, stride),
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate,
                        use_bias=self.use_bias,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        bitwidth=self.bitwidth,
                        name="%sconv_shortcut" % name_prefix,
                    )(shortcut)
                else:
                    shortcut = keras.layers.Conv2D(
                        filters,
                        (1, 1),
                        strides=(stride, stride),
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate,
                        use_bias=self.use_bias,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name="%sconv_shortcut" % name_prefix,
                    )(shortcut)
                if self.use_batch_norm:
                    shortcut = keras.layers.BatchNormalization(
                        axis=bn_axis, name="%sbn_shortcut" % name_prefix
                    )(shortcut)
            else:
                # Add projection layers to shortcut only if there is a change in dimesion.
                if dimension_changed:  # Dimension changed.
                    if self.quantize:
                        shortcut = QuantizedConv2D(
                            filters,
                            (1, 1),
                            strides=(stride, stride),
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate,
                            use_bias=self.use_bias,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            bitwidth=self.bitwidth,
                            name="%sconv_shortcut" % name_prefix,
                        )(shortcut)
                    else:
                        shortcut = keras.layers.Conv2D(
                            filters,
                            (1, 1),
                            strides=(stride, stride),
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate,
                            use_bias=self.use_bias,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            name="%sconv_shortcut" % name_prefix,
                        )(shortcut)
                    if self.use_batch_norm:
                        shortcut = keras.layers.BatchNormalization(
                            axis=bn_axis, name="%sbn_shortcut" % name_prefix
                        )(shortcut)

            x = keras.layers.add([x, shortcut])

        x = add_activation(self.activation_type, **self.activation_kwargs)(x)

        return x
