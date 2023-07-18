from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import keras
import pytest

from nvidia_tao_tf1.core.models.templates.utils import count_layers_by_class_name
from nvidia_tao_tf1.core.templates.mobilenet import MobileNet, MobileNetV2

MOBILENET_NUM_CONV_LAYERS = [14, 14]
MOBILENET_NUM_DENSE_LAYERS = [0, 1]
MOBILENET_NUM_DEPTHWISE_LAYERS = [13, 13]

MOBILENETV2_NUM_CONV_LAYERS = [35, 35]
MOBILENETV2_NUM_DENSE_LAYERS = [0, 1]
MOBILENETV2_NUM_DEPTHWISE_LAYERS = [17, 17]

topologies = [
    # Test the different nlayers
    (256, True, 'channels_first', True, 32),
    (256, False, 'channels_first', True, 32),
    (256, False, 'channels_first', False, 32),

    (256, True, 'channels_last', True, 16),
    (256, False, 'channels_last', True, 16),
    (256, False, 'channels_last', False, 16),

    (512, True, 'channels_first', True, 32),
    (512, False, 'channels_first', True, 32),
    (512, False, 'channels_first', False, 32),

    (512, True, 'channels_last', True, 16),
    (512, False, 'channels_last', True, 16),
    (512, False, 'channels_last', False, 16),

    (224, True, 'channels_first', True, 32),
    (224, False, 'channels_first', True, 32),
    (224, False, 'channels_first', False, 32),

    (224, True, 'channels_last', True, 16),
    (224, False, 'channels_last', True, 16),
    (224, False, 'channels_last', False, 16),
]


def _compute_output_size(size, stride):
    for _ in range(4):
        size = int(math.ceil(size / 2.0))
    if stride == 32:
        size = int(math.ceil(size / 2.0))
    return size


@pytest.mark.parametrize("input_size, use_batch_norm, data_format, add_head, stride", topologies)
def test_mobilenet_v1(input_size, use_batch_norm, data_format, add_head, stride, nclasses=None):
    """Test MobileNet V1 for a variety of instantiation parameter combinations."""
    # Set channel format.
    if data_format == 'channels_last':
        shape = (input_size, input_size, 3)
    elif data_format == 'channels_first':
        shape = (3, input_size, input_size)

    # Define a keras input layer for the network
    inputs = keras.layers.Input(shape=shape)

    # Add 10 class dense head if needed.
    if add_head:
        nclasses = 10

    # Instantiate model.
    model = MobileNet(inputs,
                      use_batch_norm=use_batch_norm,
                      data_format=data_format,
                      add_head=add_head,
                      stride=stride,
                      activation_type='relu',
                      nclasses=nclasses)

    # Batchnorm check.
    n_batchnorms = count_layers_by_class_name(model, ["BatchNormalization"])
    if use_batch_norm:
        assert n_batchnorms > 0
    else:
        assert n_batchnorms == 0

    # Layer count check.
    n_conv_layers_counted = count_layers_by_class_name(model, ["Conv2D"])
    n_dense_layers_counted = count_layers_by_class_name(model, ["Dense"])
    n_depthiwise_conv_2d_layers_counted = count_layers_by_class_name(model, ['DepthwiseConv2D'])

    # Setting expected number of conv layers.
    if stride == 32:
        if add_head:
            expected_conv_layers = MOBILENET_NUM_CONV_LAYERS[1]
            expected_dense_layers = MOBILENET_NUM_DENSE_LAYERS[1]
            expected_depthwise_conv_2d_layers = MOBILENET_NUM_DEPTHWISE_LAYERS[1]
        else:
            expected_conv_layers = MOBILENET_NUM_CONV_LAYERS[0]
            expected_dense_layers = MOBILENET_NUM_DENSE_LAYERS[0]
            expected_depthwise_conv_2d_layers = MOBILENET_NUM_DEPTHWISE_LAYERS[0]
        # Check number of layers in the instantiated model.
        assert n_dense_layers_counted == expected_dense_layers
        assert n_conv_layers_counted == expected_conv_layers
        assert n_depthiwise_conv_2d_layers_counted == expected_depthwise_conv_2d_layers

    # Check model output shape.
    output_shape = tuple(model.outputs[0].get_shape().as_list())

    # Set expected shape depending on whether or not pruning is set.
    if add_head:
        assert output_shape[1:] == (nclasses,)
    else:
        _output_sized_expected = _compute_output_size(input_size, stride)
        expected_shape = (_output_sized_expected, _output_sized_expected)
        if data_format == 'channels_last':
            assert output_shape[1:3] == expected_shape
        elif data_format == 'channels_first':
            assert output_shape[2:4] == expected_shape

    # Check the name of the instantiated model.
    assert "mobilenet" in model.name
    if use_batch_norm:
        assert "_bn" in model.name


@pytest.mark.parametrize("input_size, use_batch_norm, data_format, add_head, stride", topologies)
def test_mobilenet_v2(input_size, use_batch_norm, data_format, add_head, stride, nclasses=None):
    """Test MobileNet V2 for a variety of instantiation parameter combinations."""
    # Set channel format.
    if data_format == 'channels_last':
        shape = (input_size, input_size, 3)
    elif data_format == 'channels_first':
        shape = (3, input_size, input_size)

    # Define a keras input layer for the network
    inputs = keras.layers.Input(shape=shape)

    # Add 10 class dense head if needed.
    if add_head:
        nclasses = 10

    # Instantiate model.
    model = MobileNetV2(inputs,
                        use_batch_norm=use_batch_norm,
                        data_format=data_format,
                        add_head=add_head,
                        stride=stride,
                        activation_type='relu',
                        nclasses=nclasses)

    # Batchnorm check.
    n_batchnorms = count_layers_by_class_name(model, ["BatchNormalization"])
    if use_batch_norm:
        assert n_batchnorms > 0
    else:
        assert n_batchnorms == 0

    # Layer count check.
    n_conv_layers_counted = count_layers_by_class_name(model, ["Conv2D"])
    n_dense_layers_counted = count_layers_by_class_name(model, ["Dense"])
    n_depthiwise_conv_2d_layers_counted = count_layers_by_class_name(model, ['DepthwiseConv2D'])

    # Setting expected number of conv layers.
    if stride == 32:
        if add_head:
            expected_conv_layers = MOBILENETV2_NUM_CONV_LAYERS[1]
            expected_dense_layers = MOBILENETV2_NUM_DENSE_LAYERS[1]
            expected_depthwise_conv_2d_layers = MOBILENETV2_NUM_DEPTHWISE_LAYERS[1]
        else:
            expected_conv_layers = MOBILENETV2_NUM_CONV_LAYERS[0]
            expected_dense_layers = MOBILENETV2_NUM_DENSE_LAYERS[0]
            expected_depthwise_conv_2d_layers = MOBILENETV2_NUM_DEPTHWISE_LAYERS[0]
        # Check number of layers in the instantiated model.
        assert n_dense_layers_counted == expected_dense_layers
        assert n_conv_layers_counted == expected_conv_layers
        assert n_depthiwise_conv_2d_layers_counted == expected_depthwise_conv_2d_layers

    # Check model output shape.
    output_shape = tuple(model.outputs[0].get_shape().as_list())

    # Set expected shape depending on whether or not pruning is set.
    if add_head:
        assert output_shape[1:] == (nclasses,)
    else:
        _output_sized_expected = _compute_output_size(input_size, stride)
        expected_shape = (_output_sized_expected, _output_sized_expected)
        if data_format == 'channels_last':
            assert output_shape[1:3] == expected_shape
        elif data_format == 'channels_first':
            assert output_shape[2:4] == expected_shape

    # Check the name of the instantiated model.
    assert "mobilenet_v2" in model.name
    if use_batch_norm:
        assert "_bn" in model.name
