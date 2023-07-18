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
"""
converter_functions.py

Conversion Functions for common layers.
Add new functions here with a decorator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from uff.converters.tensorflow.converter import TensorFlowToUFFConverter as tf2uff
from uff.model.utils import convert_to_str
from uff.model.exceptions import *  # noqa pylint: disable = W0401,W0614

import numpy as np


@tf2uff.register(["Placeholder"])
def convert_placeholder(name, tf_node, inputs, uff_graph, **kwargs):
    dtype = tf2uff.convert_tf2numpy_dtype(tf_node.attr['dtype'].type)
    shape = tf2uff.get_tf_shape_as_int_list(tf_node.attr['shape'])
    uff_graph.input(shape, dtype, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Identity"])
def convert_identity(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.identity(inputs[0], name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Const"])
def convert_const(name, tf_node, inputs, uff_graph, **kwargs):
    array = tf2uff.convert_tf2numpy_const_node(tf_node)
    uff_node = uff_graph.const(array, name)
    uff_node.array = array
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Add"])
def convert_add(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.binary(inputs[0], inputs[1], 'add', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Sub"])
def convert_sub(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.binary(inputs[0], inputs[1], 'sub', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Mul"])
def convert_mul(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.binary(inputs[0], inputs[1], 'mul', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Div", "RealDiv"])
def convert_div(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.binary(inputs[0], inputs[1], 'div', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Relu"])
def convert_relu(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.activation(inputs[0], 'relu', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Relu6"])
def convert_relu6(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.activation(inputs[0], 'relu6', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["LeakyRelu"])
def convert_leaky_relu(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.leaky_relu(inputs[0], tf_node.attr['alpha'].f, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Tanh"])
def convert_tanh(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.activation(inputs[0], 'tanh', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Sigmoid"])
def convert_sigmoid(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.activation(inputs[0], 'sigmoid', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Elu"])
def convert_elu(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.activation(inputs[0], 'elu', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Selu"])
def convert_selu(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.activation(inputs[0], 'selu', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Softsign"])
def convert_softsign(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.activation(inputs[0], 'softsign', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Softplus"])
def convert_softplus(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.activation(inputs[0], 'softplus', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Neg"])
def convert_neg(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'neg', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Abs"])
def convert_abs(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'abs', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Acos"])
def convert_acos(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'acos', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Acosh"])
def convert_acosh(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'acosh', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Asin"])
def convert_asin(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'asin', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Asinh"])
def convert_asinh(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'asinh', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Atan"])
def convert_atan(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'atan', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Atanh"])
def convert_atanh(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'atanh', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Ceil"])
def convert_ceil(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'ceil', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Cos"])
def convert_cos(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'cos', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Cosh"])
def convert_cosh(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'cosh', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Sin"])
def convert_sin(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'sin', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Sinh"])
def convert_sinh(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'sinh', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Tan"])
def convert_tan(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'tan', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Floor"])
def convert_floor(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'floor', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Sqrt"])
def convert_sqrt(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'sqrt', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Rsqrt"])
def convert_rsqrt(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'rsqrt', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Square"])
def convert_square(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'square', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Pow"])
def convert_pow(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'pow', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Exp"])
def convert_exp(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'exp', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Log"])
def convert_log(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.unary(inputs[0], 'log', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Softmax"])
def convert_softmax(name, tf_node, inputs, uff_graph, **kwargs):
    # Some Softmax ops don't have an axis node.
    if len(inputs) > 1:
        tf_axis_node = kwargs["tf_nodes"][inputs[-1]]
        axis = int(tf2uff.convert_tf2numpy_const_node(tf_axis_node))
        inputs = inputs[:-1]
    else:
        axis = 0
    fmt = convert_to_str(tf_node.attr['data_format'].s)
    fmt = fmt if fmt else "NCHW"
    data_fmt = tf2uff.convert_tf2uff_data_format(fmt)
    uff_graph.softmax(inputs[0], axis, data_fmt, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Minimum"])
def convert_minimum(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.binary(inputs[0], inputs[1], 'min', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Maximum"])
def convert_maximum(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.binary(inputs[0], inputs[1], 'max', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Shape"])
def convert_shape(name, tf_node, inputs, uff_graph, **kwargs):
    uff_graph.shape(inputs[0], name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["ExpandDims"])
def convert_expand_dims(name, tf_node, inputs, uff_graph, **kwargs):
    # Retrieve and remove the axis node.
    tf_axis_node = kwargs["tf_nodes"][inputs[-1]]
    if tf_axis_node.op != "Const":
        raise UffException("ExpandDims Axis node has op " + str(tf_axis_node.op) + ", expected Const. The axis must be specified as a Const node.")
    axis = int(tf2uff.convert_tf2numpy_const_node(tf_axis_node))
    inputs.pop(-1)
    # Add the op.
    uff_graph.expand_dims(inputs[0], axis, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["ArgMax"])
def convert_argmax(name, tf_node, inputs, uff_graph, **kwargs):
    # Retrieve and remove the axis node.
    tf_axis_input_node = kwargs["tf_nodes"][inputs[-1]]
    if tf_axis_input_node.op != "Const":
        raise UffException("ArgMax Axis node has op " + str(tf_axis_input_node.op) + ", expected Const. The axis must be specified as a Const node.")
    axis = int(tf2uff.convert_tf2numpy_const_node(tf_axis_input_node))
    inputs.pop(-1)
    # Add the op.
    uff_graph.argmax(inputs[0], axis, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["ArgMin"])
def convert_argmin(name, tf_node, inputs, uff_graph, **kwargs):
    # Retrieve and remove the axis node.
    tf_axis_input_node = kwargs["tf_nodes"][inputs[-1]]
    if tf_axis_input_node.op != "Const":
        raise UffException("ArgMin Axis node has op " + str(tf_axis_input_node.op) + ", expected Const. The axis must be specified as a Const node.")
    axis = int(tf2uff.convert_tf2numpy_const_node(tf_axis_input_node))
    inputs.pop(-1)
    # Add the op.
    uff_graph.argmin(inputs[0], axis, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Reshape"])
def convert_reshape(name, tf_node, inputs, uff_graph, **kwargs):
    str_name = tf_node.name.split('/')
    if len(str_name) > 1 and tf_node.name.split('/')[-2].lower().find('flatten') != -1:
        print('DEBUG: convert reshape to flatten node')
        uff_graph.flatten(inputs[0], name=name)  # flatten axis is ignored here
        return [tf2uff.split_node_name_and_output(inputs[0])[0]]  # second input of shape is dropped
    uff_graph.reshape(inputs[0], inputs[1], name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

# tensorflow does not have flatten op.
#  tensorflow.contrib.slim has a flatten function that combines slice/shape to
#  implement flatten. 'We' decided to hack it through chopping the reshape and
#  slice, to add a flatten op. So it's easier to patch it with uff/TensorRT.
#
# @tf2uff.register(["Flatten"])
# def _flatten_helper(name, tf_node, inputs, uff_graph, **kwargs):
#    axis = tf2uff.get_tf_int_list(tf_node.attr['axis'])
#    uff_graph.flatten(inputs[0], name=name, axis=axis)
#    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Transpose"])
def convert_transpose(name, tf_node, inputs, uff_graph, **kwargs):
    tf_permutation_node = kwargs["tf_nodes"][inputs[1]]
    if tf_permutation_node.op != "Const":
        raise UffException("Transpose permutation has op " + str(tf_permutation_node.op) + ", expected Const. Only constant permuations are supported in UFF.")
    permutation = tf2uff.convert_tf2numpy_const_node(
        tf_permutation_node).tolist()
    inputs = inputs[:1]
    uff_graph.transpose(inputs[0], permutation, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Pack"])
def convert_pack(name, tf_node, inputs, uff_graph, **kwargs):
    axis = tf_node.attr['axis'].i
    uff_graph.stack(inputs, axis, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["ConcatV2"])
def convert_concatv2(name, tf_node, inputs, uff_graph, **kwargs):
    if "axis" in tf_node.attr:
        # Handle cases where the axis is not a node, but an attribute instead.
        axis = tf_node.attr["axis"].i
    else:
        tf_axis_node = kwargs["tf_nodes"][inputs[-1]]
        if tf_axis_node.op != "Const":
            raise UffException("Concat Axis node has op " + str(tf_axis_node.op) + ", expected Const. The axis for a Concat op must be specified as either an attribute, or a Const node.")
        axis = int(tf2uff.convert_tf2numpy_const_node(tf_axis_node))
        inputs = inputs[:-1]
    uff_graph.concat(inputs, axis, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["MaxPool"])
def convert_maxpool(name, tf_node, inputs, uff_graph, **kwargs):
    return _pool_helper(name, tf_node, inputs, uff_graph, func='max', **kwargs)


@tf2uff.register(["AvgPool"])
def convert_avgpool(name, tf_node, inputs, uff_graph, **kwargs):
    return _pool_helper(name, tf_node, inputs, uff_graph, func='avg', **kwargs)


def _pool_helper(name, tf_node, inputs, uff_graph, **kwargs):
    func = kwargs["func"]
    window_size = tf2uff.get_tf_int_list(tf_node.attr['ksize'])
    strides = tf2uff.get_tf_int_list(tf_node.attr['strides'])
    fmt = convert_to_str(tf_node.attr['data_format'].s)
    fmt = fmt if fmt else "NHWC"
    inputs, padding, fields = tf2uff.apply_fused_padding(
        tf_node, inputs, kwargs["tf_nodes"])
    data_format = tf2uff.convert_tf2uff_data_format(fmt)
    if fmt == 'NCHW':
        window_size = window_size[2:]
        strides = strides[2:]
        if padding is not None:
            padding = padding[2:]
    elif fmt == 'NHWC':
        window_size = [window_size[1], window_size[2]]
        strides = [strides[1], strides[2]]
        if padding is not None:
            padding = [padding[1], padding[2]]
    else:
        raise ValueError("Unsupported data format: " + fmt)
    uff_graph.pool(
        inputs[0], func, window_size, strides, padding,
        data_format=data_format, name=name, fields=fields)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["LRN"])
def convert_lrn(name, tf_node, inputs, uff_graph, **kwargs):
    lhs = inputs[0]
    fmt = convert_to_str(tf_node.attr['data_format'].s)
    fmt = fmt if fmt else "NC+"
    window_size = tf_node.attr["depth_radius"].i
    alpha = tf_node.attr["alpha"].f
    beta = tf_node.attr["beta"].f
    bias = tf_node.attr["bias"].f
    uff_graph.lrn(lhs, window_size, alpha, beta, bias, fmt, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["MatMul"])
def convert_matmul(name, tf_node, inputs, uff_graph, **kwargs):
    lhs, rhs = inputs
    trans_a = tf_node.attr['transpose_a'].b
    trans_b = tf_node.attr['transpose_b'].b
    lhs_fmt = 'CN' if trans_a else 'NC'
    rhs_fmt = 'KC' if trans_b else 'CK'
    uff_graph.fully_connected(
        lhs, rhs, lhs_fmt, rhs_fmt, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Conv2D"])
def convert_conv2d(name, tf_node, inputs, uff_graph, **kwargs):
    return _conv2d_helper(name, tf_node, inputs, uff_graph, func="conv2d", **kwargs)


@tf2uff.register(["DepthwiseConv2dNative"])
def convert_depthwise_conv2d_native(name, tf_node, inputs, uff_graph,
                                    **kwargs):
    return _conv2d_helper(name, tf_node, inputs, uff_graph, func="depthwise", **kwargs)


def _conv2d_helper(name, tf_node, inputs, uff_graph, **kwargs):
    func = kwargs["func"]
    fmt = convert_to_str(tf_node.attr['data_format'].s)
    fmt = fmt if fmt else "NHWC"

    strides = tf2uff.get_tf_int_list(tf_node.attr['strides'])
    inputs, padding, fields = tf2uff.apply_fused_padding(
        tf_node, inputs, kwargs["tf_nodes"])
    lhs_fmt = tf2uff.convert_tf2uff_data_format(fmt)
    rhs_fmt = '+CK'
    if fmt == 'NCHW':
        strides = strides[2:]
        if padding is not None:
            padding = padding[2:]
    elif fmt == 'NHWC':
        strides = [strides[1], strides[2]]
        if padding is not None:
            padding = [padding[1], padding[2]]
    else:
        raise ValueError("Unsupported data format: " + fmt)
    if func == "depthwise":
        wt = kwargs["tf_nodes"][inputs[1]]
        number_groups = int(wt.attr['value'].tensor.tensor_shape.dim[2].size)
    else:
        number_groups = None
    # If this node represents a dilated conv, pull in the dilations.
    dilation = None
    if "dilations" in tf_node.attr:
        if fmt == "NCHW":
            dilation = tf2uff.get_tf_int_list(tf_node.attr['dilations'])[2:]
        else:
            dilation = tf2uff.get_tf_int_list(tf_node.attr['dilations'])[1:3]

    # FIXME: Need a better way to check for dilated convs. This just checks if the block_shape input is as expected.
    # Ideally we should have a 'get_input_by_name' function. Maybe we can leverage GS here.
    # Another possibility is that GS can add these as attributes to the node rather than maintaining them as
    # separate const nodes.
    tf_block_shape_node = kwargs["tf_nodes"][inputs[1]]
    if "block_shape" in tf_block_shape_node.name.split('/')[-1] and tf_block_shape_node.op == "Const":
        # Get the second input (block_shape) - of the form [1, dilation_value, dilation_value]
        dilation = np.frombuffer(tf_block_shape_node.attr["value"].tensor.tensor_content, dtype=np.int32).tolist()
        if len(dilation) > 2:
            dilation = [dilation[1], dilation[2]]
        inputs.pop(1)

    tf_paddings_node = kwargs["tf_nodes"][inputs[1]]
    if "paddings" in tf_paddings_node.name.split('/')[-1] and tf_paddings_node.op == "Const":
        # Get the second input (paddings, since block_shape is already removed)
        paddings_temp = np.frombuffer(tf_paddings_node.attr["value"].tensor.tensor_content, dtype=np.int32).tolist()
        inputs.pop(1)

        # Get cropping information, but only if paddings is also present.
        tf_crops_node = kwargs["tf_nodes"][inputs[1]]
        if "crops" in tf_crops_node.name.split('/')[-1] and tf_crops_node.op == "Const":
            # Get the second input (crops, since block_shape is already removed)
            crops = np.frombuffer(tf_crops_node.attr["value"].tensor.tensor_content, dtype=np.int32)
            inputs.pop(1)
            paddings_temp = (np.array(paddings_temp) - crops).tolist()

        # TF paddings are [[top,bottom], [left,right]], so we need to rearrange.
        perm = [0, 2, 1, 3]
        # HACK: Sometimes paddings has [0, 0] at the front.
        if len(paddings_temp) == 6:
            paddings_temp = paddings_temp[2:]
        paddings_temp = [paddings_temp[p] for p in perm]
        # Symmetric padding ("same")
        if paddings_temp[0] == paddings_temp[2] and paddings_temp[1] == paddings_temp[3]:
            paddings_temp = paddings_temp[0:2]
            padding = paddings_temp if not padding else [p + pt for p, pt in zip(padding, paddings_temp)]
        else:
            print("Asymmetric padding for dilated convolutions is currently unsupported in the UFF converter.")

    uff_graph.conv(
        inputs[0], inputs[-1], strides, padding,
        dilation=dilation, number_groups=number_groups,
        left_format=lhs_fmt, right_format=rhs_fmt,
        name=name, fields=fields)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Conv2DBackpropInput"])
def convert_conv2d_backprop_input(name, tf_node, inputs, uff_graph, **kwargs):
    return _conv2d_transpose_helper(name, tf_node, inputs, uff_graph,
                                    func="conv2d_transpose", **kwargs)


def _conv2d_transpose_helper(name, tf_node, inputs, uff_graph, **kwargs):
    kwargs.pop("func")  # FIXME support depthwise transpose
    fmt = convert_to_str(tf_node.attr['data_format'].s)
    fmt = fmt if fmt else "NHWC"
    strides = tf2uff.get_tf_int_list(tf_node.attr['strides'])

    fields = {}
    padding = None
    number_groups = None

    tf_padding = convert_to_str(tf_node.attr['padding'].s)
    if tf_padding == "SAME":
        fields['implicit_padding'] = "same"
    elif tf_padding != "VALID":
        raise ValueError("Padding mode %s not supported" % tf_padding)

    lhs_fmt = tf2uff.convert_tf2uff_data_format(fmt)
    rhs_fmt = '+KC'

    if fmt == 'NCHW':
        strides = strides[2:]
    elif fmt == 'NHWC':
        strides = [strides[1], strides[2]]
    else:
        raise ValueError("Unsupported data format: " + fmt)

    uff_graph.conv_transpose(
        inputs[2], inputs[1], inputs[0],
        strides, padding,
        dilation=None, number_groups=number_groups,
        left_format=lhs_fmt, right_format=rhs_fmt,
        name=name, fields=fields)

    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["BiasAdd"])
def convert_bias_add(name, tf_node, inputs, uff_graph, **kwargs):
    fmt = convert_to_str(tf_node.attr['data_format'].s)
    fmt = fmt if fmt else "NHWC"
    biases_name = inputs[1]
    biases_array = tf2uff.convert_tf2numpy_const_node(
        kwargs["tf_nodes"][biases_name])
    inputs = inputs[:1]
    if fmt == 'NCHW':
        ndim = 4
        new_shape = [-1] + [1] * (ndim - 2)
        biases_array = biases_array.reshape(new_shape)
    uff_graph.const(biases_array, biases_name)
    uff_graph.binary(inputs[0], biases_name, 'add', name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["FusedBatchNorm"])
def convert_fused_batch_norm(name, tf_node, inputs, uff_graph, **kwargs):
    input_node, gamma, beta, mean, variance = inputs
    eps = tf_node.attr['epsilon'].f
    fmt = convert_to_str(tf_node.attr['data_format'].s)
    fmt = fmt if fmt else "NHWC"
    data_fmt = tf2uff.convert_tf2uff_data_format(fmt)
    uff_graph.batchnorm(input_node, gamma, beta, mean,
                        variance, eps, data_fmt, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["StridedSlice"])
def convert_strided_slice(name, tf_node, inputs, uff_graph, **kwargs):
    begin_mask = tf_node.attr['begin_mask'].i
    end_mask = tf_node.attr['end_mask'].i
    shrink_axis_mask = tf_node.attr['shrink_axis_mask'].i

    if tf_node.attr['ellipsis_mask'].i != 0:
        raise ValueError("ellipsis_mask not supported")

    if tf_node.attr['new_axis_mask'].i != 0:
        raise ValueError("new_axis_mask not supported")

    uff_graph.strided_slice(inputs[0], inputs[1], inputs[2], inputs[3],
                            begin_mask, end_mask, shrink_axis_mask, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


def _reduce_helper(name, tf_node, inputs, uff_graph, **kwargs):
    func = kwargs.pop("func")

    tf_axes_node = kwargs["tf_nodes"][inputs[1]]
    array = tf2uff.convert_tf2numpy_const_node(tf_axes_node)
    axes = array.tolist()
    inputs = inputs[:1]
    keepdims = tf_node.attr['keep_dims'].b

    print("Warning: keepdims is ignored by the UFF Parser and defaults to True")

    uff_graph.reduce(inputs[0], func, axes, keepdims, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


@tf2uff.register(["Sum"])
def convert_sum(name, tf_node, inputs, uff_graph, **kwargs):
    return _reduce_helper(name, tf_node, inputs, uff_graph, func="sum", **kwargs)


@tf2uff.register(["Prod"])
def convert_prod(name, tf_node, inputs, uff_graph, **kwargs):
    return _reduce_helper(name, tf_node, inputs, uff_graph, func="prod", **kwargs)


@tf2uff.register(["Min"])
def convert_min(name, tf_node, inputs, uff_graph, **kwargs):
    return _reduce_helper(name, tf_node, inputs, uff_graph, func="min", **kwargs)


@tf2uff.register(["Max"])
def convert_max(name, tf_node, inputs, uff_graph, **kwargs):
    return _reduce_helper(name, tf_node, inputs, uff_graph, func="max", **kwargs)


@tf2uff.register(["Mean"])
def convert_mean(name, tf_node, inputs, uff_graph, **kwargs):
    return _reduce_helper(name, tf_node, inputs, uff_graph, func="mean", **kwargs)


@tf2uff.register(["Squeeze"])
def convert_squeeze(name, tf_node, inputs, uff_graph, **kwargs):
    axis = tf2uff.get_tf_int_list(tf_node.attr['squeeze_dims'])
    uff_graph.squeeze(inputs[0], name=name, axis=axis)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]


# TODO: add attributes of MODE / constant_values
@tf2uff.register(["Pad"])
def convert_pad(name, tf_node, inputs, uff_graph, **kwargs):
    pad = inputs[1]
    uff_graph.pad(inputs[0], pad, name)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["Gather"])
def convert_gather(name, tf_node, inputs, uff_graph, **kwargs):
    indices_dtype = tf2uff.convert_tf2numpy_dtype(tf_node.attr['Tindices'].type)
    params_dtype = tf2uff.convert_tf2numpy_dtype(tf_node.attr['Tparams'].type)
    validate_indices = tf_node.attr['validate_indices'].b
    uff_graph.gather(inputs, name, indices_dtype, params_dtype, validate_indices)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["GatherV2"])
def convert_gather_v2(name, tf_node, inputs, uff_graph, **kwargs):
    if len(inputs) > 2:
        tf_axis_node = kwargs["tf_nodes"][inputs[-1]]
        axis = int(tf2uff.convert_tf2numpy_const_node(tf_axis_node))
        inputs = inputs[:-1]
    else:
        axis = 0
    indices_dtype = tf2uff.convert_tf2numpy_dtype(tf_node.attr['Tindices'].type)
    params_dtype = tf2uff.convert_tf2numpy_dtype(tf_node.attr['Tparams'].type)
    uff_graph.gather_v2(inputs, name, axis, indices_dtype, params_dtype)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

@tf2uff.register(["ResourceGather"])
def convert_resource_gather(name, tf_node, inputs, uff_graph, **kwargs):
    if len(inputs) > 2:
        tf_axis_node = kwargs["tf_nodes"][inputs[-1]]
        axis = int(tf2uff.convert_tf2numpy_const_node(tf_axis_node))
        inputs = inputs[:-1]
    else:
        axis = 0
    indices_dtype = tf2uff.convert_tf2numpy_dtype(tf_node.attr['Tindices'].type)
    params_dtype = tf2uff.convert_tf2numpy_dtype(tf_node.attr['dtype'].type)
    uff_graph.gather_v2(inputs, name, axis, indices_dtype, params_dtype)
    return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]
