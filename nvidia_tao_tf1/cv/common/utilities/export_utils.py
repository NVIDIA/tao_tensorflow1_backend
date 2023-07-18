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
"""Export utils."""

import onnx
import tensorflow as tf
from tensorflow.compat.v1 import GraphDef
import tf2onnx

from nvidia_tao_tf1.core.export._uff import keras_to_pb


def pb_to_onnx(
    input_filename,
    output_filename,
    input_node_names,
    output_node_names,
    target_opset=None,
):
    """Convert a TensorFlow model to ONNX.

    The input model needs to be passed as a frozen Protobuf file.
    The exported ONNX model may be parsed and optimized by TensorRT.

    Args:
        input_filename (str): path to protobuf file.
        output_filename (str): file to write exported model to.
        input_node_names (list of str): list of model input node names as
            returned by model.layers[some_idx].get_output_at(0).name.split(':')[0].
        output_node_names (list of str): list of model output node names as
            returned by model.layers[some_idx].get_output_at(0).name.split(':')[0].
        target_opset (int): Target opset version to use, default=<default opset for
            the current keras2onnx installation>
    Returns:
        tuple<in_tensor_name(s), out_tensor_name(s):
        in_tensor_name(s): The name(s) of the input nodes. If there is only one name, it will be
                           returned as a single string, otherwise a list of strings.
        out_tensor_name(s): The name(s) of the output nodes. If there is only one name, it will be
                            returned as a single string, otherwise a list of strings.
    """
    graphdef = GraphDef()
    with tf.gfile.GFile(input_filename, "rb") as frozen_pb:
        graphdef.ParseFromString(frozen_pb.read())

    if not isinstance(input_node_names, list):
        input_node_names = [input_node_names]
    if not isinstance(output_node_names, list):
        output_node_names = [output_node_names]

    # The ONNX parser requires tensors to be passed in the node_name:port_id format.
    # Since we reset the graph below, we assume input and output nodes have a single port.
    input_node_names = ["{}:0".format(node_name) for node_name in input_node_names]
    output_node_names = ["{}:0".format(node_name) for node_name in output_node_names]

    tf.reset_default_graph()
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graphdef, name="")

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            tf_graph,
            input_names=input_node_names,
            output_names=output_node_names,
            continue_on_error=True,
            verbose=True,
            opset=target_opset,
        )
        onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)
        model_proto = onnx_graph.make_model("test")
        with open(output_filename, "wb") as f:
            f.write(model_proto.SerializeToString())

    # Reload and check ONNX model.
    onnx_model = onnx.load(output_filename)
    onnx.checker.check_model(onnx_model)

    # Return a string instead of a list if there is only one input or output.
    if len(input_node_names) == 1:
        input_node_names = input_node_names[0]

    if len(output_node_names) == 1:
        output_node_names = output_node_names[0]

    return input_node_names, output_node_names


def convertKeras2TFONNX(
    model,
    model_name,
    output_node_names=None,
    target_opset=10,
    custom_objects=None,
    logger=None
):
    """Convert keras model to onnx via frozen tensorflow graph.

    Args:
        model (keras.model.Model): Decoded keras model to be exported.
        model_name (str): name of the model file
        output_node_names (str): name of the output node
        target_opset (int): target opset version
    """
    # replace input shape of first layer
    output_pb_filename = model_name + '.pb'
    in_tensor_names, out_tensor_names, __ = keras_to_pb(
        model,
        output_pb_filename,
        output_node_names,
        custom_objects=custom_objects)

    if logger:
        logger.info('Output Tensors: {}'.format(out_tensor_names))
        logger.info('Input Tensors: {}'.format(in_tensor_names))

    output_onnx_filename = model_name + '.onnx'
    (_, _) = pb_to_onnx(output_pb_filename,
                        output_onnx_filename,
                        in_tensor_names,
                        out_tensor_names,
                        target_opset)
