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

"""Utils for onnx export."""

import logging

import tensorflow.compat.v1 as tf
import tf2onnx

logger = logging.getLogger(__name__)


def pb_to_onnx(
    input_filename,
    output_filename,
    input_node_names,
    output_node_names,
    target_opset=12,
    verbose=False
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
    graphdef = tf.GraphDef()
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
    logger.info(
        "Input node names: {input_node_names}.".format(
            input_node_names=input_node_names
        )
    )
    logger.info(
        "Output node names: {output_node_names}.".format(
            output_node_names=output_node_names
        )
    )

    tf.reset_default_graph()
    # `tf2onnx.tfonnx.process_tf_graph` prints out layer names when
    # folding the layers. Disabling INFO logging for TLT branch.
    # logging.getLogger("tf2onnx.tfonnx").setLevel(logging.WARNING)

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graphdef, name="")

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            tf_graph,
            input_names=input_node_names,
            output_names=output_node_names,
            continue_on_error=True,
            verbose=verbose,
            opset=target_opset,
        )
        onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)
        model_proto = onnx_graph.make_model("test")
        with open(output_filename, "wb") as f:
            f.write(model_proto.SerializeToString())

    # Reload and check ONNX model.
    # Temporary disabling the load onnx section.
    # onnx_model = onnx.load(output_filename)
    # onnx.checker.check_model(onnx_model)

    # Return a string instead of a list if there is only one input or output.
    if len(input_node_names) == 1:
        input_node_names = input_node_names[0]

    if len(output_node_names) == 1:
        output_node_names = output_node_names[0]

    return input_node_names, output_node_names
