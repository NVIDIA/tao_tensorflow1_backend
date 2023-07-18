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

"""Simple script to generate templates and save these models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf

import graphsurgeon as gs
from keras import backend as K

from nvidia_tao_tf1.core.export import (
    keras_to_pb,
    pb_to_uff,
    UFFEngineBuilder
)

from nvidia_tao_tf1.cv.makenet.utils.helper import model_io

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

import tensorrt as trt

TENSORRT_LOGGER = trt.Logger(trt.Logger.Severity.VERBOSE)


def main(cl_args=None):
    """Simple function to get tensorrt engine and UFF."""
    args = parse_command_line(args=cl_args)

    model_path = args['input_model']
    key = str(args['key'])
    output_pb = None
    if args['output_pb'] is not None:
        output_pb = args['output_pb']
    else:
        output_pb = "{}.pb".format(os.path.splitext(model_path)[0])
    output_node_names = args['output_node_names'].split(',')
    output_uff = "{}.uff".format(os.path.splitext(model_path)[0])

    keras_model = model_io(model_path=model_path, enc_key=key)
    keras_model.summary()
    input_shape = list(keras_model.inputs[0].shape[1:])
    input_dims = {"input_1": input_shape}

    print("Converting keras to pb model.")
    nvidia_tao_tf1.core.export.keras_to_pb(keras_model, output_pb, output_node_names)
    print("Inspecting pb model.")
    inspect_pb(output_pb)
    nvidia_tao_tf1.core.export._uff.pb_to_uff(output_pb, output_uff, output_node_names)
    # Getting engine with TensorRT
    UFFEngineBuilder(output_uff, "input_1", input_dims,
                     output_node_names, verbose=True, max_batch_size=16,
                     max_workspace_size=1<<30)

def inspect_pb(output_pb):
    sg = gs.StaticGraph(output_pb)
    for node in sg:
        if "bn_conv1" in node.name:
            print('Node: {}'.format(node.name))


def parse_command_line(args=None):
    """Parse command-line flags passed to the training script.

    Returns:
      Namespace with all parsed arguments.
    """
    parser = argparse.ArgumentParser(prog='train',
                                     description='Train a classification model.')

    parser.add_argument(
        '-i',
        '--input_model',
        type=str,
        required=True,
        help='Path to the experiment spec file.')
    parser.add_argument(
        '--output_node_names',
        type=str,
        default="predictions/Softmax",
        help='Names of output nodes.'
    )
    parser.add_argument(
        '-o',
        '--output_pb',
        type=str,
        default=None,
        help='Path to a folder where experiment outputs should be written.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        default="",
        help='Decryption key.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Include this flag in command line invocation for verbose logs.'
    )
    return vars(parser.parse_args(args))


if __name__=="__main__":
    main()
