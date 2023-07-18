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
"""Export trained FPENet Keras model to UFF format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import graphsurgeon as gs
import keras
import uff

from nvidia_tao_tf1.core.export._uff import keras_to_pb
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p
from nvidia_tao_tf1.cv.fpenet.models.custom.softargmax import Softargmax


def parse_command_line(arg_list):
    """
    Parse command-line flags passed to the prune script.

    Args:
        arg_list: List of strings used as command-line arguments. If None,
        sys.argv is used by default.

    Returns:
      Namespace with members for all parsed arguments.
    """
    parser = argparse.ArgumentParser(prog='export', description='Export a FPENet model.')
    parser.add_argument(
        '-m',
        '--model_filename',
        type=str,
        required=True,
        help="Absolute path to the model file to export."
    )
    parser.add_argument(
        '-o',
        '--output_filename',
        type=str,
        required=True,
        help='Name of the exported model')
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default=None,
        help='Path to a folder where experiment outputs will be created, or specify in spec file.')
    args_parsed = parser.parse_args(arg_list)

    return args_parsed


def main(cl_args=None):
    """Run exporting."""
    args_parsed = parse_command_line(cl_args)

    results_dir = args_parsed.results_dir

    if results_dir:
        mkdir_p(results_dir)

    # Writing out status file for TAO.
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=1,
            append=True
        )
    )

    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.STARTED,
        message="Starting export."
    )

    model_name = args_parsed.model_filename
    output_model_name = args_parsed.output_filename

    # Load Keras model from file.
    model = keras.models.load_model(model_name,
                                    custom_objects={'Softargmax': Softargmax},
                                    compile=False)
    model.load_weights(model_name)
    model.summary()

    output_node_names = ['softargmax/strided_slice', 'softargmax/strided_slice_1']

    # Create froxen graph as .pb file.
    output_pb_filename = model_name + '.pb'
    _, out_tensor_names, __ = keras_to_pb(model,
                                          output_pb_filename,
                                          output_node_names=output_node_names,
                                          custom_objects={'Softargmax': Softargmax})

    # Add custom plugin nodes.
    SoftargmaxLayer = gs.create_plugin_node(name='softargmax', op='Softargmax')
    UpSamplingLayer_1 = gs.create_plugin_node(name='up_sampling2d_1', op='UpsamplingNearest')
    UpSamplingLayer_2 = gs.create_plugin_node(name='up_sampling2d_2', op='UpsamplingNearest')
    UpSamplingLayer_3 = gs.create_plugin_node(name='up_sampling2d_3', op='UpsamplingNearest')
    UpSamplingLayer_4 = gs.create_plugin_node(name='up_sampling2d_4', op='UpsamplingNearest')

    namespace_plugin_map = {'softargmax' : SoftargmaxLayer,
                            'up_sampling2d_1' : UpSamplingLayer_1,
                            'up_sampling2d_2' : UpSamplingLayer_2,
                            'up_sampling2d_3' : UpSamplingLayer_3,
                            'up_sampling2d_4' : UpSamplingLayer_4}

    dynamic_graph = gs.DynamicGraph(output_pb_filename)
    dynamic_graph.collapse_namespaces(namespace_plugin_map)

    out_tensor_names = ['softargmax']

    # Convert to UFF from dynamic graph.
    uff.from_tensorflow(dynamic_graph.as_graph_def(),
                        out_tensor_names,
                        output_filename=os.path.join(results_dir, output_model_name),
                        list_nodes=False,
                        debug_mode=False,
                        return_graph_info=False)


if __name__ == "__main__":
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Export finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Export was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
