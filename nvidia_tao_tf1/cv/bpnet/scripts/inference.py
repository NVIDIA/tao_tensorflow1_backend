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

"""BpNet inference script."""

import argparse
import logging
import os

from nvidia_tao_tf1.cv.bpnet.inferencer.bpnet_inferencer import BpNetInferencer
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.utilities.path_processing as io_utils


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism.

    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='infer', description='Run BpNet inference.')

    parser.add_argument(
        '-i',
        '--inference_spec',
        type=str,
        default='nvidia_tao_tf1/cv/bpnet/experiment_specs/infer_default.yaml',
        help='Path to inference spec.'
    )

    parser.add_argument(
        '-m',
        '--model_filename',
        type=str,
        required=False,
        default=None,
        help='Path to model file to use for inference. If None, \
            the model path from the inference spec will be used.'
    )

    parser.add_argument(
        '--input_type',
        default="image",
        type=str,
        choices=["image", "dir", "json"],
        help='Input type that you want to specify.'
    )

    parser.add_argument(
        '--input',
        default=None,
        type=str,
        help='Path to image / dir / json to run inference on.'
    )

    parser.add_argument(
        '--image_root_path',
        default='',
        type=str,
        help='Root dir path to image(s). If specified, \
            image paths are assumed to be relative to this.'
    )

    parser.add_argument(
        '-k',
        '--key',
        default="",
        type=str,
        required=False,
        help="The API key to decrypt the model."
    )

    parser.add_argument(
        '--results_dir',
        default=None,
        type=str,
        help='Results directory for inferences. Inference result visualizations \
            will be dumped in this directory if --dump_visualizations is set.',
    )

    parser.add_argument(
        '--dump_visualizations',
        action='store_true',
        default=False,
        help='If enabled, saves images with inference visualization to \
            `results/images_annotated` directory.'
    )

    parser.add_argument(
        '-ll',
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level.')

    return parser


def parse_command_line_args(cl_args=None):
    """Parser command line arguments to the trainer.

    Args:
        cl_args(sys.argv[1:]): Arg from the command line.

    Returns:
        args: Parsed arguments using argparse.
    """
    parser = build_command_line_parser(parser=None)
    args = parser.parse_args(cl_args)
    return args


def main(cl_args=None):
    """Launch the model inference process."""

    args = parse_command_line_args(cl_args)
    enc_key = args.key
    inference_spec_path = args.inference_spec
    model_filename = args.model_filename
    input_type = args.input_type
    _input = args.input
    image_root_path = args.image_root_path
    results_dir = args.results_dir
    dump_visualizations = args.dump_visualizations

    # Make results dir if it doesn't already exist
    if results_dir:
        io_utils.mkdir_p(results_dir)

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
        message="Starting inference."
    )

    # Init logging
    # tf.logging.set_verbosity(tf.logging.INFO)
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s')
    logger = logging.getLogger(__name__)
    logger_tf = logging.getLogger('tensorflow')
    logger.setLevel(args.log_level)
    logger_tf.setLevel(args.log_level)

    # Read input
    if input_type == "image":
        logger.info("Reading from image: {}".format(_input))
        io_utils.check_file(_input)
        data = [_input]
    elif input_type == "dir":
        io_utils.check_dir(_input)
        logger.info("Reading from directory: {}".format(_input))
        data = os.listdir(_input)
        image_root_path = _input
    elif input_type == "json":
        io_utils.check_file(_input)
        logger.info("Reading from json file: {}".format(_input))
        data = io_utils.load_json_file(_input)
    else:
        raise ValueError("Unsupported input type: {}".format(input_type))

    # Load inference spec file
    inference_spec = io_utils.load_yaml_file(inference_spec_path)
    # Load inference spec file
    experiment_spec = io_utils.load_yaml_file(inference_spec['train_spec'])

    # Enforce results_dir value if dump_visualizations is true.
    if dump_visualizations and not results_dir:
        raise ValueError("--results_dir must be specified if dumping visualizations.")

    # Load model
    if model_filename is None:
        model_full_path = inference_spec['model_path']
        logger.warning("No model provided! Using model_path from inference spec file.")
    else:
        model_full_path = model_filename
    logger.info("Loading {} for inference.".format(model_full_path))
    # logger.info(model.summary()) # Disabled for TLT

    # Initialize BpNetInferencer
    inferencer = BpNetInferencer(
        model_full_path,
        inference_spec,
        experiment_spec,
        key=enc_key
    )

    # Run inference
    inferencer.run(
        data,
        results_dir=results_dir,
        image_root_path=image_root_path,
        dump_visualizations=dump_visualizations
    )


if __name__ == "__main__":
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Inference finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
