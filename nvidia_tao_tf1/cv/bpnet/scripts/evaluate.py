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

"""BpNet evaluation script."""

import argparse
import logging
import os

from nvidia_tao_tf1.cv.bpnet.dataio.coco_dataset import COCODataset
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
        parser = argparse.ArgumentParser(prog='evaluate', description='Run BpNet evaluation.')

    parser.add_argument(
        '-i',
        '--inference_spec',
        type=str,
        default='nvidia_tao_tf1/cv/bpnet/experiment_specs/infer_default.yaml',
        help='Path to model file to evaluate.'
    )

    parser.add_argument(
        '-m',
        '--model_filename',
        type=str,
        required=False,
        default=None,
        help='Path to model file to use for evaluation. If None, \
            the model path from the inference spec will be used.'
    )

    parser.add_argument(
        '--dataset',
        default="coco",
        type=str,
        choices=["coco"],
        help='Dataset to run evaluation on.'
    )

    parser.add_argument(
        '-d',
        '--dataset_spec',
        required=True,
        help='Path to the dataset spec.'
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
        default='/workspace/tlt-experiments/bpnet/',
        type=str,
        help='Results directory',
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
    """Launch the model evaluation process."""

    args = parse_command_line_args(cl_args)
    enc_key = args.key
    inference_spec_path = args.inference_spec
    model_filename = args.model_filename
    dataset_spec = args.dataset_spec
    results_dir = args.results_dir

    # Init logging
    # tf.logging.set_verbosity(tf.logging.INFO)
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s')
    logger = logging.getLogger(__name__)
    logger_tf = logging.getLogger('tensorflow')
    logger.setLevel(args.log_level)
    logger_tf.setLevel(args.log_level)

    # Load dataset_spec file
    dataset_spec = io_utils.load_json_file(dataset_spec)
    # Load inference spec file
    inference_spec = io_utils.load_yaml_file(inference_spec_path)
    # Load inference spec file
    experiment_spec = io_utils.load_yaml_file(inference_spec['train_spec'])
    # Make results dir if it doesn't already exist
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
        message="Starting Evaluation."
    )

    # Load model
    if model_filename is None:
        model_full_path = inference_spec['model_path']
        logger.warning("No model provided! Using model_path from inference spec file.")
    else:
        model_full_path = model_filename
    logger.info("Loading {} for evaluation".format(model_full_path))
    # logger.info(model.summary()) # Disabled for TLT

    # Initialize COCODataset
    dataset = COCODataset(dataset_spec)

    # Run inference
    detections_path = dataset.infer(
        model_full_path,
        inference_spec,
        experiment_spec,
        results_dir,
        key=enc_key
    )

    # Run evaluation
    eval_results = COCODataset.evaluate(dataset.test_coco, detections_path, results_dir).stats
    eval_dict = {"AP_0.50:0.95_all": eval_results[0],
                 "AP_0.5": eval_results[1],
                 "AP_0.75": eval_results[2],
                 "AP_0.50:0.95_medium": eval_results[3],
                 "AP_0.50:0.95_large": eval_results[4],
                 "AR_0.50:0.95_all": eval_results[5],
                 "AR_0.5": eval_results[6],
                 "AR_0.75": eval_results[7],
                 "AR_0.50:0.95_medium": eval_results[8],
                 "AR_0.50:0.95_large": eval_results[9]
                 }
    status_logging.get_status_logger().kpi = eval_dict
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.RUNNING,
        message="Evaluation metrics generated."
    )


if __name__ == "__main__":
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Evaluation finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
