# Copyright (c) 2017 - 2019, NVIDIA CORPORATION.  All rights reserved.
"""Simple standalone script to evaluate a gridbox model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.build_evaluator import (
    build_evaluator_for_trained_gridbox
)
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import get_base_model_config
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.detectnet_v2.training.utilities import (
    get_singular_monitored_session,
    setup_keras_backend
)
from nvidia_tao_tf1.cv.detectnet_v2.utilities.timer import time_function

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism.

    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog='evaluate', description='Evaluate a DetectNet_v2 model.'
        )

    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Absolute path to a single file containing a complete Experiment prototxt.')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        help='Path to the .tlt model file or tensorrt engine file under evaluation.',
        required=True)
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Include this flag in command line invocation for verbose logs.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default=None,
        help="Path to report the mAP and logs file."
    )
    parser.add_argument(
        '--use_training_set',
        action='store_true',
        help='Set this flag to evaluate over entire tfrecord and not just validation fold or '
             'the validation data source mentioned in the spec file.'
    )
    parser.add_argument(
        '-k',
        '--key',
        required=False,
        help="Key to load the tlt model.",
        default=""
    )
    parser.add_argument(
        '-f',
        '--framework',
        help="The backend framework to be used.",
        choices=["tlt", "tensorrt"],
        default="tlt"
    )
    # Dummy arguments for Deploy
    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=False,
        default=None,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '-l',
        '--label_dir',
        type=str,
        required=False,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=1,
        help=argparse.SUPPRESS
    )
    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments.

    Args:
        args (list): List of strings used as command line arguments.
            If None, sys.argv is used.

    Returns:
        args_parsed: Parsed arguments.
    """
    parser = build_command_line_parser()
    args_parsed = parser.parse_args(args)
    return args_parsed


@time_function(__name__)
def main(cl_args=None):
    """
    Prepare and run gridbox evaluation process.

    Args:
        cl_args (list): list of strings used as command-line arguments to the script.
            If None (default), arguments will be parsed from sys.argv.
    Raises:
        IOError if the specified experiment spec file doesn't exist.
    """
    args_parsed = parse_command_line(cl_args)

    # Setting logger configuration
    verbosity = 'INFO'
    verbose = args_parsed.verbose
    if verbose:
        verbosity = "DEBUG"

    # Configure logging to get Maglev log messages.
    logging.basicConfig(format='%(asctime)s [%(levelname)s] '
                               '%(name)s: %(message)s',
                        level=verbosity)

    # Defining the results directory.
    results_dir = args_parsed.results_dir
    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        status_file = os.path.join(results_dir, "status.json")
        status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=True,
                append=False,
                verbosity=logger.getEffectiveLevel()
            )
        )
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.STARTED,
        message="Starting DetectNet_v2 Evaluation"
    )

    # Check that the experiment spec file exists and parse it.
    experiment_spec_file = args_parsed.experiment_spec

    if not os.path.exists(experiment_spec_file):
        raise IOError("The specified experiment file doesn't exist: %s" %
                      experiment_spec_file)

    logger.debug("Setting up experiment from experiment specs.")
    experiment_spec = load_experiment_spec(
        experiment_spec_file, merge_from_default=False,
        validation_schema="train_val")

    # Extract core model config, which might be wrapped inside a TemporalModelConfig.
    model_config = get_base_model_config(experiment_spec)

    # Set up Keras backend with correct computation precision and learning phase.
    setup_keras_backend(model_config.training_precision, is_training=False)

    # Expand and validate model file argument.
    model_path = args_parsed.model_path

    # Build dashnet evaluator engine for keras models.
    use_training_set = args_parsed.use_training_set

    logger.debug("Constructing evaluator.")
    framework = args_parsed.framework
    evaluator = build_evaluator_for_trained_gridbox(experiment_spec=experiment_spec,
                                                    model_path=model_path,
                                                    use_training_set=use_training_set,
                                                    use_confidence_models=False,
                                                    key=args_parsed.key,
                                                    framework=framework)

    # Run validation.
    logger.debug("Running evaluation session.")
    with get_singular_monitored_session(evaluator.keras_models,
                                        session_config=evaluator.get_session_config()) as session:
        metrics_results, validation_cost, median_inference_time = \
            evaluator.evaluate(session.raw_session())

    evaluator.print_metrics(
        metrics_results, validation_cost, median_inference_time)

    logger.info("Evaluation complete.")


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
        if type(e) == tf.errors.ResourceExhaustedError:
            logger.error(
                "Ran out of GPU memory, please lower the batch size, use a smaller input "
                "resolution, or use a smaller backbone."
            )
            status_logging.get_status_logger().write(
                message="Ran out of GPU memory, please lower the batch size, use a smaller input "
                        "resolution, or use a smaller backbone.",
                verbosity_level=status_logging.Verbosity.INFO,
                status_level=status_logging.Status.FAILURE
            )
            exit(1)
        else:
            # throw out the error as-is if they are not OOM error
            status_logging.get_status_logger().write(
                message=str(e),
                status_level=status_logging.Status.FAILURE
            )
            raise e
