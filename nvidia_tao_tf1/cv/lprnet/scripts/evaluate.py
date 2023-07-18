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
"""Simple Stand-alone evaluate script for LPRNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

import tensorflow as tf
from tqdm import trange
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.lprnet.dataloader.data_sequence import LPRNetDataGenerator
from nvidia_tao_tf1.cv.lprnet.models import eval_builder
from nvidia_tao_tf1.cv.lprnet.utils.ctc_decoder import decode_ctc_conf
from nvidia_tao_tf1.cv.lprnet.utils.model_io import load_model
from nvidia_tao_tf1.cv.lprnet.utils.spec_loader import (
    EVAL_EXP_REQUIRED_MSG,
    load_experiment_spec,
    spec_validator
)


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')


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
            description='Evaluate a LPRNet model.'
        )

    parser.add_argument('-m',
                        '--model_path',
                        help='Path to an TLT model or TensorRT engine.',
                        required=True,
                        type=str)
    parser.add_argument('-k',
                        '--key',
                        default="",
                        type=str,
                        help='Key to save or load a .tlt model.')
    parser.add_argument('-e',
                        '--experiment_spec',
                        required=True,
                        type=str,
                        help='Experiment spec file for training and evaluation.')
    parser.add_argument('--trt',
                        action='store_true',
                        help='Use TensorRT engine for evaluation.')
    parser.add_argument('-r',
                        '--results_dir',
                        type=str,
                        help='Path to a folder where the logs are stored.')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=1,
                        help=argparse.SUPPRESS)
    return parser


def parse_command_line(args):
    '''Parse command line arguments.'''
    parser = build_command_line_parser()
    return parser.parse_args(args)


def evaluate(arguments):
    '''make evaluation.'''
    results_dir = arguments.results_dir
    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        log_dir = results_dir
    else:
        log_dir = os.path.dirname(arguments.model_path)

    status_file = os.path.join(log_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=1,
            append=True
        )
    )

    s_logger = status_logging.get_status_logger()
    s_logger.write(
        status_level=status_logging.Status.STARTED,
        message="Starting LPRNet evaluation."
    )
    config_path = arguments.experiment_spec

    experiment_spec = load_experiment_spec(config_path)

    spec_validator(experiment_spec, EVAL_EXP_REQUIRED_MSG)

    characters_list_file = experiment_spec.dataset_config.characters_list_file
    with open(characters_list_file, "r") as f:
        temp_list = f.readlines()
    classes = [i.strip() for i in temp_list]
    blank_id = len(classes)

    output_width = experiment_spec.augmentation_config.output_width
    output_height = experiment_spec.augmentation_config.output_height
    output_channel = experiment_spec.augmentation_config.output_channel
    batch_size = experiment_spec.eval_config.batch_size
    input_shape = (batch_size, output_channel, output_height, output_width)

    if os.path.splitext(arguments.model_path)[1] in ['.tlt', '.hdf5']:
        tf.keras.backend.clear_session()  # Clear previous models from memory.
        tf.keras.backend.set_learning_phase(0)
        model = load_model(model_path=arguments.model_path,
                           max_label_length=experiment_spec.lpr_config.max_label_length,
                           key=arguments.key)

        # Build evaluation model
        model = eval_builder.build(model)

        print("Using TLT model for inference, setting batch size to the one in eval_config:",
              experiment_spec.eval_config.batch_size)
    elif arguments.trt:

        from nvidia_tao_tf1.cv.common.inferencer.trt_inferencer import TRTInferencer
        trt_inf = TRTInferencer(arguments.model_path, input_shape)

        print("Using TRT engine for inference, setting batch size to the one in eval_config:",
              experiment_spec.eval_config.batch_size)
    else:
        print("Unsupported model type: {}".format(os.path.splitext(arguments.model_path)[1]))
        sys.exit()

    batch_size = experiment_spec.eval_config.batch_size
    val_data = LPRNetDataGenerator(experiment_spec=experiment_spec,
                                   is_training=False,
                                   shuffle=False)

    tr = trange(len(val_data), file=sys.stdout)
    tr.set_description('Producing predictions')

    total_cnt = val_data.n_samples
    correct = 0
    for idx in tr:
        # prepare data:
        batch_x, batch_y = val_data[idx]
        # predict:
        if arguments.trt:
            prediction = trt_inf.infer_batch(batch_x)
        else:
            prediction = model.predict(x=batch_x, batch_size=batch_size)

        # decode prediction
        decoded_lp, _ = decode_ctc_conf(prediction,
                                        classes=classes,
                                        blank_id=blank_id)

        for idx, lp in enumerate(decoded_lp):
            if lp == batch_y[idx]:
                correct += 1

    acc = float(correct)/float(total_cnt)
    print("Accuracy: {} / {}  {}".format(correct, total_cnt,
                                         acc))
    s_logger.kpi.update({'Accuracy': acc})
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Evaluation finished successfully."
    )


def main(args=None):
    """Run the evaluation process."""
    try:
        args = parse_command_line(args)
        evaluate(args)
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


if __name__ == "__main__":
    main()
