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
"""Simple Stand-alone inference script for LPRNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import cv2
import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.lprnet.models import eval_builder
from nvidia_tao_tf1.cv.lprnet.utils.ctc_decoder import decode_ctc_conf
from nvidia_tao_tf1.cv.lprnet.utils.img_utils import preprocess
from nvidia_tao_tf1.cv.lprnet.utils.model_io import load_model
from nvidia_tao_tf1.cv.lprnet.utils.spec_loader import (
    INFERENCE_EXP_REQUIRED_MSG,
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
            description='LPRNet Inference Tool'
        )

    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='Path to a TLT model or a TensorRT engine')
    parser.add_argument('-i',
                        '--image_dir',
                        required=True,
                        type=str,
                        help='The path to input image or directory.')
    parser.add_argument('-k',
                        '--key',
                        default="",
                        type=str,
                        help='Key to save or load a .tlt model. Must present if -m is a TLT model')
    parser.add_argument('-e',
                        '--experiment_spec',
                        required=True,
                        type=str,
                        help='Path to an experiment spec file for training.')
    parser.add_argument('--trt',
                        action='store_true',
                        help='Use TensorRT engine for inference.')
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


def inference(arguments):
    '''make inference.'''
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
        message="Starting LPRNet Inference."
    )

    config_path = arguments.experiment_spec

    experiment_spec = load_experiment_spec(config_path)

    spec_validator(experiment_spec, INFERENCE_EXP_REQUIRED_MSG)

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

    image_file_list = [os.path.join(arguments.image_dir, file_name)
                       for file_name in os.listdir(arguments.image_dir)]
    batch_cnt = int(len(image_file_list) / batch_size)

    for idx in range(batch_cnt):
        # prepare data:
        batch_image_list = image_file_list[idx * batch_size: (idx + 1) * batch_size]
        batch_images = [cv2.imread(image_file) for image_file in batch_image_list]

        batch_images = preprocess(batch_images,
                                  output_width=output_width,
                                  output_height=output_height,
                                  is_training=False)
        # predict:
        if arguments.trt:
            prediction = trt_inf.infer_batch(batch_images)
        else:
            prediction = model.predict(x=batch_images, batch_size=batch_size)

        # decode prediction
        decoded_lp, _ = decode_ctc_conf(prediction,
                                        classes=classes,
                                        blank_id=blank_id)

        for image_name, decoded_lp in zip(batch_image_list, decoded_lp):
            print("{}:{} ".format(image_name, decoded_lp))

    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Inference finished successfully."
    )


def main(args=None):
    """Run the inference process."""
    try:
        args = parse_command_line(args)
        inference(args)
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


if __name__ == "__main__":
    main()
