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
"""Perform Evaluation of the trained models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from multiprocessing import cpu_count
import os

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
import numpy as np

import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import check_tf_oom

from nvidia_tao_tf1.cv.multitask_classification.data_loader.data_generator import (
    MultiClassDataGenerator
)
from nvidia_tao_tf1.cv.multitask_classification.utils.model_io import load_model
from nvidia_tao_tf1.cv.multitask_classification.utils.spec_loader import load_experiment_spec


def build_command_line_parser(parser=None):
    """Build a command line parser for eval."""
    if parser is None:
        parser = argparse.ArgumentParser(description="TLT MultiTask Classification Evaluator")

    parser.add_argument("--model_path",
                        "-m",
                        type=str,
                        required=True,
                        help="Path to TLT model file")
    parser.add_argument("--experiment_spec",
                        "-e",
                        type=str,
                        required=True,
                        help="Path to experiment spec file")
    parser.add_argument("--key",
                        "-k",
                        type=str,
                        default="",
                        help="TLT model key")
    parser.add_argument("-r",
                        "--results_dir",
                        type=str,
                        default=None,
                        help="Path to results directory")
    # Dummy arguments for Deploy
    parser.add_argument('-i',
                        '--image_dir',
                        type=str,
                        required=False,
                        default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=1,
                        help=argparse.SUPPRESS)

    return parser


def parse_command_line_arguments(args=None):
    """Parse command line arguments for eval."""
    parser = build_command_line_parser()
    return vars(parser.parse_known_args(args)[0])


@check_tf_oom
def evaluate(model_file, img_root, target_csv, key, batch_size):
    """Wrapper function for evaluating MClassification application.

    Args:
       Dictionary arguments containing parameters defined by command line parameters

    """
    s_logger = status_logging.get_status_logger()
    s_logger.write(
        status_level=status_logging.Status.STARTED,
        message="Starting evaluation."
    )
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    K.set_learning_phase(0)

    model = load_model(model_file, key=key)

    # extracting the data format parameter to detect input shape
    data_format = model.layers[1].data_format

    # Computing shape of input tensor
    image_shape = model.layers[0].input_shape[1:4]

    # Setting input shape
    if data_format == "channels_first":
        image_height, image_width = image_shape[1:3]
    else:
        image_height, image_width = image_shape[0:2]

    target_datagen = MultiClassDataGenerator(preprocessing_function=preprocess_input,
                                             horizontal_flip=False
                                             )
    # Initializing data iterator: Val
    target_iterator = target_datagen.flow_from_singledirectory(img_root,
                                                               target_csv,
                                                               target_size=(image_height,
                                                                            image_width),
                                                               batch_size=batch_size)

    print('Processing dataset (evaluation): {}'.format(target_csv))
    nclasses_list = list(target_iterator.class_dict.values())
    assert all(np.array(nclasses_list) > 0), "Invalid target dataset."

    # If number of classes does not match the new data
    assert np.sum(nclasses_list) == \
        np.sum([l.get_shape().as_list()[-1] for l in model.output]), \
        "The number of classes of the loaded model doesn't match the target dataset."

    # Printing summary of retrieved model
    model.summary()

    # Evaluate the model on the full data set.
    score = model.evaluate_generator(target_iterator,
                                     len(target_iterator),
                                     workers=cpu_count() - 1)
    print('Total Val Loss:', score[0])
    print('Tasks:', target_iterator.tasks_header)
    print('Val loss per task:', score[1:1 + target_iterator.num_tasks])
    print('Val acc per task:', score[1 + target_iterator.num_tasks:])
    # Write val accuracy per task into kpi
    tasks = target_iterator.tasks_header
    val_accuracies = score[1 + target_iterator.num_tasks:]
    kpi_dict = {key: float(value) for key, value in zip(tasks, val_accuracies)}
    kpi_dict["mean accuracy"] = sum(val_accuracies) / len(val_accuracies)
    s_logger.kpi.update(kpi_dict)

    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Evalation finished successfully."
    )


if __name__ == '__main__':
    args = parse_command_line_arguments()
    experiment_spec = load_experiment_spec(args['experiment_spec'])
    if args["results_dir"]:
        if not os.path.exists(args["results_dir"]):
            os.makedirs(args["results_dir"])
        status_file = os.path.join(args["results_dir"], "status.json")
        status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=True,
                verbosity=1,
                append=True
            )
        )
    try:
        evaluate(args['model_path'],
                 experiment_spec.dataset_config.image_directory_path,
                 experiment_spec.dataset_config.val_csv_path,
                 args['key'],
                 experiment_spec.training_config.batch_size_per_gpu)
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
