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
"""Simple Stand-alone inference script for YOLO models trained using modulus."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import keras.backend as K
import numpy as np

from nvidia_tao_tf1.cv.common.inferencer.inferencer import Inferencer
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import check_tf_oom

from nvidia_tao_tf1.cv.yolo_v3.builders import eval_builder
from nvidia_tao_tf1.cv.yolo_v3.utils.model_io import load_model
from nvidia_tao_tf1.cv.yolo_v3.utils.spec_loader import load_experiment_spec

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']


def build_command_line_parser(parser=None):
    '''build parser.'''
    if parser is None:
        parser = argparse.ArgumentParser(description='TLT YOLOv3 Inference Tool')
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='Path to a TLT model or TensorRT engine.')
    parser.add_argument('-i',
                        '--image_dir',
                        required=True,
                        type=str,
                        help='The path to input image or directory.')
    parser.add_argument('-k',
                        '--key',
                        type=str,
                        default="",
                        help='Key to save or load a .tlt model. Must present if -m is a TLT model')
    parser.add_argument('-e',
                        '--experiment_spec',
                        required=True,
                        type=str,
                        help='Path to an experiment spec file for training.')
    parser.add_argument('-t',
                        '--threshold',
                        type=float,
                        default=0.3,
                        help='Confidence threshold for inference.')
    parser.add_argument("-r",
                        '--results_dir',
                        type=str,
                        default=None,
                        help='Path to the files where the logs are stored.')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=1,
                        help=argparse.SUPPRESS)
    return parser


def parse_command_line(args):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def keras_output_process_fn(inferencer, y_encoded):
    "function to process keras model output."
    # xmin
    y_encoded[..., -4] = y_encoded[..., -4] * inferencer.model_input_width
    # ymin
    y_encoded[..., -3] = y_encoded[..., -3] * inferencer.model_input_height
    # xmax
    y_encoded[..., -2] = y_encoded[..., -2] * inferencer.model_input_width
    # ymax
    y_encoded[..., -1] = y_encoded[..., -1] * inferencer.model_input_height
    return y_encoded


def trt_output_process_fn(inferencer, y_encoded):
    "function to process TRT model output."
    keep_k, boxes, scores, cls_id = y_encoded
    result = []
    for idx, k in enumerate(keep_k.reshape(-1)):
        mul = np.array([[inferencer.model_input_width,
                         inferencer.model_input_height,
                         inferencer.model_input_width,
                         inferencer.model_input_height]])
        loc = boxes[idx].reshape(-1, 4)[:k] * mul
        cid = cls_id[idx].reshape(-1, 1)[:k]
        conf = scores[idx].reshape(-1, 1)[:k]
        result.append(np.concatenate((cid, conf, loc), axis=-1))
    return result


def inference(arguments):
    '''make inference on a folder of images.'''
    # Set up status logging
    if arguments.results_dir:
        if not os.path.exists(arguments.results_dir):
            os.makedirs(arguments.results_dir)
        status_file = os.path.join(arguments.results_dir, "status.json")
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
            message="Starting YOLOv3 inference."
        )
    config_path = arguments.experiment_spec
    experiment_spec = load_experiment_spec(config_path)
    K.clear_session()  # Clear previous models from memory.
    K.set_learning_phase(0)
    classes = sorted({str(x).lower() for x in
                      experiment_spec.dataset_config.target_class_mapping.values()})
    class_mapping = dict(zip(range(len(classes)), classes))
    img_mean = experiment_spec.augmentation_config.image_mean
    if experiment_spec.augmentation_config.output_channel == 3:
        if img_mean:
            img_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
        else:
            img_mean = [103.939, 116.779, 123.68]
    else:
        if img_mean:
            img_mean = [img_mean['l']]
        else:
            img_mean = [117.3786]

    if os.path.splitext(arguments.model_path)[1] in ['.tlt', '.hdf5']:
        img_height = experiment_spec.augmentation_config.output_height
        img_width = experiment_spec.augmentation_config.output_width
        n_channels = experiment_spec.augmentation_config.output_channel
        model = load_model(arguments.model_path, experiment_spec,
                           (n_channels, img_height, img_width),
                           key=arguments.key)
        # Load evaluation parameters
        conf_th = experiment_spec.nms_config.confidence_threshold
        iou_th = experiment_spec.nms_config.clustering_iou_threshold
        top_k = experiment_spec.nms_config.top_k
        nms_on_cpu = True
        # Build evaluation model
        model = eval_builder.build(
            model, conf_th, iou_th, top_k, nms_on_cpu=nms_on_cpu
        )
        inferencer = Inferencer(keras_model=model,
                                batch_size=experiment_spec.eval_config.batch_size,
                                infer_process_fn=keras_output_process_fn,
                                class_mapping=class_mapping,
                                img_mean=img_mean,
                                threshold=arguments.threshold)
        print("Using TLT model for inference, setting batch size to the one in eval_config:",
              experiment_spec.eval_config.batch_size)
    else:
        inferencer = Inferencer(trt_engine_path=arguments.model_path,
                                infer_process_fn=trt_output_process_fn,
                                class_mapping=class_mapping,
                                img_mean=img_mean,
                                threshold=arguments.threshold,
                                batch_size=experiment_spec.eval_config.batch_size)

        print("Using TensorRT engine for inference, setting batch size to engine's one:",
              inferencer.batch_size)

    out_image_path = os.path.join(arguments.results_dir, "images_annotated")
    out_label_path = os.path.join(arguments.results_dir, "labels")
    os.makedirs(out_image_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)

    inferencer.infer(arguments.image_dir, out_image_path, out_label_path)
    if arguments.results_dir:
        s_logger.write(
            status_level=status_logging.Status.SUCCESS,
            message="Inference finished successfully."
        )


@check_tf_oom
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
