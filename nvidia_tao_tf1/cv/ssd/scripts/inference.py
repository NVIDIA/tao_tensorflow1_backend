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
"""Simple Stand-alone inference script for SSD models trained using TAO."""

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
from nvidia_tao_tf1.cv.ssd.builders import eval_builder
import nvidia_tao_tf1.cv.ssd.models.patch_keras
from nvidia_tao_tf1.cv.ssd.utils.model_io import load_model
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import INFERENCE_EXP_REQUIRED_MSG, \
    load_experiment_spec, spec_validator

nvidia_tao_tf1.cv.ssd.models.patch_keras.patch()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')


def build_command_line_parser(parser=None):
    '''build parser.'''
    if parser is None:
        parser = argparse.ArgumentParser(description='TLT SSD Inference Tool')
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
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def keras_output_process_fn(inferencer, y_encoded):
    "function to process keras model output."
    return y_encoded


def trt_output_process_fn(inferencer, y_encoded):
    "function to process TRT model output."
    det_out, keep_k = y_encoded
    result = []
    for idx, k in enumerate(keep_k.reshape(-1)):
        det = det_out[idx].reshape(-1, 7)[:k]
        xmin = det[:, 3] * inferencer.model_input_width
        ymin = det[:, 4] * inferencer.model_input_height
        xmax = det[:, 5] * inferencer.model_input_width
        ymax = det[:, 6] * inferencer.model_input_height
        cls_id = det[:, 1]
        conf = det[:, 2]
        result.append(np.stack((cls_id, conf, xmin, ymin, xmax, ymax), axis=-1))

    return result


def inference(arguments):
    '''make inference.'''

    config_path = arguments.experiment_spec
    experiment_spec, is_dssd = load_experiment_spec(config_path)
    spec_validator(experiment_spec, INFERENCE_EXP_REQUIRED_MSG)
    classes = sorted({str(x) for x in
                      experiment_spec.dataset_config.target_class_mapping.values()})
    class_mapping = dict(zip(range(1, len(classes)+1), classes))

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
        message="Starting {} Inference.".format("DSSD" if is_dssd else "SSD")
    )

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

    K.clear_session()  # Clear previous models from memory.
    K.set_learning_phase(0)

    if os.path.splitext(arguments.model_path)[1] in ['.h5', '.tlt', '.hdf5']:
        model = load_model(arguments.model_path, experiment_spec, is_dssd, key=arguments.key)

        # Load evaluation parameters
        conf_th = experiment_spec.nms_config.confidence_threshold
        iou_th = experiment_spec.nms_config.clustering_iou_threshold
        top_k = experiment_spec.nms_config.top_k
        nms_max_output = top_k
        # Build evaluation model
        model = eval_builder.build(model, conf_th, iou_th, top_k, nms_max_output)

        inferencer = Inferencer(keras_model=model,
                                batch_size=experiment_spec.eval_config.batch_size,
                                infer_process_fn=keras_output_process_fn,
                                class_mapping=class_mapping,
                                threshold=arguments.threshold,
                                img_mean=img_mean,
                                keep_aspect_ratio=False)

        print("Using TLT model for inference, setting batch size to the one in eval_config:",
              experiment_spec.eval_config.batch_size)
    else:
        inferencer = Inferencer(trt_engine_path=arguments.model_path,
                                infer_process_fn=trt_output_process_fn,
                                class_mapping=class_mapping,
                                threshold=arguments.threshold,
                                img_mean=img_mean,
                                keep_aspect_ratio=False)

        print("Using TensorRT engine for inference, setting batch size to engine's one:",
              inferencer.batch_size)

    out_image_path = os.path.join(arguments.results_dir, "images_annotated")
    out_label_path = os.path.join(arguments.results_dir, "labels")
    os.makedirs(out_image_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)

    inferencer.infer(arguments.image_dir, out_image_path, out_label_path)
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
