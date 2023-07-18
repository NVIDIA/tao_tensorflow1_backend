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
"""Simple Stand-alone evaluate script for SSD models trained using TAO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

import keras.backend as K
from keras.utils.data_utils import OrderedEnqueuer
import numpy as np
import tensorflow as tf
from tqdm import trange

from nvidia_tao_tf1.cv.common.evaluator.ap_evaluator import APEvaluator
from nvidia_tao_tf1.cv.common.inferencer.inferencer import Inferencer
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import check_tf_oom
from nvidia_tao_tf1.cv.ssd.builders import eval_builder
from nvidia_tao_tf1.cv.ssd.builders.data_sequence import SSDDataSequence
import nvidia_tao_tf1.cv.ssd.models.patch_keras
from nvidia_tao_tf1.cv.ssd.utils.model_io import load_model
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import load_experiment_spec, validate_eval_spec

nvidia_tao_tf1.cv.ssd.models.patch_keras.patch()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')


def build_command_line_parser(parser=None):
    '''build parser.'''
    if parser is None:
        parser = argparse.ArgumentParser(description='TLT SSD Evaluation Tool')
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to an SSD TLT model or TensorRT engine.',
                        required=True,
                        type=str)
    parser.add_argument('-k',
                        '--key',
                        type=str,
                        default="",
                        help='Key to save or load a .tlt model.')
    parser.add_argument('-e',
                        '--experiment_spec',
                        required=True,
                        type=str,
                        help='Experiment spec file for training and evaluation.')
    parser.add_argument('-r',
                        '--results_dir',
                        type=str,
                        help='Path to a folder where the logs are stored.')
    parser.add_argument('-i',
                        '--image_dir',
                        type=str,
                        required=False,
                        default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('-l',
                        '--label_dir',
                        type=str,
                        required=False,
                        help=argparse.SUPPRESS)
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
        result.append(
            np.stack((cls_id, conf, xmin, ymin, xmax, ymax), axis=-1))

    return result


def evaluate(arguments):
    '''make evaluation.'''

    config_path = arguments.experiment_spec
    experiment_spec, is_dssd = load_experiment_spec(config_path)
    validate_eval_spec(experiment_spec)
    classes = sorted({str(x) for x in
                      experiment_spec.dataset_config.target_class_mapping.values()})

    K.clear_session()  # Clear previous models from memory.
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
        message="Starting {} evaluation.".format("DSSD" if is_dssd else "SSD")
    )

    if experiment_spec.eval_config.visualize_pr_curve:
        vis_path = os.path.dirname(arguments.model_path)
    else:
        vis_path = None

    if os.path.splitext(arguments.model_path)[1] in ['.h5', '.tlt', '.hdf5']:
        K.set_learning_phase(0)
        model = load_model(arguments.model_path, experiment_spec,
                           is_dssd, key=arguments.key)

        # Load evaluation parameters
        conf_th = experiment_spec.nms_config.confidence_threshold
        iou_th = experiment_spec.nms_config.clustering_iou_threshold
        top_k = experiment_spec.nms_config.top_k
        nms_max_output = top_k
        # Build evaluation model
        model = eval_builder.build(
            model, conf_th, iou_th, top_k, nms_max_output)

        inferencer = Inferencer(keras_model=model,
                                batch_size=experiment_spec.eval_config.batch_size,
                                infer_process_fn=keras_output_process_fn,
                                class_mapping=None,
                                threshold=experiment_spec.nms_config.confidence_threshold)

        print("Using TLT model for inference, setting batch size to the one in eval_config:",
              experiment_spec.eval_config.batch_size)
    else:
        # Works in python 3.6
        cpu_cnt = os.cpu_count()
        if cpu_cnt is None:
            cpu_cnt = 1

        session_config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0, 'CPU': cpu_cnt}
        )

        session = tf.Session(config=session_config)
        # Pin TF to CPU to avoid TF & TRT CUDA context conflict
        K.set_session(session)

        inferencer = Inferencer(trt_engine_path=arguments.model_path,
                                batch_size=experiment_spec.eval_config.batch_size,
                                infer_process_fn=trt_output_process_fn,
                                class_mapping=None,
                                threshold=experiment_spec.nms_config.confidence_threshold)

        print("Using TensorRT engine for inference, setting batch size to",
              f"{experiment_spec.eval_config.batch_size} from eval_config")

    ap_mode = experiment_spec.eval_config.average_precision_mode
    matching_iou = experiment_spec.eval_config.matching_iou_threshold
    matching_iou = matching_iou if matching_iou > 0 else 0.5
    ap_mode_dict = {0: "sample", 1: "integrate"}
    average_precision_mode = ap_mode_dict[ap_mode]

    evaluator = APEvaluator(len(classes)+1,
                            conf_thres=experiment_spec.nms_config.confidence_threshold,
                            matching_iou_threshold=matching_iou,
                            average_precision_mode=average_precision_mode)

    val_dataset = SSDDataSequence(dataset_config=experiment_spec.dataset_config,
                                  augmentation_config=experiment_spec.augmentation_config,
                                  batch_size=experiment_spec.eval_config.batch_size,
                                  is_training=False,
                                  encode_fn=None)

    # Prepare labels

    gt_labels = []
    pred_labels = []

    tr = trange(len(val_dataset), file=sys.stdout)
    tr.set_description('Producing predictions')

    enqueuer = OrderedEnqueuer(val_dataset, use_multiprocessing=False)
    enqueuer.start(workers=max(os.cpu_count() - 1, 1), max_queue_size=20)
    output_generator = enqueuer.get()

    output_height = val_dataset.output_height
    output_width = val_dataset.output_width

    # Loop over all batches.
    for _ in tr:
        # Generate batch.
        batch_X, batch_labs = next(output_generator)

        y_pred = inferencer._predict_batch(batch_X)

        gt_labels.extend(batch_labs)

        conf_thres = experiment_spec.nms_config.confidence_threshold

        for i in range(len(y_pred)):
            y_pred_valid = y_pred[i][y_pred[i][:, 1] > conf_thres]
            y_pred_valid[..., 2] = np.clip(y_pred_valid[..., 2].round(), 0.0,
                                           output_width)
            y_pred_valid[..., 3] = np.clip(y_pred_valid[..., 3].round(), 0.0,
                                           output_height)
            y_pred_valid[..., 4] = np.clip(y_pred_valid[..., 4].round(), 0.0,
                                           output_width)
            y_pred_valid[..., 5] = np.clip(y_pred_valid[..., 5].round(), 0.0,
                                           output_height)
            pred_labels.append(y_pred_valid)

    enqueuer.stop()

    results = evaluator(
        gt_labels,
        pred_labels,
        verbose=True,
        class_names=["bg"] + classes if experiment_spec.eval_config.visualize_pr_curve else None,
        vis_path=vis_path)
    _, average_precisions = results
    mean_average_precision = np.mean(average_precisions[1:])
    print("*******************************")
    for i in range(len(classes)):
        print("{:<14}{:<6}{}".format(
            classes[i], 'AP', round(average_precisions[i+1], 3)))

    print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))
    print("*******************************")
    s_logger.kpi.update({'mAP': float(mean_average_precision)})
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Evaluation finished successfully."
    )


@check_tf_oom
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
