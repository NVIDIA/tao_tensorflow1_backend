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
"""Simple Stand-alone evaluate script for YOLO models trained using modulus."""

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
from nvidia_tao_tf1.cv.yolo_v3.builders import eval_builder
from nvidia_tao_tf1.cv.yolo_v3.data_loader.data_loader import YOLOv3DataPipe
from nvidia_tao_tf1.cv.yolo_v3.dataio.data_sequence import YOLOv3DataSequence
from nvidia_tao_tf1.cv.yolo_v3.utils.model_io import load_model
from nvidia_tao_tf1.cv.yolo_v3.utils.spec_loader import (
    load_experiment_spec,
    validation_labels_format
)

from nvidia_tao_tf1.cv.yolo_v3.utils.tensor_utils import get_init_ops

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')


def build_command_line_parser(parser=None):
    '''build parser.'''
    if parser is None:
        parser = argparse.ArgumentParser(description='TLT YOLOv3 Evaluation Tool')
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to an YOLOv3 TLT model or TensorRT engine.',
                        required=True,
                        type=str)
    parser.add_argument('-k',
                        '--key',
                        type=str,
                        default="",
                        help='Key to load a .tlt model.')
    parser.add_argument('-e',
                        '--experiment_spec',
                        required=False,
                        type=str,
                        help='Experiment spec file for training and evaluation.')
    parser.add_argument("-r",
                        '--results_dir',
                        type=str,
                        default=None,
                        help='Path to the files where the logs are stored.')
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
    keep_k, boxes, scores, cls_id = y_encoded

    result = []
    for idx, k in enumerate(keep_k.reshape(-1)):
        loc = boxes[idx].reshape(-1, 4)[:k]
        cid = cls_id[idx].reshape(-1, 1)[:k]
        conf = scores[idx].reshape(-1, 1)[:k]
        result.append(np.concatenate((cid, conf, loc), axis=-1))

    return result


def evaluate(arguments):
    '''make evaluation.'''
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
            message="Starting YOLOv3 evaluation."
        )
    config_path = arguments.experiment_spec
    experiment_spec = load_experiment_spec(config_path)
    val_labels_format = validation_labels_format(experiment_spec)
    classes = sorted({str(x).lower() for x in
                      experiment_spec.dataset_config.target_class_mapping.values()})
    ap_mode = experiment_spec.eval_config.average_precision_mode
    matching_iou = experiment_spec.eval_config.matching_iou_threshold
    matching_iou = matching_iou if matching_iou > 0 else 0.5
    ap_mode_dict = {0: "sample", 1: "integrate"}
    average_precision_mode = ap_mode_dict[ap_mode]

    K.clear_session()  # Clear previous models from memory.

    evaluator = APEvaluator(len(classes),
                            conf_thres=experiment_spec.nms_config.confidence_threshold,
                            matching_iou_threshold=matching_iou,
                            average_precision_mode=average_precision_mode)

    if os.path.splitext(arguments.model_path)[1] in ['.tlt', '.hdf5']:
        K.set_learning_phase(0)
        img_height = experiment_spec.augmentation_config.output_height
        img_width = experiment_spec.augmentation_config.output_width
        n_channels = experiment_spec.augmentation_config.output_channel
        model = load_model(
            arguments.model_path,
            experiment_spec,
            (n_channels, img_height, img_width),
            key=arguments.key
        )
        # Load evaluation parameters
        conf_th = experiment_spec.nms_config.confidence_threshold
        iou_th = experiment_spec.nms_config.clustering_iou_threshold
        top_k = experiment_spec.nms_config.top_k
        nms_on_cpu = False
        if val_labels_format == "tfrecords":
            nms_on_cpu = True
        # Build evaluation model
        model = eval_builder.build(
            model, conf_th, iou_th, top_k, nms_on_cpu=nms_on_cpu
        )
        model.summary()
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
            device_count={'GPU' : 0, 'CPU': cpu_cnt}
        )
        session = tf.Session(config=session_config)
        # Pin TF to CPU to avoid TF & TRT CUDA context conflict
        K.set_session(session)
        inferencer = Inferencer(trt_engine_path=arguments.model_path,
                                infer_process_fn=trt_output_process_fn,
                                batch_size=experiment_spec.eval_config.batch_size,
                                class_mapping=None,
                                threshold=experiment_spec.nms_config.confidence_threshold)
        print("Using TensorRT engine for inference, setting batch size to engine's one:",
              inferencer.batch_size)
    # Prepare labels
    sess = K.get_session()
    if val_labels_format == "tfrecords":
        h_tensor = tf.constant(
            experiment_spec.augmentation_config.output_height,
            dtype=tf.int32
        )
        w_tensor = tf.constant(
            experiment_spec.augmentation_config.output_width,
            dtype=tf.int32
        )
        val_dataset = YOLOv3DataPipe(
            experiment_spec,
            label_encoder=None,
            training=False,
            h_tensor=h_tensor,
            w_tensor=w_tensor,
            sess=sess
        )
        num_samples = val_dataset.num_samples
        num_steps = num_samples // experiment_spec.eval_config.batch_size
        tr = trange(num_steps, file=sys.stdout)
        sess.run(get_init_ops())
    else:
        eval_sequence = YOLOv3DataSequence(
            experiment_spec.dataset_config,
            experiment_spec.augmentation_config,
            experiment_spec.eval_config.batch_size,
            is_training=False,
            encode_fn=None
        )
        enqueuer = OrderedEnqueuer(eval_sequence, use_multiprocessing=False)
        enqueuer.start(workers=max(os.cpu_count() - 1, 1), max_queue_size=20)
        output_generator = enqueuer.get()
        tr = trange(len(eval_sequence), file=sys.stdout)
    tr.set_description('Producing predictions')
    gt_labels = []
    pred_labels = []
    # Loop over all batches.
    for _ in tr:
        # Generate batch.
        if val_labels_format == "tfrecords":
            batch_X, batch_labs = val_dataset.get_array()
        else:
            batch_X, batch_labs = next(output_generator)
        y_pred = inferencer._predict_batch(batch_X)
        gt_labels.extend(batch_labs)
        conf_thres = experiment_spec.nms_config.confidence_threshold
        for i in range(len(y_pred)):
            y_pred_valid = y_pred[i][y_pred[i][:, 1] > conf_thres]
            pred_labels.append(y_pred_valid)
    results = evaluator(gt_labels, pred_labels, verbose=True)
    mean_average_precision, average_precisions = results
    print("*******************************")
    for i in range(len(average_precisions)):
        print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 5)))
    print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 5)))
    print("*******************************")
    if arguments.results_dir:
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


if __name__ == "__main__":
    main()
