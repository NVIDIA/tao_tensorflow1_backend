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
"""FasterRCNN evaluation script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

from keras import backend as K
import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    from nvidia_tao_tf1.cv.faster_rcnn.tensorrt_inference.tensorrt_model import TrtModel
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
from nvidia_tao_tf1.cv.common import utils as iva_utils
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.faster_rcnn.data_loader.inputs_loader import InputsLoader
from nvidia_tao_tf1.cv.faster_rcnn.models.utils import build_inference_model
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader import spec_loader, spec_wrapper
from nvidia_tao_tf1.cv.faster_rcnn.utils import utils

KERAS_MODEL_EXTENSIONS = ["tlt", "hdf5"]


def build_command_line_parser(parser=None):
    """Build a command line parser for evaluation."""
    if parser is None:
        parser = argparse.ArgumentParser(description='Evaluate a Faster-RCNN model.')
    parser.add_argument("-e",
                        "--experiment_spec",
                        type=str,
                        required=True,
                        help="Experiment spec file has all the training params.")
    parser.add_argument("-k",
                        "--key",
                        type=str,
                        required=False,
                        help="TLT encoding key, can override the one in the spec file.")
    parser.add_argument("-m",
                        "--model_path",
                        type=str,
                        required=False,
                        default=None,
                        help="Path to the model to be used for evaluation")
    parser.add_argument("-r",
                        "--results_dir",
                        type=str,
                        default=None,
                        help="Path to the files where the logs are stored.")
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


def parse_args(args_in=None):
    """Parse arguments."""
    parser = build_command_line_parser()
    return parser.parse_known_args(args_in)[0]


def main(args=None):
    """Do evaluation on a pretrained model."""
    options = parse_args(args)
    spec = spec_loader.load_experiment_spec(options.experiment_spec)
    # enc key in CLI will override the one in the spec file.
    if options.key is not None:
        spec.enc_key = options.key
    # model in CLI will override the one in the spec file.
    if options.model_path is not None:
        spec.evaluation_config.model = options.model_path
    spec = spec_wrapper.ExperimentSpec(spec)
    # Set up status logging
    if options.results_dir:
        if not os.path.exists(options.results_dir):
            os.makedirs(options.results_dir)
        status_file = os.path.join(options.results_dir, "status.json")
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
            message="Starting FasterRCNN evaluation."
        )
    verbosity = 'INFO'
    if spec.verbose:
        verbosity = 'DEBUG'
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=verbosity)
    logger = logging.getLogger(__name__)
    # setup tf and keras
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    K.set_image_data_format('channels_first')
    K.set_learning_phase(0)
    # set radom seed
    utils.set_random_seed(spec.random_seed)
    # load model and convert train model to infer model
    if spec.eval_trt_config is not None:
        # spec.eval_trt_engine will be deprecated, use spec.eval_model
        logger.info('Running evaluation with TensorRT as backend.')
        logger.warning(
            "`spec.evaluation_config.trt_evaluation` is deprecated, "
            "please use `spec.evaluation_config.model` in spec file or provide "
            "the model/engine as a command line argument instead."
        )
        infer_model = TrtModel(spec.eval_trt_engine,
                               spec.eval_batch_size,
                               spec.image_h,
                               spec.image_w)
        infer_model.build_or_load_trt_engine()
        vis_path = os.path.dirname(spec.eval_trt_engine)
    elif spec.eval_model.split('.')[-1] in KERAS_MODEL_EXTENSIONS:
        # spec.eval_model is a TLT model
        logger.info('Running evaluation with TAO Toolkit as backend.')
        train_model = iva_utils.decode_to_keras(spec.eval_model,
                                                str.encode(spec.enc_key),
                                                input_model=None,
                                                compile_model=False,
                                                by_name=None)
        config_override = {'pre_nms_top_N': spec.eval_rpn_pre_nms_top_N,
                           'post_nms_top_N': spec.eval_rpn_post_nms_top_N,
                           'nms_iou_thres': spec.eval_rpn_nms_iou_thres,
                           'bs_per_gpu': spec.eval_batch_size}
        logger.info("Building evaluation model, may take a while...")
        infer_model = build_inference_model(
            train_model,
            config_override,
            create_session=True,
            max_box_num=spec.eval_rcnn_post_nms_top_N,
            regr_std_scaling=spec.rcnn_regr_std,
            iou_thres=spec.eval_rcnn_nms_iou_thres,
            score_thres=spec.eval_confidence_thres,
            eval_rois=spec.eval_rpn_post_nms_top_N
        )
        infer_model.summary()
        vis_path = os.path.dirname(spec.eval_model)
    else:
        # spec.eval_model is a TRT engine
        logger.info('Running evaluation with TensorRT as backend.')
        infer_model = TrtModel(spec.eval_model,
                               spec.eval_batch_size,
                               spec.image_h,
                               spec.image_w)
        infer_model.build_or_load_trt_engine()
        vis_path = os.path.dirname(spec.eval_model)
    data_loader = InputsLoader(spec.training_dataset,
                               spec.data_augmentation,
                               spec.eval_batch_size,
                               spec.image_c,
                               spec.image_mean_values,
                               spec.image_scaling_factor,
                               bool(spec.image_channel_order == 'bgr'),
                               training=False,
                               max_objs_per_img=spec.max_objs_per_img,
                               session=K.get_session())
    K.get_session().run(utils.get_init_ops())
    num_examples = data_loader.num_samples
    max_steps = (num_examples + spec.eval_batch_size - 1) // spec.eval_batch_size
    prob_thresh = spec.eval_confidence_thres

    T = [dict() for _ in spec.eval_gt_matching_iou_list]
    P = [dict() for _ in spec.eval_gt_matching_iou_list]
    RPN_RECALL = {}
    for _ in tqdm(range(max_steps)):
        images, gt_class_ids, gt_bboxes, gt_diff = data_loader.get_array_with_diff()
        image_h, image_w = images.shape[2:]
        # get the feature maps and output from the RPN
        nmsed_boxes, nmsed_scores, nmsed_classes, num_dets, rois_output = \
            infer_model.predict(images)
        # apply the spatial pyramid pooling to the proposed regions
        for image_idx in range(nmsed_boxes.shape[0]):
            all_dets = utils.gen_det_boxes(
                spec.id_to_class, nmsed_classes,
                nmsed_boxes, nmsed_scores,
                image_idx, num_dets,
            )
            # get detection results for each IoU threshold, for each image
            utils.get_detection_results(
                all_dets, gt_class_ids, gt_bboxes, gt_diff, image_h,
                image_w, image_idx, spec.id_to_class, T, P,
                spec.eval_gt_matching_iou_list
            )
            # # calculate RPN recall for each class, this will help debugging
            # in TensorRT engine eval case, rois_output is None, so skip this
            if rois_output is not None:
                utils.calc_rpn_recall(
                    RPN_RECALL,
                    spec.id_to_class,
                    rois_output[image_idx, ...],
                    gt_class_ids[image_idx, ...],
                    gt_bboxes[image_idx, ...]
                )
    # finally, compute and print all the mAP values
    maps = utils.compute_map_list(
        T, P, prob_thresh,
        spec.use_voc07_metric,
        RPN_RECALL,
        spec.eval_gt_matching_iou_list,
        vis_path if spec.eval_config.visualize_pr_curve else None
    )
    mAP = np.mean(maps)
    if options.results_dir:
        s_logger.kpi.update({'mAP': float(mAP)})
        s_logger.write(
            status_level=status_logging.Status.SUCCESS,
            message="Evaluation finished successfully."
        )


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        if type(e) == tf.errors.ResourceExhaustedError:
            logger = logging.getLogger(__name__)
            logger.error(
                "Ran out of GPU memory, please lower the batch size, use a smaller input "
                "resolution, use a smaller backbone or try model parallelism. See TLT "
                "documentation on how to enable model parallelism for FasterRCNN."
            )
            status_logging.get_status_logger().write(
                message="Ran out of GPU memory, please lower the batch size, use a smaller input "
                        "resolution, use a smaller backbone or try model parallelism. See TLT "
                        "documentation on how to enable model parallelism for FasterRCNN.",
                status_level=status_logging.Status.FAILURE
            )
            sys.exit(1)
        else:
            # throw out the error as-is if they are not OOM error
            status_logging.get_status_logger().write(
                message=str(e),
                status_level=status_logging.Status.FAILURE
            )
            raise e
