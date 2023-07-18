# Copyright (c) 2017 - 2019, NVIDIA CORPORATION.  All rights reserved.
"""Simple Stand-alone inference script for gridbox models trained using TAO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time

from google.protobuf.json_format import MessageToDict
from PIL import Image
from tqdm import tqdm

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.detectnet_v2.inferencer.build_inferencer import build_inferencer
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.bbox_handler import BboxHandler
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.detectnet_v2.utilities.constants import valid_image_ext

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    '''Build argpase based command line parser for TLT infer.'''
    if parser is None:
        parser = argparse.ArgumentParser(description='TLT DetectNet_v2 Inference Tool')
    parser.add_argument("-e",
                        "--experiment_spec",
                        default=None,
                        type=str,
                        help="Path to inferencer spec file.",
                        required=True)
    parser.add_argument('-i',
                        '--image_dir',
                        help='The directory of input images or a single image for inference.',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument("-k",
                        "--key",
                        default="",
                        help="Key to load the model.",
                        type=str,
                        required=False)
    parser.add_argument('-r',
                        '--results_dir',
                        help='The directory to the output images and labels.'
                        ' The annotated images are in inference_output/images_annotated and'
                        ' \n labels are in image_dir/labels',
                        type=str,
                        required=True,
                        default=None)
    parser.add_argument('-b',
                        '--batch_size',
                        help='Batch size to be used. '
                        'If not provided, will use value from the spec file',
                        type=int,
                        default=None)
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        help='Path to the DetectNetv2 model.'
                        'If not provided, will use value from the spec file')
    parser.add_argument('-v',
                        '--verbosity',
                        action='store_true',
                        help="Flag to set for more detailed logs.")
    return parser


def parse_command_line(cl_args=None):
    """Parse the command line arguments."""
    parser = build_command_line_parser(parser=None)
    args = vars(parser.parse_args(cl_args))
    return args


def inference_wrapper_batch(inf_config, b_config,
                            inference_image_root=None,
                            output_root=None,
                            verbosity=False,
                            model_path=None,
                            batch_size=None,
                            key=None):
    """Wrapper function running batchwise inference on a directory of images using custom handlers.

    Input:
        inf_config (InferencerConfig Proto): Inferencer config proto object.
        b_config (BBoxerConfig Proto): BBoxer config proto object.
        output_root (str): Path to where the output would be stored.
        verbosity (bool): Flag to set logger verbosity level.
        model_path (str): path to the model
        batch_size (int): Batch size to use.
        key (str): Key to load the model for inference.

    Returns:
        No explicit returns.

    Outputs:
        - kitti labels in output_root/labels
        - overlain images in output_root/images_annotated
    """
    if not os.path.exists(inference_image_root):
        raise ValueError('Invalid infer image root {}'.format(inference_image_root))

    disable_overlay = not(b_config.disable_overlay)
    # If batch size was passed from argument, use that over spec value
    if batch_size:
        inf_config.batch_size = batch_size
    else:
        batch_size = inf_config.batch_size

    # If model path was passed from argument, use that over spec value
    if model_path:
        inf_config.tlt_config.model = model_path

    if disable_overlay:
        logger.info("Overlain images will be saved in the output path.")

    framework, model = build_inferencer(inf_config=inf_config,
                                        key=key,
                                        verbose=verbosity)
    bboxer = BboxHandler(save_kitti=b_config.kitti_dump,
                         image_overlay=disable_overlay,
                         batch_size=batch_size,
                         frame_height=inf_config.image_height if inf_config.image_height else 544,
                         frame_width=inf_config.image_width if inf_config.image_width else 960,
                         target_classes=inf_config.target_classes,
                         stride=inf_config.stride if inf_config.stride else 16,
                         postproc_classes=b_config.postproc_classes if b_config.postproc_classes
                         else inf_config.target_classes,
                         classwise_cluster_params=b_config.classwise_bbox_handler_config,
                         framework=framework)

    # Initialize the network for inference.
    model.network_init()
    logger.info("Initialized model")

    # # Preparing list of inference files.
    if os.path.isfile(inference_image_root):
        infer_files = [os.path.basename(inference_image_root)]
        inference_image_root = os.path.dirname(inference_image_root)
    elif os.path.isdir(inference_image_root):
        infer_files = [images for images in sorted(os.listdir(inference_image_root))
                       if os.path.splitext(images)[1].lower() in valid_image_ext]
    else:
        raise IOError("Invalid input type given for the -i flag. {}".format(inference_image_root))
    linewidth = b_config.overlay_linewidth

    # Setting up directories for outputs. including crops, labels and annotated images.
    output_image_root = os.path.join(output_root, 'images_annotated')
    output_label_root = os.path.join(output_root, 'labels')

    logger.info('Commencing inference')
    for chunk in tqdm([infer_files[x:x+batch_size] for x in range(0, len(infer_files),
                                                                  batch_size)]):
        pil_list = []
        time_start = time.time()

        # Preparing the chunk of images for inference
        for file_name in chunk:
            # By default convert the images to RGB so that the rendered boxes can be
            # set to different colors. Input preprocessing is handled in the
            # BaseInferencer class.
            pil_image = Image.open(os.path.join(inference_image_root,
                                                file_name)).convert("RGB")
            pil_list.append(pil_image)
        time_end = time.time()
        logger.debug("Time lapsed to prepare batch: {}".format(time_end - time_start))

        # Predict on a batch of images.
        time_start = time.time()
        output_inferred, resized_size = model.infer_batch(pil_list)
        time_end = time.time()
        logger.debug("Time lapsed to infer batch: {}".format(time_end - time_start))

        # Post process to obtain detections.
        processed_inference = bboxer.bbox_preprocessing(output_inferred)
        logger.debug("Preprocessing complete")
        classwise_detections = bboxer.cluster_detections(processed_inference)
        logger.debug("Classwise_detections")

        # Overlaying information after detection.
        time_start = time.time()
        logger.debug("Postprocessing detections: overlaying, metadata and crops.")
        bboxer.render_outputs(classwise_detections,
                              pil_list,
                              output_image_root,
                              output_label_root,
                              chunk,
                              resized_size,
                              linewidth=linewidth)
        time_end = time.time()
        logger.debug("Time lapsed: {}".format(time_end - time_start))

    if framework == "tensorrt":
        model.clear_buffers()
        model.clear_trt_session()
    logger.info("Inference complete")


def main(args=None):
    """Wrapper function for running inference on a single image or collection of images.

    Args:
       Dictionary arguments containing parameters defined by command line parameters
    """
    arguments = parse_command_line(args)
    # Setting up logger verbosity.
    verbosity = arguments["verbosity"]
    info_level = 'INFO'
    if verbosity:
        info_level = 'DEBUG'
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=info_level)

    results_dir = arguments['results_dir']
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
        message="Starting DetectNet_v2 Inference"
    )

    inference_spec = load_experiment_spec(spec_path=arguments['experiment_spec'],
                                          merge_from_default=False,
                                          validation_schema="inference")
    inferencer_config = inference_spec.inferencer_config
    bbox_handler_config = inference_spec.bbox_handler_config
    wandb_logged_in = check_wandb_logged_in()
    if bbox_handler_config.HasField("wandb_config"):
        wandb_config = bbox_handler_config.wandb_config
        wandb_name = f"{wandb_config.name}" if wandb_config.name \
            else "detectnet_v2_inference"
        wandb_stream_config = MessageToDict(
            inference_spec,
            preserving_proto_field_name=True,
            including_default_value_fields=True
        )
        initialize_wandb(
            project=wandb_config.project if wandb_config.project else None,
            entity=wandb_config.entity if wandb_config.entity else None,
            config=wandb_stream_config,
            notes=wandb_config.notes if wandb_config.notes else None,
            tags=wandb_config.tags if wandb_config.tags else None,
            sync_tensorboard=False,
            save_code=False,
            results_dir=results_dir,
            wandb_logged_in=wandb_logged_in,
            name=wandb_name
        )
    inference_wrapper_batch(inferencer_config, bbox_handler_config,
                            inference_image_root=arguments['image_dir'],
                            output_root=results_dir,
                            verbosity=verbosity,
                            model_path=arguments['model_path'],
                            key=arguments['key'],
                            batch_size=arguments['batch_size'])


if __name__ == "__main__":
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Inference finished successfully."
        )
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
