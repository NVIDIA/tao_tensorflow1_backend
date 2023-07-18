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

"""
Perform Inference for the Unet Segmentation.

This code does the inference. Given the paths of the test images, it predicts
masks and dumps the visualized segmented images.

Short code breakdown:
(1) Creates the Runtime_config and creates the estimator
(2) Hook up the data pipe and estimator to unet model with backbones such as
Resnet, vanilla Unet
(3) Retrieves/ Encrypts the trained checkpoint.
(4) Performs Inference and dumps images with segmentation vis.
"""

import argparse
import json
import logging
import math
import os
import random
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.unet.dllogger.logger import JSONStreamBackend, Logger, StdOutBackend, \
    Verbosity
from nvidia_tao_tf1.cv.unet.hooks.profiling_hook import ProfilingHook
from nvidia_tao_tf1.cv.unet.model.build_unet_model import build_model
from nvidia_tao_tf1.cv.unet.model.build_unet_model import select_model_proto
from nvidia_tao_tf1.cv.unet.model.model_io import _extract_ckpt
from nvidia_tao_tf1.cv.unet.model.utilities import build_target_class_list, get_train_class_mapping
from nvidia_tao_tf1.cv.unet.model.utilities import get_custom_objs, get_pretrained_ckpt, \
    update_model_params
from nvidia_tao_tf1.cv.unet.model.utilities import initialize, initialize_params, save_tmp_json
from nvidia_tao_tf1.cv.unet.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.unet.utils.data_loader import Dataset
from nvidia_tao_tf1.cv.unet.utils.inference_trt import Inferencer
from nvidia_tao_tf1.cv.unet.utils.model_fn import unet_fn

logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.INFO)


def resize_with_pad(image, f_target_width=None, f_target_height=None, inter=cv2.INTER_AREA):
    """Function to determine the padding width in all the directions."""

    (im_h, im_w) = image.shape[:2]
    ratio = max(im_w/float(f_target_width), im_h/float(f_target_height))
    resized_height_float = im_h/ratio
    resized_width_float = im_w/ratio
    resized_height = math.floor(resized_height_float)
    resized_width = math.floor(resized_width_float)
    padding_height = (f_target_height - resized_height_float)/2
    padding_width = (f_target_width - resized_width_float)/2
    f_padding_height = math.floor(padding_height)
    f_padding_width = math.floor(padding_width)
    p_height_top = max(0, f_padding_height)
    p_width_left = max(0, f_padding_width)
    p_height_bottom = max(0, f_target_height-(resized_height+p_height_top))
    p_width_right = max(0, f_target_width-(resized_width+p_width_left))

    return p_height_top, p_height_bottom, p_width_left, p_width_right


def get_color_id(dataset):
    """Function to return a list of color values for each class."""

    colors = []
    for idx in range(dataset.num_classes):
        random.seed(idx)
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return colors


def overlay_seg_image(inp_img, seg_img, resize_padding, resize_method):
    """The utility function to overlay mask on original image."""

    resize_methods_mapping = {'BILINEAR': cv2.INTER_LINEAR, 'AREA': cv2.INTER_AREA,
                              'BICUBIC': cv2.INTER_CUBIC,
                              'NEAREST_NEIGHBOR': cv2.INTER_NEAREST}
    rm = resize_methods_mapping[resize_method]
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_h = seg_img.shape[0]
    seg_w = seg_img.shape[1]

    if resize_padding:
        p_height_top, p_height_bottom, p_width_left, p_width_right = \
            resize_with_pad(inp_img, seg_w, seg_h)
        act_seg = seg_img[p_height_top:(seg_h-p_height_bottom), p_width_left:(seg_w-p_width_right)]
        seg_img = cv2.resize(act_seg, (orininal_w, orininal_h), interpolation=rm)
    else:
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=rm)

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


def visualize_masks(predictions, out_dir, input_image_type, img_names, colors,
                    mode="tlt", resize_padding=True, resize_method='BILINEAR',
                    activation="softmax"):
    """The function to visualize the segmentation masks.

    Args:
        predictions: Predicted masks numpy arrays.
        out_dir: Output dir where the visualization is saved.
        input_image_type: The input type of image (color/ grayscale).
        img_names: The input image names.
    """

    vis_dir = os.path.join(out_dir, "vis_overlay"+"_"+mode)
    label_dir = os.path.join(out_dir, "mask_labels"+"_"+mode)
    if not os.path.isdir(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    idx = 0
    for p, img_name in tqdm(zip(predictions, img_names)):
        pred = p['logits']
        tags = img_name.split("/")
        fn = tags[-1]
        idx += 1
        if activation == "softmax" or mode == "trt":
            # TRT inference is squeezed too
            output_height = pred.shape[0]
            output_width = pred.shape[1]
        else:
            output_height = pred.shape[1]
            output_width = pred.shape[2]
            pred = np.squeeze(pred, axis=0)
        if input_image_type == "grayscale":
            pred = pred.astype(np.uint8)*255
            img_resized = Image.fromarray(pred).resize(size=(output_width, output_height),
                                                       resample=Image.BILINEAR)
            img_resized.save(os.path.join(vis_dir, fn))
        else:
            segmented_img = np.zeros((output_height, output_width, 3))
            for c in range(len(colors)):
                seg_arr_c = pred[:, :] == c
                segmented_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
                segmented_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
                segmented_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')
            orig_image = cv2.imread(img_name)
            fused_img = overlay_seg_image(orig_image, segmented_img, resize_padding,
                                          resize_method)
            cv2.imwrite(os.path.join(vis_dir, fn), fused_img)

        mask_fn = "{}.png".format(os.path.splitext(fn)[0])
        cv2.imwrite(os.path.join(label_dir, mask_fn), pred)


def run_inference_tlt(dataset, params, unet_model, key,
                      output_dir, model_path):
    """Run the prediction followed by inference using the estimator.

    Args:
        dataset: Dataset object fro the dataloader utility.
        params: Parameters to feed to Estimator.
        unet_model: Keras Unet Model.
        key: The key to encrypt the model.
        output_dir: The directory where the results file is saved.
        model_path: The TLT model path for inference.

    """

    backends = [StdOutBackend(Verbosity.VERBOSE)]
    backends.append(JSONStreamBackend(Verbosity.VERBOSE, output_dir+"/log.txt"))
    profile_logger = Logger(backends)

    gpu_options = tf.compat.v1.GPUOptions()
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        save_summary_steps=1,
        tf_random_seed=None,
        session_config=config)
    checkpoint_path, model_json = _extract_ckpt(model_path, key)
    estimator = tf.estimator.Estimator(
        model_fn=unet_fn,
        model_dir=params.model_dir,
        config=run_config,
        params=params)
    predict_steps = dataset.test_size
    hooks = None
    if params.benchmark:
        hooks = [ProfilingHook(profile_logger,
                               batch_size=params.batch_size,
                               log_every=params.log_every,
                               warmup_steps=params.warmup_steps,
                               mode="test")]

        predict_steps = params.warmup_steps * 2 * params.batch_size
    predictions = estimator.predict(
        input_fn=lambda: dataset.test_fn(
            count=math.ceil(predict_steps/dataset.test_size)),
        hooks=hooks, checkpoint_path=checkpoint_path,
        )

    img_names = dataset.get_test_image_names()
    input_image_type = dataset.input_image_type
    colors = get_color_id(dataset)

    visualize_masks(predictions, output_dir, input_image_type, img_names, colors, "tlt",
                    params.resize_padding, params.resize_method,
                    activation=params["activation"])


def run_inference_trt(model_path, experiment_spec, output_dir, dataset, params, key="tlt_encode",
                      activation="softmax"):
    """Run the training loop using the estimator.

    Args:
        model_path: The path string where the trained model needs to be saved.
        experiment_spec: Experiment spec proto.
        output_dir: Folder to save the results text file.
        dataset: Dataset object.
        key: Key to encrypt the model.

    """

    inferencer = Inferencer(keras_model=None, trt_engine_path=model_path,
                            dataset=dataset, batch_size=dataset._batch_size,
                            activation=params["activation"])
    predictions, img_names = inferencer.infer(dataset.image_names_list)
    input_image_type = dataset.input_image_type
    colors = get_color_id(dataset)

    visualize_masks(predictions, output_dir, input_image_type, img_names, colors, mode="trt",
                    resize_method=params.resize_method,
                    resize_padding=params.resize_padding,
                    activation=params["activation"])


def infer_unet(model_path, experiment_spec, output_dir, key=None):
    """Run the training loop using the estimator.

    Args:
        model_dir: The path string where the trained model needs to be saved.
        experiment_spec: Experiment spec proto.
        output_dir: Folder to save the results text file.
        key: Key to encrypt the model.

    """

    # Initialize the environment
    initialize(experiment_spec)
    # Initialize Params
    params = initialize_params(experiment_spec)
    target_classes = build_target_class_list(
        experiment_spec.dataset_config.data_class_config)
    target_classes_train_mapping = get_train_class_mapping(target_classes)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'target_class_id_mapping.json'), 'w') as fp:
        json.dump(target_classes_train_mapping, fp)

    model_ext = os.path.splitext(model_path)[1]
    # Build run config
    model_config = select_model_proto(experiment_spec)
    unet_model = build_model(m_config=model_config,
                             target_class_names=target_classes)
    model_dir = os.path.abspath(os.path.join(model_path, os.pardir))
    custom_objs = None
    model_json = None
    # Update custom_objs with Internal TAO custom layers
    custom_objs = get_custom_objs(model_arch=model_config.arch)
    params = update_model_params(params=params, unet_model=unet_model,
                                 experiment_spec=experiment_spec,
                                 key=key, target_classes=target_classes,
                                 results_dir=model_dir,
                                 phase="test",
                                 custom_objs=custom_objs,
                                 model_json=model_json)

    if params.enable_qat and not params.load_graph:
        # We add QDQ nodes before session is formed
        img_height, img_width, img_channels = \
            experiment_spec.model_config.model_input_height, \
            experiment_spec.model_config.model_input_width, \
            experiment_spec.model_config.model_input_channels
        model_qat_json = unet_model.construct_model(
            input_shape=(img_channels, img_height, img_width),
            pretrained_weights_file=params.pretrained_weights_file,
            enc_key=params.key, model_json=params.model_json,
            features=None, construct_qat=True)
        model_qat_json = save_tmp_json(model_qat_json)
        params.model_json = model_qat_json

    dataset = Dataset(
                      batch_size=params.batch_size,
                      fold=params.crossvalidation_idx,
                      augment=params.augment,
                      params=params,
                      phase="test",
                      target_classes=target_classes)

    if model_ext in ['.tlt', '']:
        run_inference_tlt(dataset, params, unet_model,
                          key, output_dir, model_path)
    elif model_ext in ['.engine', '.trt']:
        run_inference_trt(model_path, experiment_spec, output_dir, dataset, params,
                          key=key, activation=params.activation)
    else:
        raise ValueError("Model extension needs to be either .engine or .trt.")


def run_experiment(model_path, config_path, output_dir,
                   override_spec_path=None, key=None):
    """
    Launch experiment that does inference.

    NOTE: Do not change the argument names without verifying that cluster submission works.

    Args:
        model_path (str): The model path for which the inference needs to be done.
        config_path (list): List containing path to a text file containing a complete experiment
            configuration and possibly a path to a .yml file containing override parameter values.
        output_dir (str): Path to a folder where the output of the inference .
            If the folder does not already exist, it will be created.
        override_spec_path (str): Absolute path to yaml file which is used to overwrite some of the
            experiment spec parameters.
        key (str): Key to save and load models from tlt.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    status_file = os.path.join(output_dir, "status.json")
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
        message="Starting UNet Inference"
    )
    logger.debug("Starting experiment.")

    # Load experiment spec.
    if config_path is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", config_path)

        # The spec in experiment_spec_path has to be complete.
        # Default spec is not merged into experiment_spec.
        experiment_spec = load_experiment_spec(
            config_path, merge_from_default=False)
    else:
        logger.info("Loading default ISBI single class experiment spec.")
        experiment_spec = load_experiment_spec()

    infer_unet(model_path, experiment_spec, output_dir, key=key)
    logger.debug("Experiment complete.")


def build_command_line_parser(parser=None):
    """
    Parse command-line flags passed to the training script.

    Returns:
      Namespace with all parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog='inference', description='Inference of segmentation model.')

    default_experiment_path = os.path.join(os.path.expanduser('~'), 'experiments')

    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        default=None,
        help='Path to spec file. Absolute path or relative to working directory. \
            If not specified, default spec from spec_loader.py is used.'
    )
    parser.add_argument(
        '-o',
        '--results_dir',
        type=str,
        default=default_experiment_path,
        help='Path to a folder where experiment annotated outputs are saved.'
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        default=default_experiment_path,
        help='Path to a folder from where the model should be taken for inference.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Set verbosity level for the logger.'
    )
    parser.add_argument(
        '-k',
        '--key',
        default="",
        type=str,
        required=False,
        help='The key to load the model provided for inference.'
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
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=1,
        help=argparse.SUPPRESS
    )

    return parser


def parse_command_line_args(cl_args=None):
    """Parser command line arguments to the trainer.

    Args:
        cl_args(sys.argv[1:]): Arg from the command line.

    Returns:
        args: Parsed arguments using argparse.
    """
    parser = build_command_line_parser(parser=None)
    args = parser.parse_args(cl_args)
    return args


def main(args=None):
    """Run the Inference process."""
    args = parse_command_line_args(args)

    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'

    # Configure the logger.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=verbosity)

    # Configure tf logger verbosity.
    tf.logging.set_verbosity(tf.logging.INFO)
    run_experiment(config_path=args.experiment_spec,
                   model_path=args.model_path,
                   output_dir=args.results_dir,
                   key=args.key)


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
