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

"""Inference and metrics computation code using a loaded model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import check_tf_oom, restore_eff
from nvidia_tao_tf1.cv.makenet.spec_handling.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.makenet.utils.helper import get_input_shape, model_io, setup_config
from nvidia_tao_tf1.cv.makenet.utils.preprocess_crop import load_and_crop_img
from nvidia_tao_tf1.cv.makenet.utils.preprocess_input import preprocess_input

logger = logging.getLogger(__name__)

VALID_IMAGE_EXT = ['.jpg', '.jpeg', '.png']


def build_command_line_parser(parser=None):
    '''Parse command line arguments.'''
    if parser is None:
        parser = argparse.ArgumentParser(
                    description="Standalone classification inference tool")
    parser.add_argument('-m', '--model_path',
                        type=str,
                        help="Path to the pretrained model (.tlt).",
                        required=True)
    parser.add_argument('-k',
                        '--key',
                        required=False,
                        default="",
                        type=str,
                        help='Key to load a .tlt model.')
    parser.add_argument('-i', '--image',
                        type=str,
                        help="Path to the inference image.")
    parser.add_argument('-d', '--image_dir',
                        type=str,
                        help="Path to the inference image directory.")
    parser.add_argument('-e',
                        '--experiment_spec',
                        required=True,
                        type=str,
                        help='Path to the experiment spec file.')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=1,
                        help="Inference batch size.")
    parser.add_argument('-cm', '--classmap',
                        type=str,
                        help="Path to the classmap file generated from training.",
                        default=None,
                        required=True)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Include this flag in command line invocation for\
                              verbose logs.')
    parser.add_argument('-r',
                        "--results_dir",
                        type=str,
                        default=None,
                        help=argparse.SUPPRESS)
    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


def batch_generator(iterable, batch_size=1):
    """Load a list of image paths in batches.

    Args:
        iterable: a list of image paths
        n: batch size
    """
    total_len = len(iterable)
    for ndx in range(0, total_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, total_len)]


def preprocess(imgpath, image_height,
               image_width, nchannels=3,
               mode='caffe',
               img_mean=None,
               interpolation='nearest',
               data_format='channels_first'):
    """Preprocess a single image.

    It includes resizing, normalization based on imagenet
    """
    # Open image and preprocessing
    color_mode = 'rgb' if nchannels == 3 else 'grayscale'
    image = load_and_crop_img(
        imgpath,
        grayscale=False,
        color_mode=color_mode,
        target_size=(image_height, image_width),
        interpolation=interpolation,
    )
    image = np.array(image).astype(np.float32)
    return preprocess_input(image.transpose((2, 0, 1)),
                            mode=mode, color_mode=color_mode,
                            img_mean=img_mean,
                            data_format=data_format)


def load_image_batch(batch, image_height,
                     image_width, nchannels=3,
                     mode='caffe',
                     img_mean=None,
                     interpolation='nearest',
                     data_format='channels_first'):
    """Group the preprocessed images in a batch."""
    ph = np.zeros(
        (len(batch), nchannels, image_height, image_width),
        dtype=np.float32)
    for i, imgpath in enumerate(batch):
        ph[i, :, :, :] = preprocess(imgpath, image_height, image_width,
                                    nchannels=nchannels, mode=mode,
                                    img_mean=img_mean,
                                    interpolation=interpolation,
                                    data_format=data_format)
    return ph


def inference(args=None):
    """Inference on an image/directory using a pretrained model file.

    Args:
        args: Dictionary arguments containing parameters defined by command
              line parameters.
    Log:
        Image Mode:
            print classifier output
        Directory Mode:
            write out a .csv file to store all the predictions
    """

    # Set up status logging
    if args.results_dir:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        status_file = os.path.join(args.results_dir, "status.json")
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
            message="Starting inference."
        )
    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'

    if not (args.image or args.image_dir):
        s_logger.write(
            status_level=status_logging.Status.FAILURE,
            message="Provide either image file or a directory of images."
        )
        return

    # Configure the logger.
    logging.basicConfig(
                format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                level=verbosity)

    # Load experiment spec.
    if args.experiment_spec is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", args.experiment_spec)
        # The spec in config_path has to be complete.
        # Default spec is not merged into es.
        es = load_experiment_spec(args.experiment_spec,
                                  merge_from_default=False,
                                  validation_schema="validation")
    else:
        logger.info("Loading the default experiment spec.")
        es = load_experiment_spec(validation_schema="validation")
    # override BN config
    if es.model_config.HasField("batch_norm_config"):
        bn_config = es.model_config.batch_norm_config
    else:
        bn_config = None

    custom_objs = {}

    # Decrypt and load the pretrained model
    model = model_io(args.model_path, enc_key=args.key, custom_objs=custom_objs)

    # reg_config and freeze_bn are actually not useful, just use bn_config
    # so the BN layer's output produces correct result.
    # of course, only the BN epsilon matters in evaluation.
    model = setup_config(
        model,
        es.train_config.reg_config,
        freeze_bn=es.model_config.freeze_bn,
        bn_config=bn_config,
        custom_objs=custom_objs
    )
    # Printing summary of retrieved model
    model.summary()
    # Get input shape
    image_height, image_width, nchannels = get_input_shape(model)

    with open(args.classmap, "r") as cm:
        class_dict = json.load(cm)

    interpolation = es.model_config.resize_interpolation_method
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    interpolation = interpolation_map[interpolation]
    if es.eval_config.enable_center_crop:
        interpolation += ":center"

    img_mean = es.train_config.image_mean
    if nchannels == 3:
        if img_mean:
            assert all(c in img_mean for c in ['r', 'g', 'b']) , (
                "'r', 'g', 'b' should all be present in image_mean "
                "for images with 3 channels."
            )
            img_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
        else:
            img_mean = [103.939, 116.779, 123.68]
    else:
        if img_mean:
            assert 'l' in img_mean, (
                "'l' should be present in image_mean for images "
                "with 1 channel."
            )
            img_mean = [img_mean['l']]
        else:
            img_mean = [117.3786]

    if args.image:
        logger.info("Processing {}...".format(args.image))
        # Load and preprocess image
        infer_input = preprocess(args.image, image_height, image_width,
                                 nchannels=nchannels,
                                 mode=es.train_config.preprocess_mode,
                                 img_mean=img_mean,
                                 interpolation=interpolation)
        infer_input.shape = (1, ) + infer_input.shape
        # Keras inference
        raw_predictions = model.predict(infer_input, batch_size=1)
        logger.debug("Raw prediction: \n{}".format(raw_predictions))
        # Class output from softmax layer
        class_index = np.argmax(raw_predictions)
        print("Current predictions: {}".format(raw_predictions))
        print("Class label = {}".format(class_index))
        # Label Name
        class_name = list(class_dict.keys())[list(class_dict.values()).index(class_index)]
        print("Class name = {}".format(class_name))

    if args.image_dir:
        logger.info("Processing {}...".format(args.image_dir))
        # Preparing list of inference files.

        result_csv_path = os.path.join(args.image_dir, 'result.csv')
        if args.results_dir:
            result_csv_path = os.path.join(args.results_dir, 'result.csv')
        csv_f = open(result_csv_path, 'w')

        imgpath_list = [os.path.join(root, filename)
                        for root, subdirs, files in os.walk(args.image_dir)
                        for filename in files
                        if os.path.splitext(filename)[1].lower()
                        in VALID_IMAGE_EXT
                        ]

        if not imgpath_list:
            s_logger.write(
                status_level=status_logging.Status.FAILURE,
                message="Image directory doesn't contain files with valid extensions" +
                        "Valid extensions are " + str(VALID_IMAGE_EXT)
            )
            return

        # Generator in batch mode
        for img_batch in batch_generator(imgpath_list, args.batch_size):
            # Load images in batch
            infer_batch = load_image_batch(img_batch,
                                           image_height,
                                           image_width,
                                           nchannels=nchannels,
                                           interpolation=interpolation,
                                           img_mean=img_mean,
                                           mode=es.train_config.preprocess_mode)
            # Run inference
            raw_predictions = model.predict(infer_batch,
                                            batch_size=args.batch_size)
            logger.debug("Raw prediction: \n{}".format(raw_predictions))
            # Class output from softmax layer
            class_indices = np.argmax(raw_predictions, axis=1)
            # Map label index to label name
            class_labels = map(lambda i: list(class_dict.keys())
                               [list(class_dict.values()).index(i)],
                               class_indices)
            conf = np.max(raw_predictions, axis=1)
            # Write predictions to file
            df = pd.DataFrame(zip(list(img_batch), class_labels, conf))
            df.to_csv(csv_f, header=False, index=False)
        logger.info("Inference complete. Result is saved at {}".format(
            result_csv_path))
        if args.results_dir:
            s_logger.write(
                status_level=status_logging.Status.SUCCESS,
                message="Inference finished successfully."
            )
        csv_f.close()


@check_tf_oom
def main(args=None):
    """Run inference on a single image or collection of images.

    Args:
       args: Dictionary arguments containing parameters defined by command
             line parameters.
    """
    try:
        # parse command line
        args = parse_command_line(args)
        inference(args)
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference convert was interrupted",
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
