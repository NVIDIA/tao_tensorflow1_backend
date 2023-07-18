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
"""Standalone inference with EfficientDet checkpoint."""

import argparse
import os

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.util import deprecation

try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    from nvidia_tao_tf1.cv.efficientdet.inferencer import inference_trt
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf1.cv.efficientdet.inferencer import inference
from nvidia_tao_tf1.cv.efficientdet.utils import hparams_config
from nvidia_tao_tf1.cv.efficientdet.utils.model_loader import decode_tlt_file
from nvidia_tao_tf1.cv.efficientdet.utils.spec_loader import (
    generate_params_from_spec,
    load_experiment_spec
)

deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']


def get_label_dict(label_txt):
    """Create label dict from txt file."""

    with open(label_txt, 'r') as f:
        labels = f.readlines()
        return {i+1 : label[:-1] for i, label in enumerate(labels)}


def batch_generator(iterable, batch_size=1):
    """Load a list of image paths in batches.

    Args:
        iterable: a list of image paths
        n: batch size
    """
    total_len = len(iterable)
    for ndx in range(0, total_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, total_len)]


def main(args=None):
    """Launch EfficientDet training."""
    disable_eager_execution()
    tf.autograph.set_verbosity(0)
    # parse CLI and config file
    args = parse_command_line_arguments(args)
    assert args.experiment_spec, "Experiment spec file must be specified."

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
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
        message="Starting EfficientDet inference."
    )
    print("Loading experiment spec at %s.", args.experiment_spec)
    spec = load_experiment_spec(args.experiment_spec, merge_from_default=False)
    label_id_mapping = {}
    if args.class_map:
        label_id_mapping = get_label_dict(args.class_map)
    if args.model_path.endswith('.tlt'):
        infer_tlt(args, spec, label_id_mapping)
    elif args.model_path.endswith('.engine'):
        inference_trt.inference(args,
                                label_id_mapping,
                                spec.eval_config.min_score_thresh or args.threshold)
    else:
        raise ValueError("Model file should be in either .tlt or .engine format.")
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Inference finished successfully."
    )


def infer_tlt(args, spec, label_id_mapping):
    """Inference with tlt model."""
    # set up config
    MODE = 'infer'
    # Parse and override hparams
    config = hparams_config.get_detection_config(spec.model_config.model_name)
    params = generate_params_from_spec(config, spec, MODE)
    config.update(params)
    config.label_id_mapping = label_id_mapping
    if config.pruned_model_path:
        config.pruned_model_path = decode_tlt_file(config.pruned_model_path, args.key)

    driver = inference.InferenceDriver(config.name, decode_tlt_file(args.model_path, args.key),
                                       config.as_dict())
    config_dict = {}
    config_dict['line_thickness'] = 5
    config_dict['max_boxes_to_draw'] = spec.eval_config.max_detections_per_image or 100
    config_dict['min_score_thresh'] = spec.eval_config.min_score_thresh or args.threshold

    out_image_path = os.path.join(args.results_dir, "images_annotated")
    out_label_path = os.path.join(args.results_dir, "labels")
    os.makedirs(out_image_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)

    if os.path.exists(args.image_dir):
        if os.path.isfile(args.image_dir):
            driver.inference(args.image_dir, out_image_path,
                             out_label_path, **config_dict)
        else:
            imgpath_list = [os.path.join(args.image_dir, imgname)
                            for imgname in sorted(os.listdir(args.image_dir))
                            if os.path.splitext(imgname)[1].lower()
                            in supported_img_format]
            for file_patterns in batch_generator(imgpath_list, config.eval_batch_size):
                driver.inference(file_patterns, out_image_path,
                                 out_label_path, **config_dict)
    else:
        raise ValueError("{} does not exist. Please verify the input image or directory.".format(
            args.image_dir
        ))


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
            prog='infer', description='Run inference with an EfficientDet model.')

    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=False,
        help='Path to spec file. Absolute path or relative to working directory. \
            If not specified, default spec from spec_loader.py is used.')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to a trained EfficientDet model.'
    )
    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=True,
        help='Path to input image.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        default="",
        required=False,
        help='Key to save or load a .tlt model.'
    )
    parser.add_argument(
        '--class_map',
        type=str,
        required=False,
        help='Path to a text file where label mapping is stored. \
            Each row corresponds to a class label sorted by class id.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default='/tmp',
        required=False,
        help='Output directory where the status log is saved.'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.4,
        help='Confidence threshold for inference.'
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    try:
        main()
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
