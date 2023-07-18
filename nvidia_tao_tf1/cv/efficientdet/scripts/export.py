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

"""Export EfficientDet model to etlt and TRT engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import shutil
import struct
import tempfile
from zipfile import ZipFile

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.util import deprecation

try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    from nvidia_tao_tf1.cv.efficientdet.exporter.trt_builder import EngineBuilder
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf1.cv.efficientdet.exporter.onnx_exporter import EfficientDetGraphSurgeon
from nvidia_tao_tf1.cv.efficientdet.inferencer import inference
from nvidia_tao_tf1.cv.efficientdet.utils import hparams_config
from nvidia_tao_tf1.cv.efficientdet.utils.model_loader import decode_tlt_file
from nvidia_tao_tf1.cv.efficientdet.utils.spec_loader import (
    generate_params_from_spec,
    load_experiment_spec
)
from nvidia_tao_tf1.encoding import encoding

deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']


def extract_zipfile_ckpt(zip_path):
    """Extract the contents of an efficientdet ckpt zip file.

    Args:
        zip_path (str): Path to a zipfile.

    Returns:
        checkpoint_path (str): Path to the checkpoint extracted.
    """
    temp_ckpt_dir = tempfile.mkdtemp()
    with ZipFile(zip_path, 'r') as zip_object:
        for member in zip_object.namelist():
            zip_object.extract(member, path=temp_ckpt_dir)
            if member.startswith('model.ckpt-'):
                step = int(member.split('model.ckpt-')[-1].split('.')[0])
    return os.path.join(temp_ckpt_dir, "model.ckpt-{}".format(step))


def extract_ckpt(encoded_checkpoint, key):
    """Get unencrypted checkpoint from tlt file."""
    logging.info("Loading weights from {}".format(encoded_checkpoint))
    try:
        # Load an unencrypted checkpoint as 5.0.
        checkpoint_path = extract_zipfile_ckpt(encoded_checkpoint)
    except BadZipFile:
        # Decrypt and load the checkpoint.
        os_handle, temp_zip_path = tempfile.mkstemp()
        os.close(os_handle)

        # Decrypt the checkpoint file.
        with open(encoded_checkpoint, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zipf:
            encoding.decode(encoded_file, tmp_zipf, key.encode())
        encoded_file.closed
        tmp_zipf.closed
        checkpoint_path = extract_zipfile_ckpt(temp_zip_path)
        os.remove(temp_zip_path)
    return checkpoint_path


def main(args=None):
    """Launch EfficientDet training."""
    disable_eager_execution()
    tf.autograph.set_verbosity(0)
    # parse CLI and config file
    args = parse_command_line_arguments(args)
    output_path = args.output_path
    if not ("onnx" in output_path):
        output_path = f"{output_path}.onnx"
    assert not os.path.exists(output_path), (
        f"Exported model already exists at \'{output_path}\'. "
        "Please change the output path or remove the current file."
    )
    assert args.max_batch_size > 0, "Max batch size for the engine must be positive."
    print("Loading experiment spec at %s.", args.experiment_spec_file)
    spec = load_experiment_spec(args.experiment_spec_file, merge_from_default=False)

    results_dir = os.path.dirname(args.output_path)
    status_file = os.path.join(results_dir, "status.json")
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
        message="Starting EfficientDet export."
    )
    # set up config
    MODE = 'export'
    # Parse and override hparams
    config = hparams_config.get_detection_config(spec.model_config.model_name)
    params = generate_params_from_spec(config, spec, MODE)
    config.update(params)
    if config.pruned_model_path:
        config.pruned_model_path = decode_tlt_file(config.pruned_model_path, args.key)

    # get output dir from etlt path
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pb_tmp_dir = tempfile.mkdtemp()
    # extract unencrypted checkpoint
    if args.model_path.endswith('.tlt'):
        ckpt_path = extract_ckpt(args.model_path, args.key)
    elif 'ckpt' in args.model_path:
        ckpt_path = args.model_path
    else:
        raise NotImplementedError(f"Invalid model file at {args.model_path}")
    # serve pb
    tf.enable_resource_variables()
    driver = inference.ServingDriver(
        config.name,
        ckpt_path,
        batch_size=args.max_batch_size,
        min_score_thresh=spec.eval_config.min_score_thresh or 0.4,
        max_boxes_to_draw=spec.eval_config.max_detections_per_image or 100,
        model_params=config.as_dict())
    driver.build()
    driver.export(pb_tmp_dir, tflite_path=None, tensorrt=None)
    # free gpu memory
    tf.reset_default_graph()
    # convert to onnx
    effdet_gs = EfficientDetGraphSurgeon(pb_tmp_dir, legacy_plugins=False)
    effdet_gs.update_preprocessor(
        [args.max_batch_size] + list(config.image_size) + [3])
    effdet_gs.update_network()
    effdet_gs.update_nms()
    # convert to etlt
    output_onnx_file = effdet_gs.save(output_path)
    if args.engine_file is not None or args.data_type == 'int8':
        if args.engine_file is None:
            engine_handle, temp_engine_path = tempfile.mkstemp()
            os.close(engine_handle)
            output_engine_path = temp_engine_path
        else:
            output_engine_path = args.engine_file

        builder = EngineBuilder(args.verbose, workspace=args.max_workspace_size)
        builder.create_network(output_onnx_file)
        builder.create_engine(
            output_engine_path,
            args.data_type,
            args.cal_image_dir,
            args.cal_cache_file,
            args.batch_size * args.batches,
            args.batch_size)
    # clean up tmp dir
    shutil.rmtree(pb_tmp_dir)
    print("Exported model is successfully exported at: {}".format(output_path))
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Export finished successfully."
    )


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='export', description='Export an EfficientDet model.')

    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        required=True,
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
        '-o',
        '--output_path',
        type=str,
        required=True,
        help='Path to the exported EfficientDet model.'
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
        "--data_type",
        type=str,
        default="fp32",
        help="Data type for the TensorRT export.",
        choices=["fp32", "fp16", "int8"])
    parser.add_argument(
        "--cal_image_dir",
        default="",
        type=str,
        help="Directory of images to run int8 calibration.")
    parser.add_argument(
        '--cal_cache_file',
        default=None,
        type=str,
        help='Calibration cache file to write to.')
    parser.add_argument(
        "--engine_file",
        type=str,
        default=None,
        help="Path to the exported TRT engine.")
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Max batch size for TensorRT engine builder.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of images per batch.")
    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Number of batches to calibrate over.")
    parser.add_argument(
        "--max_workspace_size",
        type=int,
        default=2,
        help="Max memory workspace size to allow in Gb for TensorRT engine builder (default: 2).")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbosity of the logger.")
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
            message="Export was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
