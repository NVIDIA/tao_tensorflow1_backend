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

"""Export APIs as defined in maglev sdk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import atexit
import logging
import os
import random
import sys
import struct
import tempfile

import numpy as np

from PIL import Image
import pycuda.driver as cuda
import pycuda.tools as tools

from nvidia_tao_tf1.core.export._tensorrt import CaffeEngineBuilder, Engine, UFFEngineBuilder
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.common.export.tensorfile_calibrator import \
    TensorfileCalibrator as Calibrator
from nvidia_tao_tf1.cv.common.export.tensorfile import TensorFile
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.encoding import encoding

from tqdm import tqdm

DEFAULT_MAX_WORKSPACE_SIZE = 1 << 30
DEFAULT_MAX_BATCH_SIZE = 100

logger = logging.getLogger(__name__)

def parse_etlt_model(etlt_model, key):
    """Parse etlt model file.

    Args:
        etlt_model (str): path to the etlt model file.
        key (str): String key to decode the model.

    Returns:
        uff_model (str): Path to the UFF model file."""
    if not os.path.exists(expand_path(etlt_model)):
            raise ValueError("Cannot find etlt file.")
    os_handle, tmp_uff_file = tempfile.mkstemp()
    os.close(os_handle)

    # Unpack etlt file.
    with open(expand_path(etlt_model), "rb") as efile:
        num_chars = efile.read(4)
        num_chars = struct.unpack("<i", num_chars)[0]
        input_node = str(efile.read(num_chars))
        with open(tmp_uff_file, "wb") as tfile:
            encoding.decode(efile, tfile, key.encode())
    logger.debug("Parsed ETLT model file.")
    return tmp_uff_file, input_node


def calibrate_fm_caffe(caffe_prototxt,
                       caffe_caffemodel,
                       uff_model,
                       etlt_model,
                       key,
                       input_dims=None,
                       output_node_names=None,
                       dtype='int8',
                       max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
                       max_batch_size=DEFAULT_MAX_BATCH_SIZE,
                       calibration_data_filename=None,
                       calibration_cache_filename=None,
                       calibration_n_batches=550,
                       calibration_batch_size=16,
                       parser='uff',
                       verbose=True,
                       trt_engine=True,
                       experiment_spec=None,
                       engine_serialize='engine.trt'):
    """Create a TensorRT engine out of a Keras model.
    NOTE: the current Keras session is cleared in this function.
    Do not use this function during training.

    Args:
        caffe_prototxt: (str) prototxt for caffe generated model.
        caffe_caffemodel: (str) caffemodel for caffe generated model.
        in_dims (list or dict): list of input dimensions, or a dictionary of
            input_node_name:input_dims pairs in the case of multiple inputs.
        output_node_names (list of str): list of model output node names as
            returned by model.layers[some_idx].get_output_at(0).name.split(':')[0].
            If not provided, then the last layer is assumed to be the output node.
        max_workspace_size (int): maximum TensorRT workspace size.
        max_batch_size (int): maximum TensorRT batch size.
        calibration_data_filename (str): calibratio data file to use.
        calibration_cache_filename (str): calibration cache file to write to.
        calibration_n_batches (int): number of calibration batches.
        calibration_batch_size (int): calibration batch size.
        parser='uff' (str): parser ('uff' or 'caffe') to use for intermediate representation.
        verbose (bool): whether to turn ON verbose messages.
    Returns:
        The names of the input and output nodes. These must be
        passed to the TensorRT optimization tool to identify
        input and output blobs. If multiple output nodes are specified,
        then a list of output node names is returned.
    """
    if dtype == 'int8':
        if calibration_data_filename is not  None:
            logger.info("Setting up calibrator")
            logger.info("Calibrator parameters: nbatches = {}, batch_size = {}".format(calibration_n_batches,
                                                                                calibration_batch_size
                                                                                ))
            calibrator = Calibrator(data_filename=calibration_data_filename,
                                    cache_filename=calibration_cache_filename,
                                    n_batches=calibration_n_batches,
                                    batch_size=calibration_batch_size)
        else:
            raise ValueError(
                "A valid calibration data filename or experiment spec file "
                "was required."
            )
    else:
        calibrator = None

    input_node_name = 'input_1'
    logger.info("Instantiated the calibrator")

    # Define model parser and corresponding engine builder.
    if parser == "caffe":
        assert os.path.exists(expand_path(caffe_caffemodel)), (
            "Caffemodel not found at {caffe_caffemodel}"
        )
        assert os.path.exists(expand_path(caffe_prototxt)), (
            "Prototxt model not found at {caffe_prototxt}"
        )
        logger.info(
        "Positional_args: {prototxt}, {caffemodel}, {dims}".format(
            caffemodel=caffe_caffemodel, prototxt=caffe_prototxt,
            dims=input_dims
        ))
        builder = CaffeEngineBuilder(caffe_prototxt,
                                     caffe_caffemodel,
                                     input_node_name,
                                     input_dims,
                                     output_node_names,
                                     max_batch_size=max_batch_size,
                                     max_workspace_size=max_workspace_size,
                                     dtype=dtype,
                                     verbose=verbose,
                                     calibrator=calibrator)
    elif parser == "uff":
        if not isinstance(input_dims, dict):
            input_dims = {input_node_name: input_dims}
        builder = UFFEngineBuilder(uff_model,
                                   input_node_name,
                                   input_dims,
                                   output_node_names=output_node_names,
                                   max_batch_size=max_batch_size,
                                   max_workspace_size=max_workspace_size,
                                   dtype=dtype,
                                   verbose=verbose,
                                   calibrator=calibrator)
    elif parser == "etlt":
        tmp_uff_model, _ = parse_etlt_model(etlt_model, key)
        if not isinstance(input_dims, dict):
            input_dims = {input_node_name: input_dims}
        logger.info(f"input_dims: {input_dims}")
        builder = UFFEngineBuilder(tmp_uff_model,
                                   input_node_name,
                                   input_dims,
                                   output_node_names=output_node_names,
                                   max_batch_size=max_batch_size,
                                   max_workspace_size=max_workspace_size,
                                   dtype=dtype,
                                   verbose=verbose,
                                   calibrator=calibrator)
    else:
        raise ValueError("Parser format not supported: {}".format(parser))

    engine = Engine(builder.get_engine())
    # write engine to file.
    if trt_engine:
        with open(expand_path(engine_serialize), 'wb') as efile:
            efile.write(engine._engine.serialize())
        efile.closed

    return input_node_name, output_node_names, engine


def clean_up(ctx):
    "Clear up context at exit."
    ctx.pop()
    tools.clear_context_caches()
    logger.info("Exiting execution. Thank you for using the calibrator.")


def prepare_chunk(image_ids, image_list,
                  image_width=960,
                  image_height=544,
                  channels=3,
                  scale=(1 / 255.),
                  batch_size=8):
    "Create a chunk for data for data for data dump to a tensorfile."
    dump_placeholder = np.zeros((batch_size, channels, image_height, image_width))
    for i in xrange(len(image_ids)):
        idx = image_ids[i]
        resized_image = Image.open(image_list[idx]).resize((image_width, image_height), Image.ANTIALIAS)
        dump_input = np.asarray(resized_image).astype(np.float32).transpose(2, 0, 1) * scale
        dump_placeholder[i, :, :, :] = dump_input
    return dump_placeholder


def create_data_dump(input_image_dir,
                     calibration_data_filename,
                     batch_size=16,
                     calibration_n_batches=500,
                     image_height=544,
                     image_width=960,
                     channels=3,
                     random_data=False):
    "Create a data dump tensorfile for calibration."
    # If random data then just dump random samples.
    if random_data:
        # Writing random dump file.
        with TensorFile(calibration_data_filename, 'w') as f:
            for _ in tqdm(xrange(calibration_n_batches)):
                f.write(np.random.sample((batch_size, ) + (batch_size,
                                                           image_height,
                                                           image_width)))
        f.closed
    else:
        # Else create dump from a directory of images.
        if not os.path.isdir(expand_path(input_image_dir)):
            raise ValueError("Need an valid image_dir for creating image_dump: {}".format(input_image_dir))
        num_images = calibration_n_batches * batch_size
        if os.path.is_dir(input_image_dir):
            image_idx = random.sample(xrange(len([item for item in os.listdir(expand_path(input_image_dir))
                                                if item.endswith('.jpg')])), num_images)
            image_list = [expand_path(f"{input_image_dir}/{image}") for image in os.listdir(expand_path(input_image_dir))
                        if image.endswith('.jpg')]
            # Writing out processed dump.
            with TensorFile(calibration_data_filename, 'w') as f:
                for chunk in tqdm(image_idx[x:x+batch_size] for x in xrange(0, len(image_idx), batch_size)):
                    dump_data = prepare_chunk(chunk, image_list, batch_size=batch_size)
                    f.write(dump_data)
            f.closed


def parse_command_line():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Int8 calibration table generator for Caffe/UFF models.')
    parser.add_argument('-d',
                        '--data_file_name',
                        help='The tensorrt calibration tensor file')
    parser.add_argument('-o',
                        '--output_node_names',
                        help='Comma separated node names to be marked as output blobs of the model.',
                        default='conv2d_cov/Sigmoid,conv2d_bbox')
    parser.add_argument('-p',
                        '--prototxt',
                        help='caffe inference prototxt file',
                        default=None)
    parser.add_argument('-c',
                        '--caffemodel',
                        help='caffe inference caffemodel file',
                        default=None)
    parser.add_argument('-u',
                        '--uff_model',
                        default=None,
                        help="Path to uff model file.")
    parser.add_argument('-bs',
                        '--batch_size',
                        help='Inference batch size, default=1',
                        type=int,
                        default=16)
    parser.add_argument('-b',
                        '--calibration_n_batches',
                        help="Flag to enable kitti dump",
                        type=int,
                        default=100)
    parser.add_argument('--input_dims',
                        nargs='+',
                        default=[3, 544, 960],
                        help='Input dimension. Forced to c h w',
                        type=int)
    parser.add_argument('--parser',
                        default='caffe',
                        help='Model parser to be called. Currently only caffe is supported.')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Logger verbosity level. ')
    parser.add_argument('--cache',
                        default=os.path.join(os.getcwd(),'cal_table.txt'),
                        help='Output calibration cache file')
    parser.add_argument('--create_dump',
                        action='store_true',
                        help='Create calibrator tensorfile from directory of images')
    parser.add_argument('--input_image_dir',
                        default=None,
                        help='Directory of input images to create a tensorfile',
                        type=str)
    parser.add_argument('--experiment_spec',
                        default=None,
                        help="Experiment spec file used to train the model.",
                        type=str)
    parser.add_argument('-g',
                        '--gpu_id',
                        default=0,
                        type=int,
                        help='Index of the GPU to work on')
    parser.add_argument("--etlt_model",
                        default=None,
                        type=str,
                        help="Path to the etlt model file.")
    parser.add_argument("--key",
                        default="tlt_encode",
                        help="Key to decode etlt model.",
                        default="",
                        type=str)
    parser.add_argument('--random_data',
                        action='store_true',
                        help="Calibrate on random data.")
    parser.add_argument('--trt_engine',
                        help='Save pre compiled trt engine',
                        action='store_true')
    parser.add_argument('--engine',
                        help='Path to save trt engine',
                        default=os.path.join(os.getcwd(), 'engine.trt'),
                        type=str)
    args = parser.parse_args()
    return args


def main():
    '''Main wrapper to generate calibration table from a pretrained caffe model.'''
    # Creating a cuda session.
    cuda.init()
    current_dev = cuda.Device(0)
    ctx = current_dev.make_context()
    ctx.push()

    # Parse parameters.
    args = parse_command_line()
    prototxt = args.prototxt
    caffemodel = args.caffemodel
    uff_model = args.uff_model
    etlt_model = args.etlt_model
    key = args.key
    trt_parser = args.parser
    calibration_data_filename = args.data_file_name
    calibration_cache_filename= args.cache
    output_node_names = args.output_node_names.split(',')
    (channels, image_height, image_width) = tuple(args.input_dims)
    batch_size = args.batch_size
    calibration_n_batches = args.calibration_n_batches
    input_image_dir = args.input_image_dir
    experiment_spec = args.experiment_spec

    # Defining logger configuration.
    verbosity = 'INFO'
    verbose = args.verbose
    if verbose:
        verbosity = 'DEBUG'
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=verbosity)

    logger.info("Shape: {} {} {}".format(channels, image_height, image_width))

    # Create a data dump file for calibration.
    if args.create_dump:
        logger.info("Creating data dump file for calibration process")
        create_data_dump(
            input_image_dir,
            calibration_data_filename,
            random_data=args.random_data,
            batch_size=batch_size,
            calibration_n_batches=calibration_n_batches,
            image_height=image_height,
            image_width=image_width,
            channels=channels
        )

    # Calibrate the model.
    input_node_name, output_node_names, engine = calibrate_fm_caffe(
        prototxt, caffemodel, uff_model, etlt_model, key,
        input_dims=tuple(args.input_dims),
        calibration_data_filename=calibration_data_filename,
        calibration_cache_filename=calibration_cache_filename,
        output_node_names=output_node_names,
        calibration_batch_size=batch_size,
        calibration_n_batches=args.calibration_n_batches,
        trt_engine=args.trt_engine,
        engine_serialize=args.engine,
        parser=trt_parser,
        experiment_spec=experiment_spec,
        verbose=verbose
    )
    del engine
    # Kill context at exit
    atexit.register(clean_up, ctx)


if __name__=="__main__":
    main()
