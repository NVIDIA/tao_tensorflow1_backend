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

"""Simple inference handler for TLT trained gridbox models serialized to TRT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tempfile
import traceback
import struct

import numpy as np

import keras
from keras import backend as K
import tensorflow as tf

from PIL import Image

import pytest

import pycuda.autoinit
import pycuda.driver as cuda

from nvidia_tao_tf1.core.export._uff import keras_to_uff

from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_function_parameters import get_target_class_names
from nvidia_tao_tf1.cv.detectnet_v2.inferencer.build_inferencer import build_inferencer
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import build_model
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec

from nvidia_tao_tf1.encoding import encoding

logger = logging.getLogger(__name__)

# Todo: <vpraveen> Use GB Feature extractor constructor to construct GB model and export
# to TRT serializable format for inference
detectnet_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
caffe_inferencer_spec = os.path.join(detectnet_root,
                                     "experiment_specs/inferencer_spec_caffe.prototxt")
etlt_inferencer_spec = os.path.join(detectnet_root,
                                    "experiment_specs/inferencer_spec_etlt.prototxt")
training_spec = os.path.join(detectnet_root,
                             "experiment_specs/default_spec.txt")

DEFAULT_MAX_WORKSPACE_SIZE = 1 << 30
ENC_KEY = 'tlt_encode'

topologies = [(etlt_inferencer_spec, 1, True, "resnet", 18, (3, 544, 960)),
              (etlt_inferencer_spec, 1, False, "resnet", 10, (3, 544, 960)),
              (etlt_inferencer_spec, 1, False, "vgg", 16, (3, 544, 960)),
              (etlt_inferencer_spec, 1, False, "efficientnet_b0", 16, (3, 544, 960))]

# Restricting the number of GPU's to be used by tensorflow to 0.
gpu_options = tf.compat.v1.GPUOptions(
    per_process_gpu_memory_fraction=0.33,
    allow_growth=True
)
device_count = {'GPU': 0, 'CPU': 1}
config = tf.compat.v1.ConfigProto(
    gpu_options=gpu_options,
    device_count=device_count
)
K.set_session(tf.Session(config=config))


def get_gridbox_tlt_model(arch, num_layers=None, input_shape=(3, 544, 960)):
    """Simple function to generate a TLT model."""
    experiment_spec = load_experiment_spec(training_spec)
    if hasattr(experiment_spec, "model_config"):
        model_config = experiment_spec.model_config
    else:
        raise ValueError("Invalid spec file without model_config at {}".format(training_spec))

    if hasattr(experiment_spec, "cost_function_config"):
        cost_function_config = experiment_spec.cost_function_config
    else:
        raise ValueError("Invalid spec without costfunction config at {}".format(training_spec))

    target_class_names = get_target_class_names(cost_function_config)
    model_config.arch = arch
    model_config.num_layers = num_layers
    gridbox_model = build_model(model_config, target_class_names)
    gridbox_model.construct_model(input_shape=input_shape,
                                  kernel_regularizer=None,
                                  bias_regularizer=None,
                                  pretrained_weights_file=None,
                                  enc_key=ENC_KEY)

    return gridbox_model


def convert_to_tlt(model,
                   output_node_names="output_bbox/BiasAdd,output_cov/Sigmoid"):
    """Simple function to generate etlt file from tlt file."""
    os_handle, tmp_uff_file_name = tempfile.mkstemp()
    os.close(os_handle)
    os_handler, tmp_etlt_file_name = tempfile.mkstemp()
    os.close(os_handle)

    # Convert keras to uff
    output_node_names = output_node_names.split(',')
    in_tensor_name, out_tensor_names, _ = keras_to_uff(model,
                                                       tmp_uff_file_name,
                                                       output_node_names=output_node_names)

    # We only support models with a single input tensor.
    if isinstance(in_tensor_name, list):
        in_tensor_name = in_tensor_name[0]
    K.clear_session()

    # Encode temporary uff to output file
    with open(tmp_uff_file_name, "rb") as open_temp_file, open(tmp_etlt_file_name,
                                                               "wb") as open_encoded_file:
        open_encoded_file.write(struct.pack("<i", len(in_tensor_name)))
        open_encoded_file.write(in_tensor_name.encode())
        encoding.encode(open_temp_file, open_encoded_file, ENC_KEY)

    os.remove(tmp_uff_file_name)
    return tmp_etlt_file_name


def get_inferencer_input(input_shape):
    """Simple function to get an input array.

    Args:
        input_shape (tuple): shape of the input array.

    Return:
        pil_input (pil.Image): A pil image object.
    """
    c = input_shape[0]
    h = input_shape[1]
    w = input_shape[2]
    np_input = np.random.random((h, w, c)) * 255
    pil_input = Image.fromarray(np_input.astype(np.uint8))
    return pil_input


def check_output(keras_output, trt_output, dtype='fp32', parser="caffe"):
    """Check keras and tensorrt inputs."""
    assert len(keras_output.keys()) == len(trt_output.keys())
    # ToDo <vpraveen> Check for output nodes of TensorRT and fine corresponding
    # Uff nodes that do match.
    if dtype == "fp32":
        for tclass in list(keras_output.keys()):
            np.array_equal(keras_output[tclass]['cov'],
                           trt_output[tclass]['cov'])
            np.array_equal(keras_output[tclass]['bbox'],
                           trt_output[tclass]['bbox'])


def get_keras_inferences(input_chunk, trt_inferencer, keras_model_path):
    """Get keras inferences for the current model.

    Args:
        input_chunk (list): list of PIL.Image objects to run inference on.
        trt_inferencer (nvidia_tao_tf1.cv.gridbox.inferencer.TRTInferencer): TRTInferencer object
            to run.
    Returns:
    """
    # Setting up inputs.
    input_shape = (3, trt_inferencer.image_height,
                   trt_inferencer.image_width)
    batch_size = len(input_chunk)
    infer_shape = (batch_size, ) + input_shape
    infer_input = np.zeros(infer_shape)
    assert os.path.exists(keras_model_path)
    keras_model = keras.models.load_model(keras_model_path, compile=False)
    graph = tf.get_default_graph()

    # preprocessing inputs
    for idx, image in enumerate(input_chunk):
        input_image, resized = trt_inferencer.input_preprocessing(image)
        infer_input[idx, :, :, :] = input_image

    # Inferring on the keras model.
    with graph.as_default():
        output = keras_model.predict(infer_input, batch_size=batch_size)

    infer_dict = trt_inferencer.predictions_to_dict(output)
    infer_out = trt_inferencer.keras_output_map(infer_dict)
    return infer_out, resized


def set_logger(verbose=False):
    info_level = 'INFO'
    if verbose:
        info_level = 'DEBUG'
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=info_level)


def prepare_test_input(arch, num_layers, input_shape):
    gridbox_model = get_gridbox_tlt_model(arch, num_layers=num_layers, input_shape=input_shape)
    tlt_model = gridbox_model.keras_model
    os_handle, tmp_tlt_path = tempfile.mkstemp()
    os.close(os_handle)
    tlt_model.save(tmp_tlt_path)
    etlt_model_path = convert_to_tlt(tlt_model)
    return tmp_tlt_path, etlt_model_path


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
@pytest.mark.parametrize("spec, batch_size, save_engine, arch, num_layers, input_shape", topologies)
def test_trt_inferencer(spec,
                        batch_size,
                        save_engine,
                        arch,
                        num_layers,
                        input_shape,
                        gpu_set=0):
    """Simple function to test trt inferencer engine.

    This function creates reads in a model template, creates and instance of TRT-inferencer,
    generates a TRT engine and then runs the model.

    Args:
        spec (string): Path to an inferencer spec file.
        batch_size (int): Number of images per batch of inference.
        save_engine (bool): Flag to save engine.
        gpu_set (int): Gpu device id to run inference under.
        arch (str): The architecture of the model under test.
        num_layers (int): Depth of the network if scalable.
        input_shape (tuple(ints)): Shape of the input in (C, H, W) format.

    Return:
        No explicit returns.
    """
    verbose = False
    n_batches = 1
    set_logger(verbose)

    tlt_model_path, etlt_model_path = prepare_test_input(arch, num_layers, input_shape)

    inference_spec = load_experiment_spec(spec, merge_from_default=False, validation_schema="inference")

    if hasattr(inference_spec, 'inferencer_config'):
        inferencer_config = inference_spec.inferencer_config
    else:
        raise ValueError("Invalid spec file provided at {}".format(spec))

    inferencer_config.tensorrt_config.etlt_model = etlt_model_path
    inferencer_config.image_height = input_shape[1]
    inferencer_config.image_width = input_shape[2]
    inferencer_config.batch_size = batch_size

    # Setup trt inferencer based on the test case topology.
    _, trt_inferencer = build_inferencer(inf_config=inferencer_config, 
                                         verbose=True,
                                         key=ENC_KEY)

    # Generate random inputs.
    infer_chunk = []
    for idx in range(batch_size):
        infer_chunk.append(get_inferencer_input(input_shape))

    # Generating inference for the keras model.
    keras_output, resized = get_keras_inferences(infer_chunk, trt_inferencer, tlt_model_path)
    K.clear_session()

    # Setup trt session and allocate buffers.
    trt_inferencer.network_init()
    trt_engine = trt_inferencer._engine_file

    # Run inference using TRT inferencer.
    trt_output, resized = trt_inferencer.infer_batch(infer_chunk)

    check_output(keras_output, trt_output, parser='etlt')

    if os.path.isfile(trt_engine):
        os.remove(trt_engine)

    # Free up session and buffers.
    trt_inferencer.clear_buffers()
    trt_inferencer.clear_trt_session()
