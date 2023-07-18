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

"""Functions to load model for export and other model i/o utilities."""

import logging
import os
import tempfile
from zipfile import BadZipFile, ZipFile
import keras
import tensorflow as tf

from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.quantized_conv2dtranspose import QuantizedConv2DTranspose
from nvidia_tao_tf1.core.models.templates.quantized_dense import QuantizedDense
from nvidia_tao_tf1.core.models.templates.quantized_depthwiseconv2d import QuantizedDepthwiseConv2D
from nvidia_tao_tf1.cv.unet.model.build_unet_model import build_model
from nvidia_tao_tf1.cv.unet.model.build_unet_model import select_model_proto
from nvidia_tao_tf1.cv.unet.model.utilities import build_regularizer
from nvidia_tao_tf1.cv.unet.model.utilities import build_target_class_list, save_tmp_json
from nvidia_tao_tf1.cv.unet.model.utilities import initialize, initialize_params
from nvidia_tao_tf1.encoding import encoding

logger = logging.getLogger(__name__)

QAT_LAYERS = [
    QuantizedConv2D,
    QuantizedDepthwiseConv2D,
    QDQ,
    QuantizedDense,
    QuantizedConv2DTranspose,
]


def _extract_ckpt(encoded_checkpoint, key):
    """Get unencrypted checkpoint from tlt file."""

    temp_zip_path = None
    if os.path.isdir(encoded_checkpoint):
        temp_dir = encoded_checkpoint
    else:
        temp_dir = tempfile.mkdtemp()
        logger.info("Loading weights from {}".format(encoded_checkpoint))
        os_handle, temp_zip_path = tempfile.mkstemp()
        os.close(os_handle)

        # Decrypt the checkpoint file.
        with open(encoded_checkpoint, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zipf:
                encoding.decode(encoded_file, tmp_zipf, key.encode())
        encoded_file.closed
        tmp_zipf.closed

        # Load zip file and extract members to a tmp_directory.
        try:
            with ZipFile(temp_zip_path, 'r') as zip_object:
                for member in zip_object.namelist():
                    zip_object.extract(member, path=temp_dir)
        except BadZipFile:
            raise ValueError("Please double check your encryption key.")
        except Exception:
            raise IOError("The last checkpoint file is not saved properly. \
                Please delete it and rerun the script.")

    model_json = None
    json_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".json")]
    if len(json_files) > 0:
        model_json = json_files[0]
    meta_files = [f for f in os.listdir(temp_dir) if f.endswith(".meta")]
    # This functions is used duing inference/ eval. Means user passed pruned model.
    assert(len(meta_files) > 0), "In case of pruned model, please set the load_graph to true."

    if "-" in meta_files[0]:
        print(meta_files[0])
        step = int(meta_files[0].split('model.ckpt-')[-1].split('.')[0])
        # Removing the temporary zip path.
        if temp_zip_path:
            os.remove(temp_zip_path)
        return os.path.join(temp_dir, "model.ckpt-{}".format(step)), model_json
    if temp_zip_path:
        os.remove(temp_zip_path)
    return os.path.join(temp_dir, "model.ckpt"), model_json


def method_saver(latest):
    """A function to load weights to tensorflow graph."""

    sess = keras.backend.get_session()
    tf.global_variables_initializer()
    new_saver = tf.compat.v1.train.Saver()
    new_saver.restore(sess, latest)
    logger.info("Loaded weights Successfully for Export")


def load_keras_model(experiment_spec, model_path, export=False, key=None):
    """A function to load keras model."""

    initialize(experiment_spec)
    # Initialize Params
    params = initialize_params(experiment_spec)
    target_classes = build_target_class_list(
        experiment_spec.dataset_config.data_class_config)
    model_config = select_model_proto(experiment_spec)
    custom_objs = None
    unet_model = build_model(m_config=model_config,
                             target_class_names=target_classes)
    checkpoint_path, model_json = _extract_ckpt(model_path, key)

    if model_json:
        # If there is a json in tlt, it is a pruned model
        params["model_json"] = model_json
    kernel_regularizer, bias_regularizer = build_regularizer(
        experiment_spec.training_config.regularizer)
    # Constructing the unet model
    img_height, img_width, img_channels = experiment_spec.model_config.model_input_height, \
        experiment_spec.model_config.model_input_width, \
        experiment_spec.model_config.model_input_channels

    if params.enable_qat:
        # construct the QAT graph and save as a json file
        model_qat_json = unet_model.construct_model(
            input_shape=(img_channels, img_height, img_width),
            pretrained_weights_file=params.pretrained_weights_file,
            enc_key=params.key, model_json=params.model_json,
            features=None, construct_qat=True, custom_objs=custom_objs)
        model_qat_json = save_tmp_json(model_qat_json)
        params.model_json = model_qat_json

    unet_model.construct_model(input_shape=(img_channels, img_height, img_width),
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               pretrained_weights_file=params.pretrained_weights_file,
                               enc_key=key, export=export, model_json=params.model_json,
                               custom_objs=custom_objs)

    keras_model = unet_model.keras_model
    keras_model.summary()
    method_saver(checkpoint_path)
    return keras_model, custom_objs


def check_for_quantized_layers(model):
    """Check Keras model for quantization layers."""
    for layer in model.layers:
        if type(layer) in QAT_LAYERS:
            return True
    return False


def save_keras_model(keras_model, model_path, key, save_format=None):
    """Utility function to save pruned keras model as .tlt."""

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    # Saving session to the zip file.
    model_json = keras_model.to_json()
    with open(os.path.join(model_path, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    keras_model.save_weights(os.path.join(model_path, "model.h5"))
    saver = tf.train.Checkpoint()
    keras.backend.get_session()
    saver.save(os.path.join(model_path, "model.ckpt"))
    keras.backend.clear_session()
  
