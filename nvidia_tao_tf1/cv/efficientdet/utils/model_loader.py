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

"""Helper functions to load EfficientNet/EfficientDet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from zipfile import is_zipfile, ZipFile

import keras
import tensorflow as tf

from nvidia_tao_tf1.core.templates.utils_tf import swish
from nvidia_tao_tf1.cv.efficientdet.layers.image_resize_layer import ImageResizeLayer
from nvidia_tao_tf1.cv.efficientdet.layers.weighted_fusion_layer import WeightedFusion
from nvidia_tao_tf1.cv.efficientdet.utils import utils
from nvidia_tao_tf1.encoding import encoding

CUSTOM_OBJS = {
    'swish': swish,
    'PatchedBatchNormalization': utils.PatchedBatchNormalization,
    'ImageResizeLayer': ImageResizeLayer,
    'WeightedFusion': WeightedFusion}


def load_keras_model(keras_path, is_pruned):
    """Helper function to load keras or tf.keras model."""
    if is_pruned:
        tf.keras.models.load_model(keras_path, custom_objects=CUSTOM_OBJS)
    else:
        keras.models.load_model(keras_path, custom_objects=CUSTOM_OBJS)


def load_json_model(json_path, new_objs=None):
    """Helper function to load keras model from json file."""
    new_objs = new_objs or {}
    with open(json_path, 'r') as jf:
        model_json = jf.read()
    loaded_model = tf.keras.models.model_from_json(
        model_json,
        custom_objects={**CUSTOM_OBJS, **new_objs})
    return loaded_model


def dump_json(model, out_path):
    """Model to json."""
    with open(out_path, "w") as jf:
        jf.write(model.to_json())


def get_model_with_input(model_path, input_layer):
    """Implement a trick to replace input tensor."""

    def get_input_layer(*arg, **kargs):
        return input_layer
    return load_json_model(model_path, new_objs={'InputLayer': get_input_layer})


def decode_tlt_file(filepath, key):
    """Decrypt the tlt checkpoint file."""
    if filepath and filepath.endswith('.tlt'):
        if not is_zipfile(filepath):
            # support legacy .tlt model with encryption
            os_handle, temp_filepath = tempfile.mkstemp()
            os.close(os_handle)
            # Decrypt the checkpoint file.
            with open(filepath, 'rb') as encoded_file, open(temp_filepath, 'wb') as tmp_zipf:
                encoding.decode(encoded_file, tmp_zipf, key.encode())
        else:
            # .tlt is a zip file
            temp_filepath = filepath
        # if unencrypted tlt is a zip file, it is either from efficientdet ckpt or pruned
        # else it is from classification model
        if is_zipfile(temp_filepath):
            # create a temp to store extracted ckpt
            temp_ckpt_dir = tempfile.mkdtemp()
            # Load zip file and extract members to a tmp_directory.
            try:
                with ZipFile(temp_filepath, 'r') as zip_object:
                    hdf5_found = None
                    ckpt_found = None
                    for member in zip_object.namelist():
                        if member != 'checkpoint':
                            zip_object.extract(member, path=temp_ckpt_dir)
                            if member.endswith('.hdf5'):
                                # pruned model
                                hdf5_found = member
                            if '.ckpt-' in member:
                                step = int(member.split('.')[1].split('-')[-1])
                                ckpt_found = "model.ckpt-{}".format(step)
                    assert hdf5_found or ckpt_found, "The tlt file is in a wrong format."
            except Exception:
                raise IOError("The checkpoint file is not saved properly. \
                    Please delete it and rerun the script.")
            return os.path.join(temp_ckpt_dir, hdf5_found or ckpt_found)

        if os.path.exists(temp_filepath + '.hdf5'):
            os.remove(temp_filepath + '.hdf5')
        os.rename(temp_filepath, temp_filepath + '.hdf5')
        return temp_filepath + '.hdf5'
    return filepath
