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

"""Helper function to load model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from nvidia_tao_tf1.cv.common.utils import (
    CUSTOM_OBJS,
    load_keras_model
)
from nvidia_tao_tf1.encoding import encoding


def load_model(model_path, key=None):
    """Load a model either in .tlt format or .hdf5 format."""

    _, ext = os.path.splitext(model_path)

    if ext == '.hdf5':
        model = load_keras_model(model_path, custom_objects=CUSTOM_OBJS)

    elif ext == '.tlt':
        os_handle, temp_file_name = tempfile.mkstemp(suffix='.hdf5')
        os.close(os_handle)

        with open(temp_file_name, 'wb') as temp_file, open(model_path, 'rb') as encoded_file:
            encoding.decode(encoded_file, temp_file, key)
            encoded_file.close()
            temp_file.close()

        # recursive call
        model = load_model(temp_file_name, None)
        os.remove(temp_file_name)

    else:
        raise NotImplementedError("{0} file is not supported!".format(ext))
    return model


def save_model(keras_model, model_path, key, save_format=None):
    """Save a model to either .tlt or .hdf5 format."""

    _, ext = os.path.splitext(model_path)
    if (save_format is not None) and (save_format != ext):
        # recursive call to save a correct model
        return save_model(keras_model, model_path + save_format, key, None)

    if ext == '.hdf5':
        keras_model.save(model_path, overwrite=True, include_optimizer=True)
    else:
        raise NotImplementedError("{0} file is not supported!".format(ext))
    return model_path
