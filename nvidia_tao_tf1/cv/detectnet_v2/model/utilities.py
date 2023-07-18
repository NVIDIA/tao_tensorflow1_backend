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

"""Helper functions for different DetectNet V2 models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tempfile
from zipfile import BadZipFile, ZipFile

from keras import backend as K
import six

from nvidia_tao_tf1.cv.common.utils import decode_to_keras, load_keras_model
from nvidia_tao_tf1.encoding import encoding


logger = logging.getLogger(__name__)


def inference_learning_phase(fn):
    """Decorator that sets the learning phase to 0 temporarily before switching it back."""
    def _fn_wrapper(*args, **kwargs):
        old_learning_phase = K.learning_phase()
        try:
            K.set_learning_phase(0)
            return fn(*args, **kwargs)
        finally:
            # After everything is done, restore old learning phase.
            K.set_learning_phase(old_learning_phase)
    return _fn_wrapper


def get_class_predictions(predictions, target_class_names):
    """Helper for converting predictions dictionary to be indexed by class names.

    Args:
        predictions (dict): Dictionary of model predictions indexed by objective name.
        target_class_names (list): Model target class names as a list of strings.
    Returns:
        pred_dict: Dictionary of model predictions indexed by target class name and objective name.
    """
    pred_dict = {}
    num_target_classes = len(target_class_names)

    for target_class_index, target_class_name in enumerate(target_class_names):
        pred_dict[target_class_name] = {}
        for output_name, pred in six.iteritems(predictions):
            num_predicted_classes = int(pred.shape[1])
            assert num_predicted_classes == num_target_classes, \
                "Mismatch in the number of predicted (%d) and requested (%d) target_classes." % \
                (num_predicted_classes, num_target_classes)

            pred_dict[target_class_name][output_name] = pred[:, target_class_index]

    return pred_dict


def get_pretrained_model_path(model_file):
    """Get pretrained model file name and check it exists.

    If the supplied model file is not absolute it will be prepended with the
    data root. The data root is set according to current path.

    Args:
        model_file (string): Name of the stored model file (.hdf5).
    Returns:
        Absolute path to the model file if the input model_file is not an
            empty string. Else None.
    Raises:
        AssertionError if the model file does not exist.
    """
    if model_file:
        if not os.path.isabs(model_file):
            model_file = os.path.join(os.getcwd(),
                                      model_file)
        assert os.path.isfile(model_file), "Pretrained model file not found: %s" % model_file
    else:
        model_file = None

    return model_file


def model_io(model_path, enc_key=None):
    """Simple utility to handle model file based on file extensions.

    Args:
        pretrained_model_file (str): Path to the model file.
        enc_key (str): Key to load tlt file.

    Returns:
        model (keras.models.Model): Loaded keras model.
    """
    assert os.path.exists(model_path), "Pretrained model not found at {}".format(model_path)
    if model_path.endswith('.tlt'):
        assert enc_key is not None, "Key must be provided to load the model."
        model = decode_to_keras(str(model_path), bytes(enc_key, 'utf-8'))
    elif model_path.endswith('.hdf5'):
        model = load_keras_model(str(model_path), compile=False)
    else:
        raise NotImplementedError("Invalid model file extension. {}".format(model_path))
    return model


def extract_checkpoint_file(tmp_zip_file):
    """Simple function to extract a checkpoint file.

    Args:
        tmp_zip_file (str): Path to the extracted zip file.

    Returns:
        tmp_checkpoint_path (str): Path to the extracted checkpoint.
    """
    # Set-up the temporary directory.
    temp_checkpoint_path = tempfile.mkdtemp()
    try:
        with ZipFile(tmp_zip_file, 'r') as zip_object:
            for member in zip_object.namelist():
                zip_object.extract(member, path=temp_checkpoint_path)
    except BadZipFile:
        raise ValueError(
            "The zipfile extracted was corrupt. Please check your key "
            "or delete the latest `*.ckzip` and re-run the command."
        )
    except Exception:
        raise IOError(
            "The last checkpoint file is not saved properly. "
            "Please delete it and rerun the script."
        )
    return temp_checkpoint_path


def get_tf_ckpt(ckzip_path, enc_key, latest_step):
    """Simple function to extract and get a trainable checkpoint.

    Args:
        ckzip_path (str): Path to the encrypted checkpoint.

    Returns:
        tf_ckpt_path (str): Path to the decrypted tf checkpoint
    """
    os_handle, temp_zip_path = tempfile.mkstemp()
    os.close(os_handle)

    # Decrypt the checkpoint file.
    try:
        # Try reading a checkpoint file directly.
        temp_checkpoint_path = extract_checkpoint_file(ckzip_path)
    except ValueError:
        # Decrypt and load checkpoints for TAO < 5.0
        with open(ckzip_path, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zip_file:
            encoding.decode(encoded_file, tmp_zip_file, bytes(enc_key, 'utf-8'))
        encoded_file.closed
        tmp_zip_file.closed
        # Load zip file and extract members to a tmp_directory.
        temp_checkpoint_path = extract_checkpoint_file(temp_zip_path)
        # Removing the temporary zip path.
        os.remove(temp_zip_path)

    return os.path.join(temp_checkpoint_path,
                        "model.ckpt-{}".format(latest_step))
