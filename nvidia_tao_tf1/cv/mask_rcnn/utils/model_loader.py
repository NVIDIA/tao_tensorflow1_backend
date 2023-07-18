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

"""Helper functions to load MRCNN models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from zipfile import BadZipFile, ZipFile

import tensorflow as tf
from nvidia_tao_tf1.cv.common.utils import load_tf_keras_model
from nvidia_tao_tf1.cv.mask_rcnn.layers.anchor_layer import AnchorLayer
from nvidia_tao_tf1.cv.mask_rcnn.layers.box_input_layer import BoxInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.box_target_encoder import BoxTargetEncoder
from nvidia_tao_tf1.cv.mask_rcnn.layers.class_input_layer import ClassInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.foreground_selector_for_mask import \
    ForegroundSelectorForMask
from nvidia_tao_tf1.cv.mask_rcnn.layers.gpu_detection_layer import GPUDetections
from nvidia_tao_tf1.cv.mask_rcnn.layers.image_input_layer import ImageInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.info_input_layer import InfoInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.mask_input_layer import MaskInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.mask_postprocess_layer import MaskPostprocess
from nvidia_tao_tf1.cv.mask_rcnn.layers.mask_targets_layer import MaskTargetsLayer
from nvidia_tao_tf1.cv.mask_rcnn.layers.multilevel_crop_resize_layer import MultilevelCropResize
from nvidia_tao_tf1.cv.mask_rcnn.layers.multilevel_proposal_layer import MultilevelProposal
from nvidia_tao_tf1.cv.mask_rcnn.layers.proposal_assignment_layer import ProposalAssignment
from nvidia_tao_tf1.cv.mask_rcnn.layers.reshape_layer import ReshapeLayer
from nvidia_tao_tf1.encoding import encoding


custom_obj = {'AnchorLayer': AnchorLayer,
              'BoxTargetEncoder': BoxTargetEncoder,
              'ForegroundSelectorForMask': ForegroundSelectorForMask,
              'GPUDetections': GPUDetections,
              'MaskTargetsLayer': MaskTargetsLayer,
              'MaskPostprocess': MaskPostprocess,
              'MultilevelCropResize': MultilevelCropResize,
              'MultilevelProposal': MultilevelProposal,
              'ProposalAssignment': ProposalAssignment,
              'ReshapeLayer': ReshapeLayer}


def load_keras_model(keras_path):
    """Helper function to load keras model."""
    input_objs = {
        'ImageInput': ImageInput,
        'InfoInput': InfoInput,
        'ClassInput': ClassInput,
        'BoxInput': BoxInput,
        'MaskInput': MaskInput}
    model = load_tf_keras_model(keras_path,
                                custom_objects={**custom_obj, **input_objs})
    return model


def load_json_model(json_path):
    """Helper function to load keras model from json file."""
    input_objs = {
        'ImageInput': ImageInput,
        'InfoInput': InfoInput,
        'ClassInput': ClassInput,
        'BoxInput': BoxInput,
        'MaskInput': MaskInput}
    # load json and create model
    with open(json_path, 'r') as jf:
        model_json = jf.read()
    loaded_model = tf.keras.models.model_from_json(model_json,
                                                   custom_objects={**custom_obj, **input_objs})
    return loaded_model


def get_model_with_input(json_path,
                         input_layer1, input_layer2, input_layer3,
                         input_layer4, input_layer5,
                         input_layers=None):
    """Implement a trick to replace input tensor."""

    def get_input_layer1(*arg, **kargs):
        return input_layer1

    def get_input_layer2(*arg, **kargs):
        return input_layer2

    def get_input_layer3(*arg, **kargs):
        return input_layer3

    def get_input_layer4(*arg, **kargs):
        return input_layer4

    def get_input_layer5(*arg, **kargs):
        return input_layer5

    layer_names = ['ImageInput', 'InfoInput', 'BoxInput', 'ClassInput', 'MaskInput']
    # load json and create model
    with open(json_path, 'r') as jf:
        model_json = jf.read()

    if input_layer3 is not None:
        updated_dict = dict(
            zip(layer_names,
                [get_input_layer1, get_input_layer2,
                 get_input_layer3, get_input_layer4, get_input_layer5]))
    else:
        updated_dict = dict(zip(layer_names[:2], [get_input_layer1, get_input_layer2]))

    loaded_model = tf.keras.models.model_from_json(model_json,
                                                   custom_objects={**custom_obj, **updated_dict})
    # Following syntax only works in python3.
    return loaded_model


def dump_json(model, out_path):
    """Model to json."""
    with open(out_path, "w") as jf:
        jf.write(model.to_json())


def extract_checkpoint(ckpt_path, temp_ckpt_dir=None):
    """Extract checkpoint files."""
    temp_ckpt_dir = temp_ckpt_dir or tempfile.mkdtemp()
    try:
        with ZipFile(ckpt_path, 'r') as zip_object:
            for member in zip_object.namelist():
                zip_object.extract(member, path=temp_ckpt_dir)
                if member.startswith('model.ckpt-'):
                    step = int(member.split('model.ckpt-')[-1].split('.')[0])
    except BadZipFile:
        raise ValueError(
            "The zipfile extracted was corrupt. Please check your key "
            "or delete the latest `*.tlt` and re-run the command."
        )
    except Exception:
        raise IOError(
            "The last checkpoint file is not saved properly. "
            "Please delete it and rerun the script."
        )
    return os.path.join(temp_ckpt_dir, f"model.ckpt-{step}")


def load_mrcnn_tlt_model(ckpt_path, key=None, temp_ckpt_dir=None):
    """Load the MaskRCNN .tlt model file.

    Args:
        model_path (str): Path to the model file.
        key (str): Key to load the .tlt model.

    Returns:
        checkpoint_path (str): Path to the checkpoint file.
    """

    try:
        checkpoint_path = extract_checkpoint(ckpt_path, temp_ckpt_dir)
    except ValueError:
        os_handle, temp_zip_path = tempfile.mkstemp()
        os.close(os_handle)
        with open(ckpt_path, "rb") as encoded_model, open(temp_zip_path, "wb") as tmp_zfile:
            encoding.decode(encoded_model, tmp_zfile, key)
        assert encoded_model.closed, "Encoded file should be closed."
        assert tmp_zfile.closed, "Temporary zip file should be closed."
        checkpoint_path = extract_checkpoint(temp_zip_path, temp_ckpt_dir)
        os.remove(temp_zip_path)
    return checkpoint_path
