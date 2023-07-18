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

"""IVA common utils used across all apps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import lru_cache
import importlib
import io
import logging
import math
from math import exp, log
import os
import sys
import tempfile

from eff.core import Archive
import keras
from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l1, l2
from keras.utils.generic_utils import CustomObjectScope

import numpy as np

from PIL import Image, ImageDraw
import tensorflow as tf

from nvidia_tao_tf1.core.utils import set_random_seed
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.core.templates.utils import mish, swish
from nvidia_tao_tf1.cv.faster_rcnn.layers.custom_layers import (
    CropAndResize, NmsInputs,
    OutputParser, Proposal,
    ProposalTarget, TFReshape
)
from nvidia_tao_tf1.cv.retinanet.initializers.prior_prob import PriorProbability
from nvidia_tao_tf1.cv.retinanet.layers.anchor_box_layer import RetinaAnchorBoxes
from nvidia_tao_tf1.cv.ssd.layers.anchor_box_layer import AnchorBoxes
from nvidia_tao_tf1.cv.yolo_v3.layers.yolo_anchor_box_layer import YOLOAnchorBox
from nvidia_tao_tf1.cv.yolo_v4.layers.bbox_postprocessing_layer import BBoxPostProcessingLayer
from nvidia_tao_tf1.cv.yolo_v4.layers.split import Split
from nvidia_tao_tf1.encoding import encoding

ENCRYPTION_OFF = False
reg_dict = {0: None, 1: l1, 2: l2}
ap_mode_dict = {0: "sample", 1: "integrate"}

CUSTOM_OBJS = {'CropAndResize': CropAndResize,
               "NmsInputs": NmsInputs,
               'OutputParser': OutputParser,
               'Proposal': Proposal,
               'ProposalTarget': ProposalTarget,
               'TFReshape': TFReshape,
               'PriorProbability': PriorProbability,
               'RetinaAnchorBoxes': RetinaAnchorBoxes,
               'AnchorBoxes': AnchorBoxes,
               'YOLOAnchorBox': YOLOAnchorBox,
               'BBoxPostProcessingLayer': BBoxPostProcessingLayer,
               'swish': swish,
               'mish': mish,
               # loss is not needed if loaded from utils.
               # But the loss output must have gradient in TF1.15
               'compute_loss': lambda x, y: K.max(x) - K.max(y),
               'Split': Split}

# Define 1MB for filesize calculation.
MB = 1 << 20


@lru_cache()
def hvd_keras():
    """Lazily load and return the (cached) horovod module."""
    import horovod.keras as hvd

    return hvd


def raise_deprecation_warning(task, subtask, args):
    """Raise a deprecation warning based on the module.

    Args:
        task (str): The TLT task to be deprecated.
        subtask (str): The subtask supported by that task.
        args (list): List of arguments to be appended.

    Raises:
        DeprecationWarning: With the actual command to be run.
    """
    if not isinstance(args, list):
        raise TypeError("There should a list of arguments.")
    args_string = " ".join(args)
    new_command = "{} {} {}".format(
        task, subtask, args_string
    )
    raise DeprecationWarning(
        "This command has been deprecated in this version of TLT. "
        "Please run \n{}".format(new_command)
    )


def parse_arguments(cl_args, supported_tasks=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('module',
                        default='classification',
                        choices=supported_tasks)
    args, unknown_args = parser.parse_known_args(cl_args)
    args = vars(args)
    return args, unknown_args


def initialize(random_seed, hvd, training_precision='float32'):
    """Initialization.

    Args:
        random_seed: Random_seed in experiment spec.
        training_precision: (TrainingPrecision or None) Proto object with FP16/FP32 parameters or
            None. None leaves K.floatx() in its previous setting.
    """
    setup_keras_backend(training_precision, is_training=True)

    # Set Maglev random seed. Take care to give different seed to each process.
    seed = random_seed + hvd.rank()
    set_random_seed(seed)


def get_num_params(model):
    """Get the number of parameters in a model.

    Args:
        model(keras.model.Model): Model object to run count params.

    Returns:
        num_params(int): Number of parameters in a model. Represented
        in units per million.
    """
    return model.count_params()/1e6


def get_model_file_size(model_path):
    """Get the size of the model.

    Args:
        model_path (str): UNIX path to the model.

    Returns:
        file_size (float): File size in MB.
    """
    if not os.path.exists(expand_path(model_path)):
        raise FileNotFoundError(f"Model file wasn't found at {model_path}")
    file_size = os.path.getsize(model_path) / MB
    return file_size


def setup_keras_backend(training_precision, is_training):
    """Setup Keras-specific backend settings for training or inference.

    Args:
        training_precision: (TrainingPrecision or None) Proto object with FP16/FP32 parameters or
            None. None leaves K.floatx() in its previous setting.
        is_training: (bool) If enabled, Keras is set in training mode.
    """
    # Learning phase of '1' indicates training mode -- important for operations
    # that behave differently at training/test times (e.g. batch normalization)
    if is_training:
        K.set_learning_phase(1)
    else:
        K.set_learning_phase(0)

    # Set training precision, if given. Otherwise leave K.floatx() in its previous setting.
    # K.floatx() determines how Keras creates weights and casts them (Keras default: 'float32').
    if training_precision is not None:
        if training_precision == 'float32':
            K.set_floatx('float32')
        elif training_precision == 'float16':
            K.set_floatx('float16')
        else:
            raise RuntimeError('Invalid training precision selected')


def summary_from_value(tag, value, scope=None):
    """Generate a manual simple summary object with a tag and a value."""
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    if scope:
        summary_value.tag = '{}/{}'.format(scope, tag)
    else:
        summary_value.tag = tag
    return summary


def summary_from_image(
    summary,
    tag,
    value,
    box,
    scope=None,
    img_means=(103.939, 116.779, 123.68),
    channels_first=True,
    reverse_channels=True,
    idx=0
):
    """Summary from single image."""
    summary_value = summary.value.add()
    summary_value.image.height = value.shape[0]
    summary_value.image.width = value.shape[1]
    # de-preprocessing to get the INT8 image
    img_means = np.array(img_means)
    lambda_gray = np.array([0.1140, 0.5870, 0.2990])
    if channels_first:
        n_channels = value.shape[0]
        summary_value.image.colorspace = n_channels
        if n_channels == 3:
            img_means = img_means.reshape(3, 1, 1)
            value = value + img_means
            value = value.transpose(1, 2, 0)
            if reverse_channels:
                value = value[..., [2, 1, 0]]
        else:
            delta = np.dot(img_means.reshape(1, 3), lambda_gray.reshape(3, 1))
            value = value + delta
            value = value.transpose(1, 2, 0)
    else:
        n_channels = value.shape[-1]
        summary_value.image.colorspace = n_channels
        if n_channels == 3:
            img_means = img_means.reshape(1, 1, 3)
            value = value + img_means
            if reverse_channels:
                value = value[..., [2, 1, 0]]
        else:
            delta = np.dot(img_means.reshape(1, 3), lambda_gray.reshape(3, 1))
            value = value + delta
    value = value.astype(np.uint8)
    image = Image.fromarray(np.squeeze(value))
    draw = ImageDraw.Draw(image)
    h, w = value.shape[:2]
    box = box * np.array([w, h, w, h])
    box = box.astype(np.int32)
    for b in box:
        draw.rectangle(
            ((b[0], b[1]), (b[2], b[3])),
            outline="Black"
        )
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    summary_value.image.encoded_image_string = img_byte_arr
    if scope:
        summary_value.tag = '{}/{}/image/{}'.format(scope, tag, idx)
    else:
        summary_value.tag = '{}/image/{}'.format(tag, idx)


def summary_from_images(
    tag,
    value,
    boxes,
    scope=None,
    img_means=(103.939, 116.779, 123.68),
    channels_first=True,
    reverse_channels=True,
    max_num=3
):
    """Generate a manual image summary object with a tag and a value."""
    summary = tf.Summary()
    for idx, img in enumerate(value):
        if idx < max_num:
            summary_from_image(
                summary,
                tag,
                img,
                boxes[idx],
                scope,
                img_means,
                channels_first,
                reverse_channels,
                idx
            )
    return summary


def tensorboard_images(
    tag,
    value,
    boxes,
    writer,
    step,
    scope=None,
    img_means=(103.939, 116.779, 123.68),
    channels_first=True,
    reverse_channels=True,
    max_num=3
):
    """Vis images in TensorBoard."""
    summary = summary_from_images(
        tag,
        value,
        boxes,
        scope,
        img_means,
        channels_first,
        reverse_channels,
        max_num
    )
    writer.add_summary(summary, step)
    writer.flush()


def encode_from_keras(keras_model, output_filename, enc_key, only_weights=False,
                      custom_objects=None):
    """A simple function to encode a keras model into magnet export format.

    Args:
        keras_model (keras.models.Model object): The input keras model to be encoded.
        output_file_name (str): The name of the encoded output file.
        enc_key (bytes): Byte text to encode the model.
        custom_objects(dict): Custom objects for serialization and deserialization.

    Returns:
        None
    """
    # TODO(madil): Ensure switched off for production.
    custom_objs = dict()
    custom_objs.update(CUSTOM_OBJS)
    if custom_objects is not None:
        custom_objs.update(custom_objects)

    if output_filename.endswith(".hdf5"):
        with CustomObjectScope(custom_objs):
            if only_weights:
                keras_model.save_weights(output_filename)
            else:
                keras_model.save(output_filename)
        return

    # Make sure that input model is a keras model object.
    if not isinstance(keras_model, keras.models.Model):
        raise TypeError("The model should be a keras.models.Model object")

    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)

    # Create a temporary model file for the keras model.
    with CustomObjectScope(custom_objs):
        if only_weights:
            keras_model.save_weights(temp_file_name)
        else:
            keras_model.save(temp_file_name)

    # Encode the keras model file.
    with open(expand_path(output_filename), 'wb') as outfile, open(temp_file_name, 'rb') as infile:
        encoding.encode(infile, outfile, enc_key)
    infile.closed
    outfile.closed
    # Remove the temporary keras file.
    os.remove(temp_file_name)


def get_decoded_filename(input_file_name, enc_key, custom_objects=None):
    """Extract keras model file and get model dtype.

    Args:
        input_file_name (str): Path to input model file.
        enc_key (bytes): Byte text to decode model.
        custom_objects(dict): Custom objects for serialization and deserialization.

    Returns:
        model_dtype: Return the decoded model filename.
    """
    if input_file_name.endswith(".hdf5"):
        return input_file_name
    custom_objs = dict()
    custom_objs.update(CUSTOM_OBJS)
    if custom_objects is not None:
        custom_objs.update(custom_objects)

    if ENCRYPTION_OFF:
        return input_file_name

    # Check if input file exists.
    if not os.path.isfile(input_file_name):
        raise ValueError("Cannot find input file name.")

    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)

    with open(temp_file_name, 'wb') as temp_file, open(input_file_name, 'rb') as encoded_file:
        encoding.decode(encoded_file, temp_file, enc_key)
    encoded_file.closed
    temp_file.closed

    # Check if the model is valid hdf5
    try:
        with CustomObjectScope(custom_objs):
            keras.models.load_model(temp_file_name, compile=False)
    except (OSError, IOError):
        sys.exit("Invalid decryption. {}. The key used to load the model "
                 "is incorrect.".format(sys.exc_info()[1]))
    except ValueError:
        raise ValueError("Invalid decryption. {}. The key used to load the model "
                         "is incorrect.".format(sys.exc_info()[1]))

    return temp_file_name


def decode_to_keras(input_file_name, enc_key,
                    input_model=None, compile_model=True, by_name=True,
                    custom_objects=None):
    """A simple function to decode an encrypted file to a keras model.

    Args:
        input_file_name (str): Path to encoded input file.
        enc_key (bytes): Byte text to decode the model.
        custom_objects(dict): Custom objects for serialization and deserialization.

    Returns:
        decrypted_model (keras.models.Model): Returns a decrypted keras model.
    """
    custom_objs = dict()
    custom_objs.update(CUSTOM_OBJS)
    if custom_objects is not None:
        custom_objs.update(custom_objects)

    if input_file_name.endswith(".hdf5"):
        with CustomObjectScope(custom_objs):
            if input_model is None:
                return keras.models.load_model(input_file_name, compile=compile_model)
            assert isinstance(input_model, keras.models.Model), (
                "Input model not a valid Keras model."
            )
            input_model.load_weights(input_file_name, by_name=by_name)
            return input_model

    # Check if input file exists.
    if not os.path.isfile(expand_path(input_file_name)):
        raise ValueError("Cannot find input file name.")

    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)

    with open(temp_file_name, 'wb') as temp_file, open(expand_path(input_file_name), 'rb') as encoded_file:
        encoding.decode(encoded_file, temp_file, enc_key)
    encoded_file.closed
    temp_file.closed

    if input_model is None:
        try:
            # Patch for custom layers.
            with CustomObjectScope(custom_objs):
                decrypted_model = keras.models.load_model(temp_file_name, compile=compile_model)
        except (OSError, IOError):
            sys.exit("Invalid decryption. {}. The key used to load the model "
                     "is incorrect.".format(sys.exc_info()[1]))
        except ValueError:
            raise ValueError("Invalid decryption. {}".format(sys.exc_info()[1]))

        os.remove(temp_file_name)
        return decrypted_model
    assert isinstance(input_model, keras.models.Model), 'Input model not a valid Keras moodel.'
    try:
        # Patch for custom layers.
        with CustomObjectScope(custom_objs):
            input_model.load_weights(temp_file_name, by_name=by_name)
    except (OSError, IOError):
        sys.exit("Invalid decryption. {}. The key used to load the model "
                 "is incorrect.".format(sys.exc_info()[1]))
    except ValueError:
        raise ValueError("Invalid decryption. {}. The key used to load the model "
                         "is incorrect.".format(sys.exc_info()[1]))

    os.remove(temp_file_name)
    return input_model


def model_io(model_path, enc_key=None, custom_objects=None, compile=False):
    """Simple utility to handle model file based on file extensions.

    Args:
        pretrained_model_file (str): Path to the model file.
        enc_key (str): Key to load tlt file.

    Returns:
        model (keras.models.Model): Loaded keras model.
    """
    custom_objs = dict()
    custom_objs.update(CUSTOM_OBJS)
    if custom_objects is not None:
        custom_objs.update(custom_objects)

    assert os.path.exists(
        model_path), "Model not found at {}".format(model_path)
    if model_path.endswith('.tlt'):
        assert enc_key is not None, "Key must be provided to load the model."
        return decode_to_keras(str(model_path),
                               enc_key,
                               custom_objects=custom_objs)
    elif model_path.endswith('.hdf5'):
        with CustomObjectScope(custom_objs):
            return keras.models.load_model(str(model_path),
                                           compile=compile)
    else:
        raise NotImplementedError(
            "Invalid model file extension. {}".format(model_path))


def deserialize_custom_layers(art):
    """Deserialize the code for custom layer from EFF.

    Args:
        art (eff.core.artifact.Artifact): Artifact restored from EFF Archive.

    Returns:
        final_dict (dict): Dictionary representing CUSTOM_OBJS used in the EFF stored Keras model.
    """
    # Get class.
    source_code = art.get_content()
    spec = importlib.util.spec_from_loader('helper', loader=None)
    helper = importlib.util.module_from_spec(spec)
    exec(source_code, helper.__dict__) # noqa pylint: disable=W0122

    final_dict = {}
    # Get class name from attributes.
    class_names = art["class_names"]
    for cn in class_names:
        final_dict[cn] = getattr(helper, cn)
    return final_dict


def restore_eff(eff_path, passphrase=None):
    """Restore Keras Model from EFF Archive.

    Args:
        eff_path (str): Path to the eff file.
        passphrase (str): Key to load EFF file.

    Returns:
        model (keras.models.Model): Loaded keras model.
        EFF_CUSTOM_OBJS (dict): Dictionary of custom layers from the eff file.
    """
    model_name = os.path.basename(eff_path).split(".")[0]
    with Archive.restore_from(restore_path=eff_path, passphrase=passphrase) as restored_effa:
        EFF_CUSTOM_OBJS = deserialize_custom_layers(restored_effa.artifacts['custom_layers.py'])

        art = restored_effa.artifacts['{}.hdf5'.format(model_name)]
        weights, m = art.get_content()

    with CustomObjectScope(EFF_CUSTOM_OBJS):
        model = keras.models.model_from_json(m, custom_objects=EFF_CUSTOM_OBJS)
        model.set_weights(weights)

    return model, EFF_CUSTOM_OBJS


def load_keras_model(
        filepath, custom_objects=None, compile=True):  # pylint: disable=redefined-builtin
    """Wrap keras load model to catch incorrect keywords error."""
    if not os.path.exists(expand_path(filepath)):
        raise FileNotFoundError(f"Model not found: {filepath}")
    try:
        return keras.models.load_model(filepath, custom_objects, compile=compile)
    except (OSError, IOError):
        raise ValueError(
            f"Invalid model: {filepath}, please check the key used to load the model"
        )


def load_tf_keras_model(
        filepath, custom_objects=None, compile=True):  # pylint: disable=redefined-builtin
    """Wrap tf keras load model to catch incorrect keywords error."""
    try:
        return tf.keras.models.load_model(filepath, custom_objects, compile=compile)
    except (OSError, IOError):
        sys.exit("Invalid decryption. {}. The key used to load the model "
                 "is incorrect.".format(sys.exc_info()[1]))


def build_regularizer_from_config(reg_config):
    '''Build Keras regularizer based on config protobuf.'''

    reg_type = reg_config.type
    reg_weight = reg_config.weight
    kr = None
    if reg_type and reg_type > 0:
        assert 0 < reg_weight < 1, "Weight decay should be no less than 0 and less than 1"
        kr = reg_dict[reg_type](reg_weight)
    return kr


def build_optimizer_from_config(optim_config, **kwargs):
    '''Build Keras optimizer based on config protobuf.'''

    optim_type = optim_config.WhichOneof('optimizer')
    assert optim_type, "Optimizer must be specified in config file!"

    cfg = getattr(optim_config, optim_type)
    if optim_type == 'adam':
        assert 1 > cfg.beta1 > 0, "beta1 must be within (0, 1)."
        assert 1 > cfg.beta2 > 0, "beta2 must be within (0, 1)."
        assert cfg.epsilon > 0, "epsilon must be greater than 0."
        optim = Adam(beta_1=cfg.beta1, beta_2=cfg.beta2, epsilon=cfg.epsilon,
                     amsgrad=cfg.amsgrad, **kwargs)
    elif optim_type == 'sgd':
        assert cfg.momentum >= 0, "momentum must be >=0."
        optim = SGD(momentum=cfg.momentum, nesterov=cfg.nesterov, **kwargs)
    elif optim_type == 'rmsprop':
        assert 1 > cfg.beta2 > 0, "rho must be within (0, 1)."
        assert cfg.momentum >= 0, "momentum must be >=0."
        assert cfg.epsilon > 0, "epsilon must be greater than 0."
        optim = RMSprop(rho=cfg.rho, momentum=cfg.momentum, epsilon=cfg.epsilon,
                        centered=cfg.centered, **kwargs)
    else:
        raise NotImplementedError("The optimizer specified is not implemented!")
    return optim


def build_lrs_from_config(lrs_config, max_iterations, lr_multiplier):
    '''
    Build Keras learning schedule based on config protobuf.

    Args:
        lrs_config: LearningRateConfig
        max_iterations: max iterations of training
        lr_multiplier: lr = config.lr * lr_multiplier

    Returns:
        lr_schedule as keras.callback
    '''

    lrs_type = lrs_config.WhichOneof('learning_rate')
    assert lrs_type, "learning rate schedule must be specified in config file!"

    cfg = getattr(lrs_config, lrs_type)
    assert cfg.min_learning_rate > 0.0, "min_learning_rate should be positive"
    assert cfg.max_learning_rate > cfg.min_learning_rate, \
        "max learning rate should be larger than min_learning_rate"
    if lrs_type == 'soft_start_annealing_schedule':
        lrs = SoftStartAnnealingLearningRateScheduler(
            max_iterations=max_iterations,
            base_lr=cfg.max_learning_rate * lr_multiplier,
            min_lr_ratio=cfg.min_learning_rate / cfg.max_learning_rate,
            soft_start=cfg.soft_start,
            annealing_start=cfg.annealing)
    elif lrs_type == 'soft_start_cosine_annealing_schedule':
        lrs = SoftStartCosineAnnealingScheduler(
            base_lr=cfg.max_learning_rate * lr_multiplier,
            min_lr_ratio=cfg.min_learning_rate / cfg.max_learning_rate,
            soft_start=cfg.soft_start,
            max_iterations=max_iterations)
    else:
        raise NotImplementedError("The Learning schedule specified is not implemented!")
    return lrs


def parse_model_load_from_config(train_config):
    '''Parse model loading config from protobuf.

    Input:
        the protobuf config at training_config level.
    Output
        model_path (string): the path of model to be loaded. None if not given
        load_graph (bool): Whether to load whole graph. If False, will need to recompile the model
        reset_optim (bool): Whether to reset optim. This field must be true if load_graph is false.
        initial_epoch (int): the starting epoch number. 0 - based
    '''

    load_type = train_config.WhichOneof('load_model')
    if load_type is None:
        return None, False, True, 0
    if load_type == 'resume_model_path':
        try:
            epoch = int(train_config.resume_model_path.split('.')[-2].split('_')[-1])
        except Exception:
            raise ValueError("Cannot parse the checkpoint path. Did you rename it?")
        return train_config.resume_model_path, True, False, epoch
    if load_type == 'pretrain_model_path':
        return train_config.pretrain_model_path, False, True, 0
    if load_type == 'pruned_model_path':
        return train_config.pruned_model_path, True, True, 0
    raise ValueError("training configuration contains invalid load_model type.")


def check_tf_oom(func):
    '''A decorator function to check OOM and raise informative errors.'''

    def return_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if type(e) == tf.errors.ResourceExhaustedError:
                logger = logging.getLogger(__name__)
                logger.error(
                    "Ran out of GPU memory, please lower the batch size, use a smaller input "
                    "resolution, use a smaller backbone, or enable model parallelism for "
                    "supported TLT architectures (see TLT documentation)."
                )
                sys.exit(1)
            else:
                # throw out the error as-is if they are not OOM error
                raise e
    return return_func


class StepLRScheduler(keras.callbacks.Callback):
    """Step learning rate annealing schedule.

    This callback implements the step learning rate annnealing schedule according to
    the progress of the current experiment. The training progress is defined as the
    ratio of the current iteration to the maximum iterations. The scheduler adjusts the
    learning rate of the experiment in steps at regular intervals.

    Args:
        base lr: Learning rate at the start of the experiment
        gamma : ratio by which the learning rate reduces at every steps
        step_size : step size as percentage of maximum iterations
        max_iterations : Total number of iterations in the current experiment
                         phase
    """

    def __init__(self, base_lr=1e-2, gamma=0.1, step_size=33, max_iterations=12345):
        """__init__ method."""
        super(StepLRScheduler, self).__init__()

        if not 0.0 <= step_size <= 100.0:
            raise ValueError('StepLRScheduler ' 'does not support a step size < 0.0 or > 100.0')
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('StepLRScheduler ' 'does not support gamma < 0.0 or > 1.0')
        self.base_lr = base_lr
        self.gamma = gamma
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global_step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """Start of training method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max iterations."""
        if not 0. <= progress <= 1.:
            raise ValueError('StepLRScheduler '
                             'does not support a progress value < 0.0 or > 1.0 '
                             'received (%f)' % progress)

        numsteps = self.max_iterations * self.step_size // 100
        exp_factor = self.global_step / numsteps
        lr = self.base_lr * pow(self.gamma, exp_factor)
        return lr


class MultiGPULearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler implementation.

    Implements https://arxiv.org/pdf/1706.02677.pdf (Accurate, Large Minibatch SGD:
    Training ImageNet in 1 Hour) style learning rate schedule.
    Learning rate scheduler modulates learning rate according to the progress in the
    training experiment. Specifically the training progress is defined as the ratio of
    the current iteration to the maximum iterations. Learning rate scheduler adjusts
    learning rate in the following phases:
        Phase 1: 0.0 <= progress < soft_start:
                 Starting from start_lr linearly increase the learning rate to base_lr.
        Phase 2: at every annealing point, divide learning rate by annealing divider.

    Example:
        ```python
        lrscheduler = MultiGPULearningRateScheduler(
            max_iterations=max_iterations)

        model.fit(X_train, Y_train, callbacks=[lrscheduler])
        ```

    Args:
        max_iterations: Total number of iterations in the experiment.
        start_lr: Learning rate at the beginning. In the paper this is the learning rate used
                  with single GPU training.
        base_lr: Maximum learning rate. In the paper base_lr is set as start_lr * number of
                 GPUs.
        soft_start: The progress at which learning rate achieves base_lr when starting from
                    start_lr. Default value set as in the paper.
        annealing_points: A list of progress values at which learning rate is divided by
                          annealing_divider. Default values set as in the paper.
        annealing_divider: A divider for learning rate applied at each annealing point.
                           Default value set as in the paper.
    """

    def __init__(  # pylint: disable=W0102
            self,
            max_iterations,
            start_lr=3e-4,
            base_lr=5e-4,
            soft_start=0.056,
            annealing_points=[0.33, 0.66, 0.88],
            annealing_divider=10.0):
        """__init__ method."""
        super(MultiGPULearningRateScheduler, self).__init__()

        if not 0.0 <= soft_start <= 1.0:
            raise ValueError('The soft_start varible should be >= 0.0 or <= 1.0.')
        prev = 0.
        for p in annealing_points:
            if not 0.0 <= p <= 1.0:
                raise ValueError('annealing_point should be >= 0.0 or <= 1.0.')
            if p < prev:
                raise ValueError('annealing_points should be in increasing order.')
            if not soft_start < p:
                raise ValueError('soft_start should be less than the first annealing point.')
            prev = p

        self.start_lr = start_lr
        self.base_lr = base_lr
        self.soft_start = soft_start  # Increase to lr from start_lr until this point.
        self.annealing_points = annealing_points  # Divide lr by annealing_divider at these points.
        self.annealing_divider = annealing_divider
        self.max_iterations = max_iterations
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global_step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """on_train_begin method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max_iterations."""
        if not 0. <= progress <= 1.:
            raise ValueError('MultiGPULearningRateScheduler '
                             'does not support a progress value < 0.0 or > 1.0 '
                             'received (%f)' % progress)

        if not self.base_lr:
            return self.base_lr

        lr = self.base_lr
        if progress < self.soft_start:
            soft_start = progress / self.soft_start
            lr = soft_start * self.base_lr + (1. - soft_start) * self.start_lr
        else:
            for p in self.annealing_points:
                if progress > p:
                    lr /= self.annealing_divider

        return lr


class SoftStartAnnealingLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler implementation.

    Learning rate scheduler modulates learning rate according to the progress in the
    training experiment. Specifically the training progress is defined as the ratio of
    the current iteration to the maximum iterations. Learning rate scheduler adjusts
    learning rate in the following 3 phases:
        Phase 1: 0.0 <= progress < soft_start:
                 Starting from min_lr exponentially increase the learning rate to base_lr
        Phase 2: soft_start <= progress < annealing_start:
                 Maintain the learning rate at base_lr
        Phase 3: annealing_start <= progress <= 1.0:
                 Starting from base_lr exponentially decay the learning rate to min_lr

    Example:
        ```python
        lrscheduler = SoftStartAnnealingLearningRateScheduler(
            max_iterations=max_iterations)

        model.fit(X_train, Y_train, callbacks=[lrscheduler])
        ```

    Args:
        base_lr: Maximum learning rate
        min_lr_ratio: The ratio between minimum learning rate (min_lr) and base_lr
        soft_start: The progress at which learning rate achieves base_lr when starting from min_lr
        annealing_start: The progress at which learning rate starts to drop from base_lr to min_lr
        max_iterations: Total number of iterations in the experiment
    """

    def __init__(self, max_iterations, base_lr=5e-4, min_lr_ratio=0.01, soft_start=0.1,
                 annealing_start=0.7):
        """__init__ method."""
        super(SoftStartAnnealingLearningRateScheduler, self).__init__()

        if not 0.0 <= soft_start <= 1.0:
            raise ValueError('The soft_start variable should be >= 0.0 or <= 1.0.')
        if not 0.0 <= annealing_start <= 1.0:
            raise ValueError('The annealing_start variable should be >= 0.0 or <= 1.0.')
        if not soft_start < annealing_start:
            raise ValueError('Variable soft_start should be less than annealing_start.')

        self.base_lr = base_lr
        self.min_lr_ratio = min_lr_ratio
        self.soft_start = soft_start  # Increase to lr from min_lr until this point.
        self.annealing_start = annealing_start  # Start annealing to min_lr at this point.
        self.max_iterations = max_iterations
        self.min_lr = min_lr_ratio * base_lr
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global_step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """on_train_begin method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max_iterations."""
        if not 0. <= progress <= 1.:
            raise ValueError('SoftStartAnnealingLearningRateScheduler '
                             'does not support a progress value < 0.0 or > 1.0 '
                             'received (%f)' % progress)

        if not self.base_lr:
            return self.base_lr

        if self.soft_start > 0.0:
            soft_start = progress / self.soft_start
        else:  # learning rate starts from base_lr
            soft_start = 1.0

        if self.annealing_start < 1.0:
            annealing = (1.0 - progress) / (1.0 - self.annealing_start)
        else:   # learning rate is never annealed
            annealing = 1.0

        t = soft_start if progress < self.soft_start else 1.0
        t = annealing if progress > self.annealing_start else t

        lr = exp(log(self.min_lr) + t * (log(self.base_lr) - log(self.min_lr)))

        return lr


class OneIndexedCSVLogger(keras.callbacks.CSVLogger):
    """CSV Logger with epoch number started from 1."""

    def on_epoch_end(self, epoch, logs=None):
        """On epoch end."""
        super(OneIndexedCSVLogger, self).on_epoch_end(epoch+1, logs)


class SoftStartCosineAnnealingScheduler(keras.callbacks.Callback):
    """Soft Start Cosine annealing scheduler.

        learning rate in the following 2 phases:
        Phase 1: 0.0 <= progress < soft_start:
                 Starting from min_lr linearly increase the learning rate to base_lr
        Phase 2: soft_start <= progress <= 1.0:
                 Starting from base_lr cosine decay the learning rate to min_lr

    Args:
        base_lr: Maximum learning rate
        min_lr_ratio: The ratio between minimum learning rate (min_lr) and base_lr
        soft_start: The progress at which learning rate achieves base_lr when starting from min_lr
        max_iterations: Total number of iterations in the experiment

        (https://arxiv.org/pdf/1608.03983.pdf)
    """

    def __init__(self, base_lr, min_lr_ratio, soft_start, max_iterations):
        """Initalize global parameters."""
        super(SoftStartCosineAnnealingScheduler, self).__init__()

        if not 0.0 <= soft_start <= 1.0:
            raise ValueError('The soft_start varible should be >= 0.0 or <= 1.0.')

        self.max_iterations = max_iterations
        self.soft_start = soft_start
        self.base_lr = base_lr
        self.min_lr = self.base_lr * min_lr_ratio
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """on_train_begin method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max_iterations."""
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if not 0. <= progress <= 1.:
            raise ValueError('SoftStartCosineAnnealingScheduler '
                             'does not support a progress value < 0.0 or > 1.0 '
                             'received (%f)' % progress)

        if not self.base_lr:
            return self.base_lr

        if self.soft_start > 0.0:
            soft_start = progress / self.soft_start
        else:  # learning rate starts from base_lr
            soft_start = 1.0
        if soft_start < 1:
            lr = (self.base_lr - self.min_lr) * soft_start + self.min_lr
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                (1 + math.cos(math.pi * (progress - self.soft_start))) / 2
        return lr


def build_class_weights(spec):
    """Build the class weights list."""
    mapping_dict = spec.dataset_config.target_class_mapping
    classes = sorted({str(x).lower() for x in mapping_dict.values()})
    class_weights_dict = spec.class_weighting_config.class_weighting
    class_weights_list = []
    for cls_name in classes:
        if cls_name in class_weights_dict:
            class_weights_list.append(class_weights_dict[cls_name])
        else:
            class_weights_list.append(1.0)

    return class_weights_list


class TensorBoard(keras.callbacks.Callback):
    """Callback to log some things to TensorBoard. Quite minimal, and just here as an example."""

    def __init__(self, log_dir='./logs', write_graph=True, weight_hist=False):
        """__init__ method.

        Args:
          log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
          write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
          weight_hist: whether plot histogram of weights.
        """
        super(TensorBoard, self).__init__()
        self.log_dir = log_dir
        self.write_graph = write_graph
        self._merged = None
        self._step = 0
        self._weight_hist = weight_hist
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        """on_epoch_begin method."""
        # Run user defined summaries
        if self._merged is not None:
            summary_str = self.sess.run(self._merged)
            self.writer.add_summary(summary_str, epoch)
            self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        """on_epoch_end method."""
        for name, value in logs.items():
            if ("AP" in name) or ("loss" in name) or ("acc" in name):
                if isinstance(value, np.ndarray):
                    if not np.isnan(value.item()):
                        summary = summary_from_value(name, value.item())
                        self.writer.add_summary(summary, epoch)
                        self.writer.flush()
                else:
                    if not np.isnan(value):
                        summary = summary_from_value(name, value)
                        self.writer.add_summary(summary, epoch)
                        self.writer.flush()

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = summary_from_value(name, value.item())
            self.writer.add_summary(summary, self._step)

        summary = summary_from_value('lr', K.get_value(self.model.optimizer.lr))
        self.writer.add_summary(summary, self._step)

        self._step += 1
        self.writer.flush()

    def set_model(self, model):
        """set_model method."""
        self.model = model
        self.sess = K.get_session()
        if self._weight_hist:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
        self._merged = tf.summary.merge_all()
        if self.write_graph:
            self.writer.add_graph(self.sess.graph)

    def on_train_end(self, *args, **kwargs):
        """on_train_end method."""
        self.writer.close()
