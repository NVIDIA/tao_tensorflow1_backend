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

"""Utilities for training and inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import tempfile
import warnings
from zipfile import BadZipFile, ZipFile
from addict import Dict
import keras
from keras import backend as K
from keras.models import model_from_json
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.core.templates.shufflenet import ChannelShuffle, GroupLayer
from nvidia_tao_tf1.cv.common.utils import restore_eff
from nvidia_tao_tf1.cv.unet.distribution import distribution
from nvidia_tao_tf1.cv.unet.proto.regularizer_config_pb2 import RegularizerConfig
from nvidia_tao_tf1.encoding import encoding

CUSTOM_OBJS = {}

logger = logging.getLogger(__name__)


class TargetClass(object):
    """Target class parameters."""

    def __init__(self, name, label_id, train_id=None):
        """Constructor.

        Args:
            name (str): Name of the target class.
            label_id (str):original label id of every pixel of the mask
            train_id (str): The mapped train id of every pixel in the mask
        Raises:
            ValueError: On invalid input args.
        """
        self.name = name
        self.train_id = train_id
        self.label_id = label_id


def initialize(experiment_spec):
    """Initialization. Initializes the environment variables.

    Args:
        experiment_spec: Loaded Unet Experiment spec.
        training_precision: (TrainingPrecision or None) Proto object with
        FP16/FP32 parameters or None. None leaves K.floatx()
        in its previous setting.
    """
    random_seed = experiment_spec.random_seed
    training_precision = experiment_spec.model_config.training_precision
    setup_keras_backend(training_precision, is_training=True)

    # Set Maglev random seed. Take care to give different seed to each process.
    seed = distribution.get_distributor().distributed_seed(random_seed)
    set_random_seed(seed)
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '0'
    os.environ['CACHE'] = 'false'
    os.environ['TF_ADJUST_HUE_FUSED'] = 'data'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = 'data'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = 'data'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'


def initialize_params(experiment_spec):
    """Initialization of the params object to the estimator runtime config.

    Args:
        experiment_spec: Loaded Unet Experiment spec.
    """

    training_config = experiment_spec.training_config
    dataset_config = experiment_spec.dataset_config
    model_config = experiment_spec.model_config

    return Dict({
        'exec_mode': "train",
        'model_dir': None,
        'resize_padding': dataset_config.resize_padding if
        dataset_config.resize_padding else False,
        'resize_method': dataset_config.resize_method.upper() if
        dataset_config.resize_method else 'BILINEAR',
        'log_dir': None,
        'batch_size': training_config.batch_size if training_config.batch_size else 1,
        'learning_rate': training_config.learning_rate if
        training_config.learning_rate else 0.0001,
        'activation': model_config.activation if model_config.activation else "softmax",
        'crossvalidation_idx': training_config.crossvalidation_idx if
        training_config.crossvalidation_idx else None,
        'max_steps': None,
        'regularizer_type': training_config.regularizer.type if
        training_config.regularizer.type else None,
        'weight_decay': training_config.regularizer.weight if
        training_config.regularizer.weight else 0,
        'log_summary_steps': training_config.log_summary_steps if
        training_config.log_summary_steps else 1,
        'warmup_steps': training_config.warmup_steps,
        'augment': dataset_config.augment if dataset_config.augment else False,
        'use_amp': training_config.use_amp if training_config.use_amp else False,
        'filter_data': dataset_config.filter_data if dataset_config.filter_data else False,
        'use_trt': training_config.use_trt if training_config.use_trt else False,
        'use_xla': training_config.use_xla if training_config.use_xla else False,
        'loss': training_config.loss if training_config.loss else "cross_entropy",
        'epochs': training_config.epochs if training_config.epochs else None,
        'pretrained_weights_file': model_config.pretrained_model_file if
        model_config.pretrained_model_file else None,
        'lr_scheduler': training_config.lr_scheduler if
        training_config.HasField("lr_scheduler") else None,
        'unet_model': None,
        'key': None,
        'experiment_spec': None,
        'seed': experiment_spec.random_seed,
        'benchmark': False,
        'temp_dir': tempfile.mkdtemp(),
        'num_classes': None,
        'num_conf_mat_classes': None,
        'start_step': 0,
        'checkpoint_interval': training_config.checkpoint_interval if
        training_config.checkpoint_interval else 1,
        'model_json': None,
        'custom_objs': CUSTOM_OBJS,
        'load_graph': model_config.load_graph if model_config.load_graph else False,
        'remove_head': model_config.remove_head if model_config.remove_head else False,
        'buffer_size': training_config.buffer_size if
        training_config.buffer_size else None,
        'data_options': training_config.data_options if
        training_config.data_options else False,
        'weights_monitor': training_config.weights_monitor if
        training_config.weights_monitor else False,
        'visualize': training_config.visualizer.enabled if
        training_config.visualizer.enabled else False,
        'save_summary_steps': training_config.visualizer.save_summary_steps if
        training_config.visualizer.save_summary_steps else None,
        'infrequent_save_summary_steps': training_config.visualizer.infrequent_save_summary_steps if
        training_config.visualizer.infrequent_save_summary_steps else None,
        'enable_qat': model_config.enable_qat if model_config.enable_qat else False
    })


def save_tmp_json(tmp_model_obj):
    """Function to save the QAT model to temporary json file."""

    tmp_json_dir = tempfile.mkdtemp()
    tmp_json = os.path.join(tmp_json_dir, "tmp_model.json")
    with open(tmp_json, 'w') as json_file:
        json_file.write(tmp_model_obj)

    return tmp_json


def get_weights_dir(results_dir):
    """Return weights directory.

    Args:
        results_dir: Base results directory.
    Returns:
        A directory for saved model and weights.
    """
    save_weights_dir = os.path.join(results_dir, 'weights')
    if distribution.get_distributor().is_master() and not os.path.exists(save_weights_dir):
        os.makedirs(save_weights_dir)
    return save_weights_dir


def set_random_seed(seed):
    """Set random seeds.

    This sets the random seeds of Python, Numpy and TensorFlow.

    Args:
        seed (int): seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def get_latest_tlt_model(results_dir):
    """Utility function to return the latest tlt model in a dir."""

    trainable_ckpts = [int(item.split('.')[1].split('-')[1]) for item in os.listdir(results_dir)
                       if item.endswith(".tlt")]
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = os.path.join(results_dir, "model.epoch-{}.tlt".format(latest_step))
    return latest_checkpoint


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
    else:
        model_file = None

    return model_file


def extract_pruned_model(model_file, key=None):
    """Get pruned model file name and check it exists.

    If the supplied model file is not absolute it will be prepended with the
    data root. The data root is set according to current path.

    Args:
        model_file (string): Name of the pruned model file (.tlt).
    Returns:
        The model files, jsons and meta files that exist in that tlt.
    Raises:
        AssertionError if the model file does not exist.
    """

    encrypted = False
    if os.path.isdir(model_file):
        temp_dir = model_file 
    else:
        encrypted = True
        temp_dir = tempfile.mkdtemp()
        logger.info("Loading weights from {}".format(model_file))
        os_handle, temp_zip_path = tempfile.mkstemp()
        os.close(os_handle)

        # Decrypt the checkpoint file.
        with open(model_file, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zipf:
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
    model_names = [os.path.join(temp_dir, f) for f
                   in os.listdir(temp_dir) if f.endswith(".h5")]
    model_jsons = [os.path.join(temp_dir, f) for f
                   in os.listdir(temp_dir) if f.endswith(".json")]
    meta_files = [f for f in os.listdir(temp_dir) if f.endswith(".meta")]
    meta_files_final = []
    if len(meta_files) > 0:
        if "-" in meta_files[0]:
            step = int(meta_files[0].split('model.ckpt-')[-1].split('.')[0])
            # Removing the temporary zip path.
            if encrypted:
                os.remove(temp_zip_path)
            meta_files_final.append(os.path.join(temp_dir, "model.ckpt-{}".format(step)))
        else:
            if encrypted:
                os.remove(temp_zip_path)
            meta_files_final = meta_files_final.append(os.path.join(temp_dir, "model.ckpt"))

    return model_names, model_jsons, meta_files_final


def get_pretrained_ckpt(model_file, key=None, custom_objs=None):
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
    _, ext = os.path.splitext(model_file)
    pruned_graph = False
    if ext == ".tlt" or os.path.isdir(model_file):
        model_names, model_jsons, meta_files = extract_pruned_model(model_file, key=key)
        pruned_graph = True
        # This is loading from a tlt which is ckpt
        if len(meta_files) > 0:
            model_json = model_jsons[0] if len(model_jsons) > 0 else None
            return meta_files[0], model_json, pruned_graph
        # This is loading from pruned hdf5
        model_name = model_names[0]
        model_json = model_jsons[0]
        tmp_ckpt_dir = tempfile.mkdtemp()
        tmp_ckpt_path = os.path.join(tmp_ckpt_dir, "model.ckpt")
        with open(model_json, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objs)
        # load weights into new model
        loaded_model.load_weights(model_name)
        km_weights = tf.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                       scope=None)
        tf.compat.v1.train.get_or_create_global_step()
        global_step = tf.compat.v1.train.get_global_step()
        km_weights.append(global_step)
        saver = tf.train.Saver(km_weights)
        keras_session = keras.backend.get_session()
        save_path = saver.save(keras_session, tmp_ckpt_path)
        keras.backend.clear_session()
        return save_path, model_json, pruned_graph
    if ext in (".h5", ".hdf5"):
        tmp_ckpt_dir = tempfile.mkdtemp()
        tmp_ckpt_path = os.path.join(tmp_ckpt_dir, "model.ckpt")
        km_weights = tf.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                       scope=None)
        tf.compat.v1.train.get_or_create_global_step()
        global_step = tf.compat.v1.train.get_global_step()
        km_weights.append(global_step)
        saver = tf.train.Saver(km_weights)
        keras_session = keras.backend.get_session()
        save_path = saver.save(keras_session, tmp_ckpt_path)
        keras.backend.clear_session()
        return save_path, None, pruned_graph

    raise NotImplementedError("{0} file is not supported!".format(ext))


def setup_keras_backend(training_precision, is_training):
    """Setup Keras-specific backend settings for training or inference.

    Args:
        training_precision: (TrainingPrecision or None) Proto object with
        FP16/FP32 parameters or None. None leaves K.floatx() in its previous
        setting.
        is_training: (bool) If enabled, Keras is set in training mode.
    """
    # Learning phase of '1' indicates training mode -- important for operations
    # that behave differently at training/test times (e.g. batch normalization)
    if is_training:
        K.set_learning_phase(1)
    else:
        K.set_learning_phase(0)
    # Set training precision, if given. Otherwise leave K.floatx() in its
    # previous setting. K.floatx() determines how Keras creates weights and
    # casts them (Keras default: 'float32').
    if training_precision is not None:
        if training_precision.backend_floatx == training_precision.FLOAT32:
            K.set_floatx('float32')
        elif training_precision.backend_floatx == training_precision.FLOAT16:
            K.set_floatx('float16')
        else:
            raise RuntimeError('Invalid training precision selected')


def get_results_dir(results_dir):
    """Return weights directory.

    Args:
        results_dir: Base results directory.
    Returns:
        Creates the result dir if not present and returns the path.
    """
    if distribution.get_distributor().is_master() and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def build_target_class_list(data_class_config):
    """Build a list of TargetClasses based on proto.

    Arguments:
        cost_function_config: CostFunctionConfig.
    Returns:
        A list of TargetClass instances.
    """

    target_classes = []
    orig_class_label_id_map = {}
    for target_class in data_class_config.target_classes:
        orig_class_label_id_map[target_class.name] = target_class.label_id

    class_label_id_calibrated_map = orig_class_label_id_map.copy()
    for target_class in data_class_config.target_classes:
        label_name = target_class.name
        train_name = target_class.mapping_class
        class_label_id_calibrated_map[label_name] = orig_class_label_id_map[train_name]

    train_ids = sorted(list(set(class_label_id_calibrated_map.values())))
    train_id_calibrated_map = {}
    for idx, tr_id in enumerate(train_ids):
        train_id_calibrated_map[tr_id] = idx

    class_train_id_calibrated_map = {}
    for label_name, train_id in class_label_id_calibrated_map.items():
        class_train_id_calibrated_map[label_name] = train_id_calibrated_map[train_id]

    for target_class in data_class_config.target_classes:
        target_classes.append(
            TargetClass(target_class.name, label_id=target_class.label_id,
                        train_id=class_train_id_calibrated_map[target_class.name]))

    for target_class in target_classes:
        logger.info("Label Id {}: Train Id {}".format(target_class.label_id, target_class.train_id))

    return target_classes


def get_train_class_mapping(target_classes):
    """Utility function that returns the mapping of the train id to orig class."""

    train_id_name_mapping = {}
    for target_class in target_classes:
        if target_class.train_id not in train_id_name_mapping.keys():
            train_id_name_mapping[target_class.train_id] = [target_class.name]
        else:
            train_id_name_mapping[target_class.train_id].append(target_class.name)
    return train_id_name_mapping


def compute_steps_per_epoch(num_samples, batch_size_per_gpu):
    """Compute steps per epoch based on data set size, minibatch size.

        and number of GPUs.

    Args:
        num_samples (int): Number of samples in a data set.
        batch_size_per_gpu (int): Minibatch size for a single GPU.
        logger: logger instance.
    Returns:
        Number of steps needed to iterate through the data set once.
    """
    logger.info("The total number of training samples {} and the batch size per \
                GPU {}".format(num_samples, batch_size_per_gpu))
    steps_per_epoch, remainder = divmod(num_samples, batch_size_per_gpu)
    if remainder != 0:
        logger.info("Cannot iterate over exactly {} samples with a batch size of {}; "
                    "each epoch will therefore take one extra step.".format(
                        num_samples, batch_size_per_gpu))
        steps_per_epoch = steps_per_epoch + 1

    number_of_processors = distribution.get_distributor().size()
    steps_per_epoch, remainder = divmod(steps_per_epoch, number_of_processors)
    if remainder != 0:
        logger.info("Cannot iterate over exactly {} steps per epoch with {} processors; "
                    "each processor will therefore take one extra step per epoch.".format(
                        steps_per_epoch, batch_size_per_gpu))
        steps_per_epoch = steps_per_epoch + 1
    logger.info("Steps per epoch taken: {}".format(steps_per_epoch))
    return steps_per_epoch


def compute_summary_logging_frequency(steps_per_epoch_per_gpu, num_logging_points=10):
    """Compute summary logging point frequency.

    Args:
        steps_per_epoch_per_gpu (int): Number of steps per epoch for single GPU.
        num_logging_points (int): Number of logging points per epoch.
    Returns:
        Summary logging frequency (int).
    """
    if num_logging_points > steps_per_epoch_per_gpu:
        return 1  # Log every step in epoch.

    return steps_per_epoch_per_gpu // num_logging_points


def build_regularizer(regularizer_config):
    """Build kernel and bias regularizers.

    Arguments:
        regularizer_config (regularizer_config_pb2.RegularizerConfig): Config for
          regularization.
    Returns:
        kernel_regularizer, bias_regularizer: Keras regularizers created.
    """
    # Check the config and create objects.
    if regularizer_config.weight < 0.0:
        raise ValueError("TrainingConfig.regularization_weight must be >= 0")

    if regularizer_config.type == RegularizerConfig.NO_REG:
        kernel_regularizer = None
        bias_regularizer = None
    elif regularizer_config.type == RegularizerConfig.L1:
        kernel_regularizer = keras.regularizers.l1(regularizer_config.weight)
        bias_regularizer = keras.regularizers.l1(regularizer_config.weight)
    elif regularizer_config.type == RegularizerConfig.L2:
        kernel_regularizer = keras.regularizers.l2(regularizer_config.weight)
        bias_regularizer = keras.regularizers.l2(regularizer_config.weight)
    else:
        raise NotImplementedError("The selected regularizer is not supported.")

    return kernel_regularizer, bias_regularizer


def get_num_unique_train_ids(target_classes):
    """Return the final number classes used for training.

    Arguments:
        target_classes: The target classes object that contain the train_id and
        label_id.
    Returns:
        Number of classes to be segmented.
    """

    train_ids = [target.train_id for target in target_classes]
    train_ids = np.array(train_ids)
    train_ids_unique = np.unique(train_ids)
    return len(train_ids_unique)


def update_train_params(params, num_training_examples):
    """Update the estimator with number epochs parameter."""

    if not params["max_steps"]:
        assert(params["epochs"]), "Num Epochs value needs to be provided."
        steps_per_epoch = compute_steps_per_epoch(num_training_examples,
                                                  params["batch_size"])
        params["steps_per_epoch"] = steps_per_epoch
        params["max_steps"] = steps_per_epoch * params["epochs"]
        if not params["save_summary_steps"]:
            params["save_summary_steps"] = min(1, params["steps_per_epoch"])
        if not params["infrequent_save_summary_steps"]:
            params["infrequent_save_summary_steps"] = steps_per_epoch
        assert(params["save_summary_steps"] <= params["steps_per_epoch"]), \
            "Number of save_summary_steps should be less than number of steps per epoch."
        assert(params["infrequent_save_summary_steps"] <= params["max_steps"]), \
            "Number of infrequent_save_summary_steps should be less than total number of steps."

    return params


def get_custom_objs(model_arch=None):
    """Function to return the custom layers as objects."""

    CUSTOM_OBJS = {}
    if model_arch == "shufflenet":
        CUSTOM_OBJS = {'GroupLayer': GroupLayer, 'ChannelShuffle': ChannelShuffle}

    return CUSTOM_OBJS


def update_model_params(params, unet_model, input_model_file_name=None,
                        experiment_spec=None, key=None, results_dir=None,
                        target_classes=None, phase=None, model_json=None,
                        custom_objs=None):
    """Update the estimator with additional parameters.

    Args:
        params: Additional parameters for the estimator config.
        unet_model: Unet model object.
        input_model_file_name: Path to pretrained weights.
        experiment_spec: The experiment proto.
        key: encryption key for trained models.
        results_dir: Path to the result trained weights.
        target_classes: Target classes object that contains the train_id/
        label_id
    """
    params["unet_model"] = unet_model
    params["key"] = key
    params["pretrained_weights_file"] = input_model_file_name
    params["experiment_spec"] = experiment_spec
    params['model_dir'] = results_dir
    params['seed'] = experiment_spec.random_seed
    params['num_classes'] = get_num_unique_train_ids(target_classes)
    params['num_conf_mat_classes'] = get_num_unique_train_ids(target_classes)
    # Sanity check for the activation being sigmoid
    if params['activation'] == 'sigmoid' and params['num_classes'] > 2:
        warnings.warn("Sigmoid activation can only be used for binary segmentation. \
                       Defaulting to softmax activation.")
        params['activation'] = 'softmax'
    if params['activation'] == 'sigmoid' and params['num_classes'] == 2:
        params['num_classes'] = 1
    params['phase'] = phase
    params['model_json'] = model_json
    model_arch = experiment_spec.model_config.arch
    params['custom_objs'] = get_custom_objs(model_arch=model_arch)

    return params


def get_init_ops():
    """Return all ops required for initialization."""

    return tf.group(tf.local_variables_initializer(),
                    tf.tables_initializer(),
                    *tf.get_collection('iterator_init'))
