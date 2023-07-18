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

"""
Perform continuous training for Unet Object Segmentation with Images and Masks.

This code does nothing else than training. There's no validation or
inference in this code. Use separate scripts for those purposes.

Short code breakdown:
(1) Creates the Runtime_config and creates the estimator
(2) Hook up the data pipe and estimator to unet model with backbones such as
Resnet, Vanilla Unet (https://arxiv.org/abs/1505.04597)
(3) Set up losses, metrics, hooks.
(4) Perform training steps.
"""

import argparse
import json
import logging
import os
import sys
import time
import shutil
from google.protobuf.json_format import MessageToDict
import tensorflow as tf
import wandb

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.unet.distribution import distribution
from nvidia_tao_tf1.cv.unet.dllogger.logger import JSONStreamBackend, Logger, StdOutBackend, \
    Verbosity
from nvidia_tao_tf1.cv.unet.hooks.checkpoint_saver_hook import IVACheckpointSaverHook
from nvidia_tao_tf1.cv.unet.hooks.latest_checkpoint import LatestCheckpoint
from nvidia_tao_tf1.cv.unet.hooks.pretrained_restore_hook import PretrainedWeightsLoadingHook
from nvidia_tao_tf1.cv.unet.hooks.profiling_hook import ProfilingHook
from nvidia_tao_tf1.cv.unet.hooks.training_hook import TrainingHook
from nvidia_tao_tf1.cv.unet.model.build_unet_model import build_model
from nvidia_tao_tf1.cv.unet.model.build_unet_model import get_base_model_config
from nvidia_tao_tf1.cv.unet.model.build_unet_model import select_model_proto
from nvidia_tao_tf1.cv.unet.model.utilities import build_target_class_list
from nvidia_tao_tf1.cv.unet.model.utilities import (
    get_custom_objs,
    get_latest_tlt_model,
    get_pretrained_ckpt,
    get_pretrained_model_path,
    get_results_dir,
    get_train_class_mapping,
    get_weights_dir
)
from nvidia_tao_tf1.cv.unet.model.utilities import initialize, initialize_params, save_tmp_json
from nvidia_tao_tf1.cv.unet.model.utilities import update_model_params, update_train_params
from nvidia_tao_tf1.cv.unet.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.unet.utils.data_loader import Dataset
from nvidia_tao_tf1.cv.unet.utils.model_fn import unet_fn


logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.WARN)


def run_training_loop(estimator, dataset, params, unet_model,
                      profile_logger, key, pre_train_hook, warm_start):
    """Run the training loop using the estimator.

    Args:
        estimator (class): estimator object wrapped with run config parameters.
        dataset (dataset object): Dataset object fro the dataloader utility.
        params (dict): Parameters to feed to Estimator.
        unet_model (keras model instance): Keras Unet Model.
        profile_logger (logger instance): Logging the training updates.
        key (str): The key to encrypt the model.
        pre_train_hook (class): The hook used to load the pre-trained weights.
    """

    logger.debug("Running training loop.")
    status_logging.get_status_logger().write(data=None, message="Running training loop.")
    logger.info("Running for {} Epochs".format(params.epochs))
    hooks = []
    steps_to_train = params.max_steps - params.start_step

    if steps_to_train == 0:
        # There are no more steps to be trained
        raise ValueError("Check the number of epochs mentioned in spec file should"
                         " be above 0 or if you are resuming training"
                         " the trainig has already completed for {} epochs".format(params.epochs))

    if distribution.get_distributor().is_master():
        if pre_train_hook:
            hooks.append(pre_train_hook)

    hooks.append(distribution.get_distributor().broadcast_global_variables_hook())
    hooks.append(TrainingHook(logger,
                 steps_per_epoch=params.steps_per_epoch,
                 max_epochs=params.epochs,
                 params=params,
                 log_every=params.log_summary_steps,
                 save_path=params.model_dir))
    checkpoint_n_steps = params.steps_per_epoch * params.checkpoint_interval
    if distribution.get_distributor().is_master():
        hooks.append(ProfilingHook(profile_logger,
                                   batch_size=params.batch_size,
                                   log_every=params.log_summary_steps,
                                   warmup_steps=params.warmup_steps,
                                   mode='train'))
        hooks.append(IVACheckpointSaverHook(checkpoint_dir=params.model_dir,
                                            save_secs=None,
                                            save_steps=checkpoint_n_steps,
                                            model_json=params.model_json,
                                            saver=None,
                                            checkpoint_basename="model.ckpt",
                                            steps_per_epoch=params.steps_per_epoch,
                                            scaffold=None,
                                            listeners=None,
                                            load_graph=params.load_graph
                                            ))

    estimator.train(
        input_fn=dataset.input_fn,
        steps=steps_to_train,
        hooks=hooks
        )


def train_unet(results_dir, experiment_spec, ptm, model_file,
               model_json=None, pruned_graph=False, key="None", custom_objs=None):
    """Run the training loop using the estimator.

    Args:
        results_dir (str): The path string where the trained model needs to be saved.
        experiment_spec (dict): Experiment spec proto.
        model_file (model_file): pre-trained model name for training starting point.

        key: Key to encrypt the model.

    """
    # Initialize the environment
    initialize(experiment_spec)
    # Initialize Params
    params = initialize_params(experiment_spec)
    if distribution.get_distributor().is_master():
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
    target_classes = build_target_class_list(
        experiment_spec.dataset_config.data_class_config)
    target_classes_train_mapping = get_train_class_mapping(target_classes)
    with open(os.path.join(results_dir, 'target_class_id_mapping.json'), 'w') as fp:
        json.dump(target_classes_train_mapping, fp)
    # Build run config
    model_config = select_model_proto(experiment_spec)
    unet_model = build_model(m_config=model_config,
                             target_class_names=target_classes,
                             seed=params["seed"])
    params = update_model_params(params=params, unet_model=unet_model,
                                 experiment_spec=experiment_spec, key=key,
                                 results_dir=results_dir,
                                 target_classes=target_classes,
                                 model_json=model_json,
                                 custom_objs=custom_objs,
                                 phase="train"
                                 )

    if params.enable_qat:
        # Remove model json condition to re-train pruned model with QAT
        # If load graph it is from pruned model
        # We add QDQ nodes before session is formed
        qat_on_pruned = params.load_graph
        img_height, img_width, img_channels = \
            experiment_spec.model_config.model_input_height, \
            experiment_spec.model_config.model_input_width, \
            experiment_spec.model_config.model_input_channels
        model_qat_obj = unet_model.construct_model(
            input_shape=(img_channels, img_height, img_width),
            pretrained_weights_file=params.pretrained_weights_file,
            enc_key=params.key, model_json=params.model_json,
            features=None, construct_qat=True, qat_on_pruned=qat_on_pruned)
        model_qat_json = save_tmp_json(model_qat_obj)
        params.model_json = model_qat_json

    backends = [StdOutBackend(Verbosity.VERBOSE)]

    backends.append(JSONStreamBackend(
        Verbosity.VERBOSE, params.model_dir+"/profile_log.txt"))
    profile_logger = Logger(backends)

    # Initialize env for AMP training
    if params.use_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

    if os.getenv('TF_ENABLE_AUTO_MIXED_PRECISION'):
        # Enable automatic loss scaling
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1"

    # Initialize the config for multi-gpu training
    distributor = distribution.get_distributor()
    config = distributor.get_config()
    config.gpu_options.force_gpu_compatible = True
    config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
    config.inter_op_parallelism_threads = \
        max(2, 40 // distributor.size() - 2)

    if params.use_xla:
        config.graph_options.optimizer_options.global_jit_level = \
            tf.compat.v1.OptimizerOptions.ON_1

    run_config = tf.estimator.RunConfig(
        save_summary_steps=None,
        tf_random_seed=None,
        session_config=config,
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        keep_checkpoint_every_n_hours=None,
        log_step_count_steps=None
        )
    res_hook = LatestCheckpoint(key, params.model_dir)
    warm_start = None
    pre_train_hook = None
    if res_hook.ckpt:
        # Load the latest checkpoint in dir if the training is resumed
        skip_checkpoint_variables = None
        logger.debug("Resuming from checkpoint {}.".format(res_hook.ckpt))
        warm_start = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=res_hook.ckpt,
                                                    vars_to_warm_start=[".*"])
        resuming_checkpoint = int(res_hook.ckpt.split("-")[-1])
        status_logging.get_status_logger().write(data=None, message="Resuming from checkpoint.")
        if res_hook.model_json:
            # We update json if pruned model resuming
            params.model_json = res_hook.model_json
            pruned_graph = True
    else:
        # If the user has provided a pre-trained weights path
        if ptm:
            pre_trained_weights = ptm[1]
            # We start training as a new experiment from step 0. Hence donot
            # restore the global step
            # For hdf5 checkpoint, we need to manualy develop the assignment map
            skip_checkpoint_variables = "global_step"
            # Hook initialize the session with the checkpoint to be initialized
            pre_train_hook = PretrainedWeightsLoadingHook(
                prefix="",
                checkpoint_path=pre_trained_weights,
                skip_variables_regex=skip_checkpoint_variables,
                remove_head=params.remove_head
            )
    # If the model_son is present, it is a pruned model. Check load graph is set.
    if params.model_json and pruned_graph:
        assert params.load_graph, "Load graph needs to be set for re-training of pruned model."
    if not pruned_graph:
        assert not params.load_graph, "Load graph should not be set if not \
            re-training pruned model."

    # Check if model json is available if load graph is set
    if params.load_graph:
        assert params.model_json, \
            "Load graph should be set only when you fine-tuning from a pruned model/ \
             Resuming training in phase 1 from a pruned checkpoint."

        logger.info("Retrieving model template from {}".format(params.model_json))

    dataset = Dataset(batch_size=params.batch_size,
                      fold=params.crossvalidation_idx,
                      augment=params.augment,
                      gpu_id=distributor.rank(),
                      num_gpus=distributor.size(),
                      params=params,
                      phase="train",
                      target_classes=target_classes,
                      buffer_size=params.buffer_size,
                      data_options=params.data_options,
                      filter_data=params.filter_data
                      )
    # Update params for number of epochs
    params = update_train_params(params, num_training_examples=dataset.train_size)
    if res_hook.ckpt:
        params.start_step = resuming_checkpoint * params.steps_per_epoch

    estimator = tf.estimator.Estimator(
        model_fn=unet_fn,
        model_dir=params.model_dir,
        config=run_config,
        params=params,
        warm_start_from=warm_start)
    run_training_loop(estimator, dataset, params, unet_model,
                      profile_logger, key, pre_train_hook, warm_start)
    # Saving the last training step model to weights directory
    latest_tlt = get_latest_tlt_model(params.model_dir)
    logger.info("Saving the final step model to {}".format(model_file))
    if distribution.get_distributor().is_master():
        shutil.copytree(latest_tlt, model_file)
        

def run_experiment(config_path, results_dir, pretrained_model_file=None,
                   model_name="model", override_spec_path=None,
                   key=None, verbosity="INFO", wandb_logged_in=False):
    """
    Launch experiment that trains the model.

    NOTE: Do not change the argument names without verifying that
    cluster submission works.

    Args:
        config_path (list): List containing path to a text file containing a
            complete experiment configuration and possibly a path to a .yml file
            containing override parameter values.
        results_dir (str): Path to a folder where various training
            outputs will be written. If the folder
            does not already exist, it will be created.
        pretrained_model_file (str):Optional path to a pretrained model file. This maybe invoked
            from the CLI if needed. For now, we have disabled support to maintain
            consistency across all magnet apps.
        model_name (str): Model name to be used as a part of model file name.
        override_spec_path (str): Absolute path to yaml file which is used to
            overwrite some of the experiment spec parameters.
        key (str): Key to save and load models from tlt.
        verbosity (str): Logging verbosity among ["INFO", "DEBUG"].
        wandb_logged_in (bool): Check if wandb credentials were set.
    """

    logger.debug("Starting experiment.")

    model_path = get_weights_dir(results_dir)
    model_file = os.path.join(model_path, '%s.tlt' % model_name)

    if distribution.get_distributor().is_master():
        output_file_handler = logging.FileHandler(os.path.join(results_dir, "output.log"))
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(output_file_handler)
        logger.addHandler(stdout_handler)
        logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                            level=verbosity)
    # Load experiment spec.
    if config_path is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", config_path)

        # The spec in experiment_spec_path has to be complete.
        # Default spec is not merged into experiment_spec.
        experiment_spec = load_experiment_spec(
            config_path, merge_from_default=False)
    else:
        logger.info("Loading default ISBI single class experiment spec.")
        experiment_spec = load_experiment_spec()

    # Extract core model config, which might be wrapped inside a
    # TemporalModelConfig.
    model_config = get_base_model_config(experiment_spec)

    # Pretrained model can be provided either through CLI or spec. Expand and validate the path.
    assert not (pretrained_model_file and model_config.pretrained_model_file), \
        "Provide only one pretrained model file."
    custom_objs = None
    ptm = ()

    input_model_file_name = get_pretrained_model_path(pretrained_model_file)
    pre_trained_weights = None
    # Dump experiment spec to result directory.
    if distribution.get_distributor().is_master():
        with open(os.path.join(results_dir, 'experiment_spec.txt'), 'w') as f:
            f.write(str(experiment_spec))
    if input_model_file_name:
        _, ext = os.path.splitext(input_model_file_name)
        logging.info("Initializing the pre-trained weights from {}".format
                     (input_model_file_name))
        # Get the model_json here
        pre_trained_weights, model_json, pruned_graph = \
            get_pretrained_ckpt(input_model_file_name, key=key, custom_objs=custom_objs)
        ptm = (ext, pre_trained_weights)
    else:
        # Assert if freeze blocks is provided only if pretrained weights are present.
        if model_config.freeze_blocks:
            raise ValueError("Freeze blocks is only possible if a pretrained model"
                             "file is provided.")
        pre_trained_weights = None
        model_json = None
        pruned_graph = False
    if distribution.get_distributor().is_master():
        if experiment_spec.training_config.HasField("visualizer"):
            visualizer_config = experiment_spec.training_config.visualizer
            if visualizer_config.HasField("wandb_config"):
                wandb_config = visualizer_config.wandb_config
                logger.info("Integrating with W&B")
                wandb_name = f"{wandb_config.name}" if wandb_config.name \
                    else "unet"
                wandb_stream_config = MessageToDict(
                    experiment_spec,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True
                )
                initialize_wandb(
                    project=wandb_config.project if wandb_config.project else None,
                    entity=wandb_config.entity if wandb_config.entity else None,
                    config=wandb_stream_config,
                    notes=wandb_config.notes if wandb_config.notes else None,
                    tags=wandb_config.tags if wandb_config.tags else None,
                    sync_tensorboard=True,
                    save_code=False,
                    results_dir=results_dir,
                    wandb_logged_in=wandb_logged_in,
                    name=wandb_name
                )
            if visualizer_config.HasField("clearml_config"):
                logger.info("Integrating with clearml")
                clearml_config = visualizer_config.clearml_config
                get_clearml_task(clearml_config, "unet")

    # Update custom_objs with Internal TAO custom layers
    custom_objs = get_custom_objs(model_arch=model_config.arch)
    train_unet(results_dir, experiment_spec, ptm, model_file,
               model_json, pruned_graph, key=key, custom_objs=custom_objs)
    status_logging.get_status_logger().write(data=None, message="Unet training complete.")

    logger.debug("Experiment complete.")
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.SUCCESS,
        message="Experiment complete."
    )


def build_command_line_parser(parser=None):
    """
    Parse command-line flags passed to the training script.

    Returns:
      Namespace with all parsed arguments.
    """

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='train', description='Train a segmentation model.')

    default_experiment_path = os.path.join(os.path.expanduser('~'), 'experiments',
                                           time.strftime("drivenet_%Y%m%d_%H%M%S"))

    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        default=None,
        help='Path to spec file. Absolute path or relative to working directory. \
            If not specified, default spec from spec_loader.py is used.'
    )
    parser.add_argument(
        '-m',
        '--pretrained_model_file',
        type=str,
        default=None,
        help='Model path to the pre-trained weights.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default=default_experiment_path,
        help='Path to a folder where experiment outputs should be written.'
    )
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='model',
        help='Name of the model file. If not given, then defaults to model.tlt.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Set verbosity level for the logger.'
    )
    parser.add_argument(
        '-k',
        '--key',
        default="",
        type=str,
        required=False,
        help='The key to load pretrained weights and save intermediate snapshopts and final model.'
    )

    return parser


def parse_command_line_args(cl_args=None):
    """Parser command line arguments to the trainer.

    Args:
        cl_args(sys.argv[1:]): Arg from the command line.

    Returns:
        args: Parsed arguments using argparse.
    """
    parser = build_command_line_parser(parser=None)
    args = parser.parse_args(cl_args)
    return args


def main(args=None):
    """Run the training process."""
    args = parse_command_line_args(args)

    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'
    # Enable Horovod distributor for multi-GPU training.
    distribution.set_distributor(distribution.HorovodDistributor())
    # Check the results dir path and create
    results_dir = args.results_dir
    results_dir = get_results_dir(results_dir)
    events_dir = os.path.join(results_dir, "events")
    is_master = distribution.get_distributor().is_master()
    wandb_logged_in = False
    if is_master:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(events_dir):
            os.makedirs(events_dir)
        wandb_logged_in = check_wandb_logged_in()

    # Configure tf logger verbosity.
    tf.logging.set_verbosity(tf.logging.INFO)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=is_master,
                verbosity=logger.getEffectiveLevel(),
                append=True
            )
        )
    try:
        status_logging.get_status_logger().write(
            data=None,
            status_level=status_logging.Status.STARTED,
            message="Starting UNet Training job"
        )
        run_experiment(config_path=args.experiment_spec_file,
                       results_dir=args.results_dir,
                       model_name=args.model_name,
                       key=args.key,
                       pretrained_model_file=args.pretrained_model_file,
                       verbosity=verbosity,
                       wandb_logged_in=wandb_logged_in)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Training was interrupted.")
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
    finally:
        if distribution.get_distributor().is_master():
            if wandb_logged_in:
                wandb.finish()


if __name__ == "__main__":
    main()
