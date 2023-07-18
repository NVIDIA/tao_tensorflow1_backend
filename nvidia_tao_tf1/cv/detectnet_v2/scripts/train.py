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
Perform continuous training for gridbox object detection networks on a tfrecords dataset.

This code does nothing else than training. There's no validation or inference in this code.
Use separate scripts for those purposes.

Short code breakdown:
(1) Set up some processors (yield tfrecords batches, data decoding, ground-truth generation, ..)
(2) Hook up the data pipe and processors to a DNN, for example a Resnet18, or Vgg16 template.
(3) Set up losses, metrics, hooks.
(4) Perform training steps.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import logging
import os
import time

from google.protobuf.json_format import MessageToDict

import tensorflow as tf
import wandb
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import Bbox2DLabel
import nvidia_tao_tf1.core
from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.core.hooks.sample_counter_hook import SampleCounterHook
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.common.utils import get_model_file_size
from nvidia_tao_tf1.cv.common.utilities.serialization_listener import (
    EpochModelSerializationListener
)
from nvidia_tao_tf1.cv.detectnet_v2.common.graph import get_init_ops
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_auto_weight_hook import (
    build_cost_auto_weight_hook
)
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_function_parameters import (
    build_target_class_list
)
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_function_parameters import (
    get_target_class_names
)
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import build_dataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import select_dataset_proto
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.evaluation import Evaluator
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.evaluation_config import build_evaluation_config
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import build_model
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import get_base_model_config
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import select_model_proto
from nvidia_tao_tf1.cv.detectnet_v2.model.utilities import get_pretrained_model_path, get_tf_ckpt
from nvidia_tao_tf1.cv.detectnet_v2.objectives.build_objective_label_filter import (
    build_objective_label_filter
)
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing_config import (
    build_postprocessing_config
)
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer import BboxRasterizer
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.build_bbox_rasterizer_config import (
    build_bbox_rasterizer_config
)
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.loss_mask_rasterizer import LossMaskRasterizer
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.early_stopping_hook import build_early_stopping_hook
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.task_progress_monitor_hook import (
    TaskProgressMonitorHook
)
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.utils import get_common_training_hooks
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.validation_hook import ValidationHook
from nvidia_tao_tf1.cv.detectnet_v2.training.training_proto_utilities import (
    build_learning_rate_schedule,
    build_optimizer,
    build_regularizer,
    build_train_op_generator
)
from nvidia_tao_tf1.cv.detectnet_v2.training.utilities import compute_steps_per_epoch
from nvidia_tao_tf1.cv.detectnet_v2.training.utilities import compute_summary_logging_frequency
from nvidia_tao_tf1.cv.detectnet_v2.training.utilities import get_singular_monitored_session
from nvidia_tao_tf1.cv.detectnet_v2.training.utilities import get_weights_dir
from nvidia_tao_tf1.cv.detectnet_v2.training.utilities import initialize
from nvidia_tao_tf1.cv.detectnet_v2.utilities.timer import time_function
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer

logger = logging.getLogger(__name__)

loggable_tensors = {}


def run_training_loop(experiment_spec, results_dir, gridbox_model, hooks, steps_per_epoch,
                      output_model_file_name, maglev_experiment, model_version_labels,
                      visualizer_config, key):
    """Train the model.

    Args:
        experiment_spec (experiment_pb2.Experiment): Experiment spec.
        results_dir (str): Path to a folder where various training outputs will be written.
        gridbox_model (Gridbox): Network to train.
        hooks (list): A list of hooks.
        steps_per_epoch (int): Number of steps per epoch.
        output_model_file_name (str): Name of a model to be saved after training.
        maglev_experiment (maglev.platform.experiment.Experiment): Maglev Experiment object.
        model_version_labels (dict): Labels to attach to the created ModelVersions.
        visualizer_config (VisualizerConfigProto): Configuration element for the visualizer.
        key (str): A key to load and save models.
    """
    # Get all the objects necessary for the SingularMonitoredSession
    status_logging.get_status_logger().write(data=None, message="Running training loop.")
    num_epochs = experiment_spec.training_config.num_epochs
    num_training_steps = steps_per_epoch * num_epochs
    # Setting default checkpoint interval.
    checkpoint_interval = 10
    if experiment_spec.training_config.checkpoint_interval:
        checkpoint_interval = experiment_spec.training_config.checkpoint_interval
    logger.info("Checkpoint interval: {}".format(checkpoint_interval))
    global_step = tf.train.get_or_create_global_step()
    distributor = distribution.get_distributor()
    config = distributor.get_config()
    is_master = distributor.is_master()

    # Number of points per epoch to log scalars.
    num_logging_points = visualizer_config.scalar_logging_frequency if \
        visualizer_config.scalar_logging_frequency else 10
    if num_logging_points > steps_per_epoch:
        validation_message = f"Number of logging points {num_logging_points} "\
            f"must be <= than the number of steps per epoch {steps_per_epoch}."
        status_logging.get_status_logger().write(
            message=validation_message,
            status_level=status_logging.Status.FAILURE
        )
        raise ValueError(validation_message)

    # Compute logging frequency based on user defined number of logging points.
    summary_every_n_steps = compute_summary_logging_frequency(
        steps_per_epoch,
        num_logging_points=num_logging_points
    )

    # Infrequent logging frequency in epochs
    if Visualizer.enabled:
        infrequent_logging_frequency = visualizer_config.infrequent_logging_frequency if \
            visualizer_config.infrequent_logging_frequency else 1
        if infrequent_logging_frequency > num_epochs:
            validation_message = f"Infrequent logging frequency {infrequent_logging_frequency} "\
                f"must be lesser than the total number of epochs {num_epochs}."
            status_logging.get_status_logger().write(
                message=validation_message,
                status_level=status_logging.Status.FAILURE
            )
            raise ValueError(validation_message)
        infrequent_summary_every_n_steps = steps_per_epoch * infrequent_logging_frequency
    else:
        infrequent_summary_every_n_steps = 0

    logger.info(
        "Scalars logged at every {summary_every_n_steps} steps".format(
            summary_every_n_steps=summary_every_n_steps
        )
    )
    logger.info(
        "Images logged at every {infrequent_summary_every_n_steps} steps".format(
            infrequent_summary_every_n_steps=infrequent_summary_every_n_steps
        )
    )
    scaffold = tf.compat.v1.train.Scaffold(local_init_op=get_init_ops())

    # Get a listener that will serialize the metadata upon each checkpoint saving call.
    serialization_listener = EpochModelSerializationListener(
        checkpoint_dir=results_dir,
        model=gridbox_model,
        key=key,
        steps_per_epoch=steps_per_epoch,
        max_to_keep=None)
    listeners = [serialization_listener]
    loggable_tensors.update({
        'epoch': global_step / steps_per_epoch,
        'step': global_step,
        'loss': gridbox_model.get_total_cost()})

    training_hooks = get_common_training_hooks(
        log_tensors=loggable_tensors,
        log_every_n_secs=5,
        checkpoint_n_steps=steps_per_epoch * checkpoint_interval,
        model=None,
        last_step=num_training_steps,
        checkpoint_dir=results_dir,
        steps_per_epoch=steps_per_epoch,
        scaffold=scaffold,
        summary_every_n_steps=summary_every_n_steps,
        infrequent_summary_every_n_steps=infrequent_summary_every_n_steps,
        listeners=listeners,
        key=key)

    training_hooks.extend(hooks)
    # Add task progress monitoring hook to the master process.
    if is_master:
        training_hooks.append(TaskProgressMonitorHook(loggable_tensors,
                                                      results_dir,
                                                      num_epochs,
                                                      steps_per_epoch))
        total_batch_size = experiment_spec.training_config.batch_size_per_gpu * \
            distributor.size()
        training_hooks.append(SampleCounterHook(batch_size=total_batch_size, name="Train"))

    checkpoint_filename = get_latest_checkpoint(results_dir, key)

    with get_singular_monitored_session(keras_models=gridbox_model.get_keras_training_model(),
                                        session_config=config,
                                        hooks=training_hooks,
                                        scaffold=scaffold,
                                        checkpoint_filename=checkpoint_filename) as session:
        try:
            while not session.should_stop():
                session.run([gridbox_model.get_train_op()])
            status_logging.get_status_logger().write(
                data=None,
                message="Training loop completed."
            )
        except (KeyboardInterrupt, SystemExit) as e:
            logger.info("Training was interrupted.")
            status_logging.get_status_logger().write(
                data={"Error": "{}".format(e)},
                message="Training was interrupted"
            )
        finally:
            # Saves the last best model before the graph is finalized.
            save_model(gridbox_model, output_model_file_name, key=key)


def get_latest_checkpoint(results_dir, key):
    """Get the latest checkpoint path from a given results directory.

    Parses through the directory to look for the latest checkpoint file
    and returns the path to this file.

    Args:
        results_dir (str): Path to the results directory.

    Returns:
        ckpt_path (str): Path to the latest checkpoint.
    """
    trainable_ckpts = [int(item.split('.')[1].split('-')[1]) for item in os.listdir(results_dir)
                       if item.endswith(".ckzip")]
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = os.path.join(results_dir, "model.epoch-{}.ckzip".format(latest_step))
    return get_tf_ckpt(latest_checkpoint, key, latest_step)


def save_model(gridbox_model, output_model_file_name, key):
    """Save final Helnet model to disk and create a ModelVersion if we are in a workflow.

    Args:
        gridbox_model (GridboxModel): Final gridbox detector model.
        output_model_file_name: Name of a model to be saved.
        key (str): A key to save and load models in tlt format.
    """
    # Master process saves the model to disk. This saves the final model even if checkpointer
    # hook was not enabled.
    status_logging.get_status_logger().write(
        data=None,
        message="Saving trained model."
    )
    if distribution.get_distributor().is_master():
        gridbox_model.save_model(file_name=output_model_file_name,
                                 enc_key=key)
        s_logger = status_logging.get_status_logger()
        s_logger.kpi = {
            "size": get_model_file_size(output_model_file_name),
            "param_count": gridbox_model.num_params
        }
        s_logger.write(
            message="Model saved."
        )


def build_rasterizers(experiment_spec, input_width, input_height, output_width, output_height):
    """Build bbox and loss mask rasterizers.

    Args:
        experiment_spec (experiment_pb2.Experiment): Experiment spec.
        input_width/height (int): Model input size.
        output_width/height (int): Model output size.

    Returns:
        bbox_rasterizer (BboxRasterizer): A rasterizer for ground truths.
        loss_mask_rasterizer (LossMaskRasterizer): A rasterizer for loss masks.
    """
    # Build a BboxRasterizer with which to generate ground truth tensors.
    status_logging.get_status_logger().write(data=None, message="Building rasterizer.")
    target_class_names = get_target_class_names(experiment_spec.cost_function_config)
    target_class_mapping = dict(experiment_spec.dataset_config.target_class_mapping)
    bbox_rasterizer_config = build_bbox_rasterizer_config(experiment_spec.bbox_rasterizer_config)
    bbox_rasterizer = BboxRasterizer(input_width=input_width,
                                     input_height=input_height,
                                     output_width=output_width,
                                     output_height=output_height,
                                     target_class_names=target_class_names,
                                     bbox_rasterizer_config=bbox_rasterizer_config,
                                     target_class_mapping=target_class_mapping)

    # Build a LossMaskRasterizer with which to generate loss masks.
    loss_mask_rasterizer = LossMaskRasterizer(input_width=input_width,
                                              input_height=input_height,
                                              output_width=output_width,
                                              output_height=output_height)
    status_logging.get_status_logger().write(data=None, message="Rasterizers built.")
    return bbox_rasterizer, loss_mask_rasterizer


def rasterize_source_weight(batch_labels):
    """Method that users will call to generate source_weight tensors.

    Args:
        batch_labels (nested dict or BBox2DLabel): If nested dict, has two levels:
            [target_class_name][objective_name]. The leaf values are the corresponding filtered
            ground truth labels in tf.Tensor for a batch of frames.
            If BBox2DLabel, it incorporates labels for all frames.

    Returns:
        source_weight_tensor (Tensor): source weight tensor with shape [N,], where N is the
            batch size. It should be expanded to [N,1,1...] before it is computed in loss
            function.
    """
    source_weight_tensor = None
    # Step1_0: we try to get source_weight_tensors with shape [N,].
    if isinstance(batch_labels, list):
        source_weight_tensor_arrs = []
        for gt_label in batch_labels:
            # Have to reshape the "source_weight" tensor to [1], so that tf.concat could work.
            if "source_weight" in gt_label:
                source_weight_tensor_arrs.append(tf.reshape(gt_label["source_weight"], [1]))
            else:
                return source_weight_tensor
        source_weight_tensor = tf.concat(source_weight_tensor_arrs, axis=0)

    elif isinstance(batch_labels, Bbox2DLabel):
        # source_weight_tensor is in the shape [N,].
        source_weight_tensor = tf.squeeze(batch_labels.source_weight)
    else:
        raise TypeError("Only dict or BBox2dLabel could be handled by sw rasterize")

    # TODO(ashen): whether we need below normalization methods:
    # Reciprocal of mean value of source_weight tensor, used for normalization
    # Step1_1: source_weight_mean_norm = 1.0 / tf.reduce_mean(source_weight_base_tensor)
    # Step1_2: source_weight_tensor = source_weight_tensor * source_weight_mean_norm
    return source_weight_tensor


def merge_source_weight_to_loss_mask(source_weight_tensor, loss_masks, ground_truth_tensors):
    """Merge source weight tensors into loss masks.

    Args:
        source_weight_tensor (Tensor): source weight tensor with shape [N,]
        loss_masks (Nested dict): dict with 2 levels:
            [target_class_name][objective_name]. The leaf values are the loss_mask
            tensors. Also the dict could be empty.
        ground_truth_tensors (Nested dict): the ground truth dictionary to contain
            ground_truth tensors.

    Returns:
        loss_masks (Nested dict): Modified loss_masks dictionary to incorporate
            source weight tensors.
    """
    if source_weight_tensor is None or source_weight_tensor.shape.ndims != 1:
        return loss_masks

    for class_name in ground_truth_tensors.keys():
        if class_name not in loss_masks.keys():
            loss_masks[class_name] = dict()
        for objective_name in ground_truth_tensors[class_name].keys():
            # We expand the source_weight_tensor to be [N,1,1,...], which is like
            # ground_truth_tensors[class_name][objective_name].
            gt_tensor = ground_truth_tensors[class_name][objective_name]
            # Step1: broadcast from [N,] to [1,1...,N].
            exp_source_weight_tensor = tf.broadcast_to(source_weight_tensor,
                                                       shape=[1] * (gt_tensor.shape.ndims - 1)
                                                       + [source_weight_tensor.shape[0]])
            # Step2: transpose to get the tensor with shape [N,1,1,..].
            exp_source_weight_tensor = tf.transpose(exp_source_weight_tensor)

            if objective_name in loss_masks[class_name]:
                # If loss_mask exists, we merge it with source_weight_tensor.
                loss_mask_tensor = loss_masks[class_name][objective_name]
                # Assign merged loss mask tensors.
                loss_masks[class_name][objective_name] = tf.multiply(loss_mask_tensor,
                                                                     exp_source_weight_tensor)
            else:
                # If loss_mask does not exist, we directly assign it to be source_weight_tensor.
                loss_masks[class_name][objective_name] = exp_source_weight_tensor
    return loss_masks


def rasterize_tensors(gridbox_model, loss_mask_label_filter, bbox_rasterizer, loss_mask_rasterizer,
                      ground_truth_labels):
    """Rasterize ground truth and loss mask tensors.

    Args:
        gridbox_model (HelnetGridbox): A HelnetGridbox instance.
        loss_mask_label_filter (ObjectiveLabelFilter): A label filter for loss masks.
        bbox_rasterizer (BboxRasterizer): A rasterizer for ground truths.
        loss_mask_rasterizer (LossMaskRasterizer): A rasterizer for loss masks.
        ground_truth_labels (list): Each element is a dict of target features (each a tf.Tensor).

    Returns:
        ground_truth_tensors (dict): [target_class_name][objective_name] rasterizer ground truth
            tensor.
        loss_masks (tf.Tensor): rasterized loss mask corresponding to the input labels.
    """
    status_logging.get_status_logger().write(data=None, message="Rasterizing tensors.")
    # Get ground truth tensors.
    ground_truth_tensors = \
        gridbox_model.generate_ground_truth_tensors(bbox_rasterizer=bbox_rasterizer,
                                                    batch_labels=ground_truth_labels)
    # Get the loss mask labels.
    loss_mask_labels = loss_mask_label_filter.apply_filters(ground_truth_labels)
    ground_truth_mask = ground_truth_tensors if loss_mask_label_filter.preserve_ground_truth else \
        None
    # Get the loss masks.
    loss_masks = loss_mask_rasterizer(
        loss_mask_batch_labels=loss_mask_labels,
        ground_truth_tensors=ground_truth_mask,
        mask_multiplier=loss_mask_label_filter.mask_multiplier)

    source_weight_tensor = rasterize_source_weight(ground_truth_labels)

    # Merge source_weight_tensors with loss_masks
    loss_masks = merge_source_weight_to_loss_mask(source_weight_tensor,
                                                  loss_masks,
                                                  ground_truth_tensors)

    status_logging.get_status_logger().write(data=None, message="Tensors rasterized.")
    return ground_truth_tensors, loss_masks


def build_gridbox_model(experiment_spec, input_shape, model_file_name=None, key=None):
    """Instantiate a HelnetGridbox or a child class, e.g. a HelnetGRUGridbox.

    Args:
        experiment_spec (experiment_pb2.Experiment): Experiment spec.
        input_shape (tuple): Model input shape as a CHW tuple. Not used if
            model_file_name is not None.
        model_file_name: Model file to load, or None is a new model should be created.
        key (str): A key to load and save tlt models.
    Returns:
        A HelnetGridbox or a child class instance, e.g. a HelnetGRUGridbox.
    """
    status_logging.get_status_logger().write(data=None, message="Building DetectNet V2 model")
    target_class_names = get_target_class_names(experiment_spec.cost_function_config)
    # Select the model config, which might have ModelConfig / TemporalModelConfig type.
    model_config = select_model_proto(experiment_spec)
    enable_qat = experiment_spec.training_config.enable_qat
    gridbox_model = build_model(m_config=model_config,
                                target_class_names=target_class_names,
                                enable_qat=enable_qat)
    # Set up regularization.
    kernel_regularizer, bias_regularizer = build_regularizer(
        experiment_spec.training_config.regularizer)

    if not model_config.load_graph:
        # Construct model if the pretrained model is not pruned.
        gridbox_model.construct_model(input_shape=input_shape,
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      pretrained_weights_file=model_file_name,
                                      enc_key=key)
    else:
        # Load model if with structure for pruned models.
        assert model_config.pretrained_model_file is not None, "Please provide pretrained"\
                                                               "model with the is_pruned flag."
        gridbox_model.load_model_weights(model_file_name, enc_key=key)
        gridbox_model.update_regularizers(kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer)
        # If the model is loaded from a file, we need to make sure that
        # model contains all the objectives as defined in the spec file.
        gridbox_model.add_missing_outputs(kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer)

    gridbox_model.print_model_summary()
    status_logging.get_status_logger().write(data=None, message="DetectNet V2 model built.")
    return gridbox_model


def build_training_graph(experiment_spec,
                         gridbox_model,
                         loss_mask_label_filter,
                         bbox_rasterizer,
                         loss_mask_rasterizer,
                         dataloader,
                         learning_rate,
                         cost_combiner_func):
    """Build training graph.

    Args:
        experiment_spec (experiment_pb2.Experiment): Experiment spec.
        gridbox_model (HelnetGridbox): A HelnetGridbox instance.
        loss_mask_label_filter (ObjectiveLabelFilter): A label filter for loss masks.
        bbox_rasterizer (BboxRasterizer): A rasterizer for ground truths.
        loss_mask_rasterizer (LossMaskRasterizer): A rasterizer for loss masks.
        dataloader (Dataloader): A dataloader instance (eg. DefaultDataloader).
        learning_rate (tf.Variable): Learning rate variable.
        cost_combiner_func: A function that takes in a dictionary of objective costs,
            and total cost by computing a weighted sum of the objective costs.
    """
    status_logging.get_status_logger().write(
        data=None,
        message="Building training graph."
    )
    # Get training image and label tensors from dataset.
    batch_size = experiment_spec.training_config.batch_size_per_gpu
    training_images, training_ground_truth_labels, num_training_samples = \
        dataloader.get_dataset_tensors(batch_size, training=True, enable_augmentation=True)

    logger.info("Found %d samples in training set", num_training_samples)

    # # Add input images to Tensorboard. Specify value range to avoid Tensorflow automatic scaling.
    Visualizer.image('images', training_images, value_range=[0.0, 1.0],
                     collections=[nvidia_tao_tf1.core.hooks.utils.INFREQUENT_SUMMARY_KEY])

    # Rasterize ground truth and loss mask tensors.
    training_ground_truth_tensors, training_loss_masks =\
        rasterize_tensors(gridbox_model, loss_mask_label_filter, bbox_rasterizer,
                          loss_mask_rasterizer,
                          training_ground_truth_labels)

    # Set up optimizer.
    optimizer = build_optimizer(experiment_spec.training_config.optimizer, learning_rate)

    # Build training graph.
    train_op_generator = build_train_op_generator(experiment_spec.training_config.cost_scaling)
    target_classes = build_target_class_list(experiment_spec.cost_function_config)
    gridbox_model.build_training_graph(training_images, training_ground_truth_tensors,
                                       optimizer, target_classes, cost_combiner_func,
                                       train_op_generator, training_loss_masks)
    gridbox_model.visualize_predictions()
    status_logging.get_status_logger().write(data=None, message="Training graph built.")


def build_validation_graph(experiment_spec,
                           gridbox_model,
                           loss_mask_label_filter,
                           bbox_rasterizer,
                           loss_mask_rasterizer,
                           dataloader,
                           num_validation_steps,
                           cost_combiner_func):
    """Build validation graph.

    Args:
        experiment_spec (experiment_pb2.Experiment): Experiment spec.
        gridbox_model (HelnetGridbox): A HelnetGridbox instance.
        loss_mask_label_filter (ObjectiveLabelFilter): A label filter for loss masks.
        bbox_rasterizer (BboxRasterizer): A rasterizer for ground truths.
        loss_mask_rasterizer (LossMaskRasterizer): A rasterizer for loss masks.
        dataloader (Dataloader): A dataloader instance (eg. DefaultDataloader).
        num_validation_steps (int): Number of validation steps.
        cost_combiner_func: A function that takes in a dictionary of objective costs,
            and total cost by computing a weighted sum of the objective costs.

    Returns:
        Evaluator instance.
    """
    status_logging.get_status_logger().write(data=None, message="Building validation graph.")
    # Get validation image and label tensors from dataset.
    batch_size = experiment_spec.training_config.batch_size_per_gpu
    validation_images, validation_ground_truth_labels, num_validation_samples = \
        dataloader.get_dataset_tensors(batch_size, training=False, enable_augmentation=False)

    logger.info("Found %d samples in validation set", num_validation_samples)

    assert num_validation_samples > 0,\
        "Validation period is not 0, but no validation data found. "\
        "Either turn off validation by setting `validation_period = 0` or specify correct "\
        "path/fold for validation data."

    # Rasterize ground truth and loss mask tensors.
    validation_ground_truth_tensors, validation_loss_masks =\
        rasterize_tensors(gridbox_model, loss_mask_label_filter, bbox_rasterizer,
                          loss_mask_rasterizer, validation_ground_truth_labels)

    # Build validation graph.
    target_classes = build_target_class_list(experiment_spec.cost_function_config)
    gridbox_model.build_validation_graph(validation_images, validation_ground_truth_tensors,
                                         target_classes,
                                         cost_combiner_func, validation_loss_masks)

    postprocessing_config = build_postprocessing_config(experiment_spec.postprocessing_config)
    evaluation_config = build_evaluation_config(experiment_spec.evaluation_config,
                                                gridbox_model.target_class_names)
    confidence_models = None
    evaluator = Evaluator(postprocessing_config=postprocessing_config,
                          evaluation_config=evaluation_config,
                          gridbox_model=gridbox_model,
                          images=validation_images,
                          ground_truth_labels=validation_ground_truth_labels,
                          steps=num_validation_steps,
                          confidence_models=confidence_models)
    status_logging.get_status_logger().write(data=None, message="Validation graph built.")
    return evaluator


def train_gridbox(results_dir, experiment_spec, output_model_file_name,
                  input_model_file_name=None, maglev_experiment=None, model_version_labels=None,
                  key=None):
    """Construct, train, and save a gridbox_model gridbox model.

    Args:
        results_dir (str): Path to a folder where various training outputs will be written.
            If the folder does not already exist, it will be created.
        experiment_spec (experiment_pb2.Experiment): Experiment spec.
        output_model_file_name (str): Name of a model to be saved after training.
        input_model_file_name: Name of a model file to load, or None if a model should be
            created from scratch.
        maglev_experiment (maglev.platform.experiment.Experiment): Maglev Experiment object.
        model_version_labels (dict): Labels to attach to the created ModelVersions.
    """
    # Extract core model config, which might be wrapped inside a TemporalModelConfig.
    status_logging.get_status_logger().write(data=None, message="Training gridbox model.")
    model_config = get_base_model_config(experiment_spec)
    # Initialization of distributed seed, training precision and learning phase.
    initialize(experiment_spec.random_seed, model_config.training_precision)

    is_master = distribution.get_distributor().is_master()

    # TODO: vpraveen <test without visualizer>
    # Set up visualization.
    visualizer_config = experiment_spec.training_config.visualizer
    # Disable visualization for other than the master process.
    if not is_master:
        visualizer_config.enabled = False
    Visualizer.build_from_config(visualizer_config)

    dataset_proto = select_dataset_proto(experiment_spec)

    # Build a dataloader.
    dataloader = build_dataloader(dataset_proto=dataset_proto,
                                  augmentation_proto=experiment_spec.augmentation_config)

    # Compute steps per training epoch, and number of training and validation steps.
    num_training_samples = dataloader.get_num_samples(training=True)
    num_validation_samples = dataloader.get_num_samples(training=False)
    batch_size = experiment_spec.training_config.batch_size_per_gpu
    steps_per_epoch = compute_steps_per_epoch(num_training_samples, batch_size, logger)
    num_training_steps = steps_per_epoch * experiment_spec.training_config.num_epochs
    num_validation_steps = num_validation_samples // batch_size

    # Set up cost auto weighter hook.
    cost_auto_weight_hook = build_cost_auto_weight_hook(experiment_spec.cost_function_config,
                                                        steps_per_epoch)
    hooks = [cost_auto_weight_hook]

    # Construct a model.
    gridbox_model = build_gridbox_model(experiment_spec=experiment_spec,
                                        input_shape=dataloader.get_data_tensor_shape(),
                                        model_file_name=input_model_file_name,
                                        key=key)

    # Build ground truth and loss mask rasterizers.
    bbox_rasterizer, loss_mask_rasterizer =\
        build_rasterizers(experiment_spec,
                          gridbox_model.input_width, gridbox_model.input_height,
                          gridbox_model.output_width, gridbox_model.output_height)

    # Build an ObjectiveLabelFilter for loss mask generation.
    loss_mask_label_filter = build_objective_label_filter(
        objective_label_filter_proto=experiment_spec.loss_mask_label_filter,
        target_class_to_source_classes_mapping=dataloader.target_class_to_source_classes_mapping,
        learnable_objective_names=[x.name for x in gridbox_model.objective_set.learnable_objectives]
    )

    # Set up validation.
    evaluation_config = build_evaluation_config(experiment_spec.evaluation_config,
                                                gridbox_model.target_class_names)
    validation_period = evaluation_config.validation_period_during_training
    use_early_stopping = (experiment_spec.training_config.
                          learning_rate.HasField("early_stopping_annealing_schedule"))
    learning_rate = None
    early_stopping_hook = None
    # Build learning rate and hook for early stopping.
    if use_early_stopping:
        learning_rate, hook = build_early_stopping_annealing_schedule(evaluation_config,
                                                                      steps_per_epoch,
                                                                      num_validation_steps,
                                                                      results_dir,
                                                                      experiment_spec,
                                                                      None)
        early_stopping_hook = hook
        hooks.append(hook)
    # Default learning rate.
    else:
        learning_rate = build_learning_rate_schedule(experiment_spec.training_config.learning_rate,
                                                     num_training_steps)
    loggable_tensors.update({
        "learning_rate": learning_rate
    })
    tf.summary.scalar("learning_rate", learning_rate)

    # Build training graph.
    build_training_graph(experiment_spec,
                         gridbox_model,
                         loss_mask_label_filter,
                         bbox_rasterizer,
                         loss_mask_rasterizer,
                         dataloader,
                         learning_rate,
                         cost_auto_weight_hook.cost_combiner_func)

    if is_master and validation_period > 0:
        evaluator = build_validation_graph(experiment_spec,
                                           gridbox_model,
                                           loss_mask_label_filter,
                                           bbox_rasterizer,
                                           loss_mask_rasterizer,
                                           dataloader,
                                           num_validation_steps,
                                           cost_auto_weight_hook.cost_combiner_func)

        num_epochs = experiment_spec.training_config.num_epochs
        first_validation_epoch = evaluation_config.first_validation_epoch

        # This logic is the only one that currently seems to work for early stopping:
        # - Can't build validation graph before training graph (if we only build
        #   validation graph on master, horovod complains about missing broadcasts,
        #   but if we build validation graph on all nodes, we get a lot of errors at
        #   end of training, complaining some variables didn't get used).
        # - Need the learning rate to build training graph, so need to build stopping
        #   hook before building training graph
        # - Need validation cost tensor for stopping hook, so need the validation graph
        #   to build stopping hook
        if use_early_stopping:
            early_stopping_hook.validation_cost = gridbox_model.validation_cost
        else:
            validation_hook = ValidationHook(evaluator, validation_period, num_epochs,
                                             steps_per_epoch, results_dir, first_validation_epoch)
            hooks.append(validation_hook)

    # Train the model.
    run_training_loop(experiment_spec, results_dir, gridbox_model, hooks, steps_per_epoch,
                      output_model_file_name, maglev_experiment, model_version_labels,
                      visualizer_config, key)
    status_logging.get_status_logger().write(data=None, message="Training op complete.")


def build_early_stopping_annealing_schedule(evaluation_config, steps_per_epoch,
                                            num_validation_steps, results_dir, experiment_spec,
                                            validation_cost):
    """Build early stopping annealing hook and learning rate.

    Args:
        evaluation_config (nvidia_tao_tf1.cv.detectnet_v2.evaluation.EvaluationConfig):
            Configuration for evaluation.
        steps_per_epoch (int): Number of steps per epoch.
        num_validation_steps (int): Number of steps needed for a pass over validation data.
        results_dir (str): Directory for results. Will be used to write tensorboard logs.
        experiment_spec (nvidia_tao_tf1.cv.detectnet_v2.proto.experiment_pb2):
            Experiment spec message.
        validation_cost (Tensor): Validation cost tensor. Can be None for workers, since
            validation cost is only computed on master.

    """
    stopping_hook = build_early_stopping_hook(evaluation_config, steps_per_epoch,
                                              os.path.join(results_dir, 'val'),
                                              num_validation_steps, experiment_spec,
                                              validation_cost=validation_cost)
    return stopping_hook.learning_rate, stopping_hook


def run_experiment(config_path, results_dir, pretrained_model_file=None, model_name="model",
                   override_spec_path=None, model_version_labels=None, key=None,
                   wandb_logged_in=False):
    """
    Launch experiment that trains the model.

    NOTE: Do not change the argument names without verifying that cluster submission works.

    Args:
        config_path (list): List containing path to a text file containing a complete experiment
            configuration and possibly a path to a .yml file containing override parameter values.
        results_dir (str): Path to a folder where various training outputs will be written.
            If the folder does not already exist, it will be created.
        pretrained_model_file (str): Optional path to a pretrained model file. This maybe invoked
            from the CLI if needed. For now, we have disabled support to maintain consistency
            across all magnet apps.
        model_name (str): Model name to be used as a part of model file name.
        override_spec_path (str): Absolute path to yaml file which is used to overwrite some of the
            experiment spec parameters.
        model_version_labels (dict): Labels to attach to the created ModelVersions.
        key (str): Key to save and load models from tlt.
        wandb_logger_in (bool): Flag on whether wandb was logged in.
    """
    model_path = get_weights_dir(results_dir)
    # Load experiment spec.
    if config_path is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", config_path)

        # The spec in experiment_spec_path has to be complete.
        # Default spec is not merged into experiment_spec.
        experiment_spec = load_experiment_spec(
            config_path, merge_from_default=False, validation_schema="train_val"
        )
    else:
        logger.info("Loading default KITTI single class experiment spec.")
        experiment_spec = load_experiment_spec()

    # TODO: vpraveen <test without visualizer>
    # Set up visualization.
    is_master = distribution.get_distributor().is_master()
    visualizer_config = experiment_spec.training_config.visualizer
    # Disable visualization for other than the master process.
    if is_master:
        # Setup wandb initializer.
        if visualizer_config.HasField("wandb_config"):
            wandb_config = visualizer_config.wandb_config
            wandb_name = f"{wandb_config.name}" if wandb_config.name \
                else f"{model_name}"
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
            logger.info("Integrating with clearml.")
            clearml_config = visualizer_config.clearml_config
            get_clearml_task(clearml_config, "detectnet_v2")
    else:
        visualizer_config.enabled = False
    Visualizer.build_from_config(visualizer_config)

    # If hyperopt is used, sample hyperparameters and apply them to spec.
    # @TODO: disabling hyperopt for this release.
    # experiment_spec, maglev_experiment = sample_hyperparameters_and_apply_to_spec(experiment_spec)
    maglev_experiment = None

    model_file = os.path.join(model_path, '%s.hdf5' % model_name)

    # Extract core model config, which might be wrapped inside a TemporalModelConfig.
    model_config = get_base_model_config(experiment_spec)
    # Pretrained model can be provided either through CLI or spec. Expand and validate the path.
    assert not (pretrained_model_file and model_config.pretrained_model_file), \
        "Provide only one pretrained model file."
    pretrained_model_file = pretrained_model_file or model_config.pretrained_model_file
    input_model_file_name = get_pretrained_model_path(pretrained_model_file)

    output_model_file_name = model_file

    # Dump experiment spec to result directory.
    if distribution.get_distributor().is_master():
        with open(os.path.join(results_dir, 'experiment_spec.txt'), 'w') as f:
            f.write(str(experiment_spec))

    # Train a model.
    train_gridbox(results_dir, experiment_spec, output_model_file_name, input_model_file_name,
                  maglev_experiment, model_version_labels, key=key)
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.SUCCESS,
        message="DetectNet_v2 training job complete."
    )


def build_command_line_parser(parser=None):
    """
    Parse command-line flags passed to the training script.

    Returns:
      Namespace with all parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='train', description='Train a DetectNet_v2 model.')

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
        help='Name of the model file. If not given, then defaults to model.hdf5.'
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
        help='The key to load pretrained weights and save intermediate snapshots and final model.'
    )
    parser.add_argument(
        '--enable_determinism',
        action="store_true",
        help="Flag to enable deterministic training.",
        default=False
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


def enable_deterministic_training():
    """Define relevant trainer environment variables."""
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"


@time_function(__name__)
def main(args=None):
    """Run the training process."""
    args = parse_command_line_args(args)

    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'

    # Configure the logger.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=verbosity)

    # Setting results dir to realpath if the user
    # doesn't provide an absolute path.
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.realpath(results_dir)

    wandb_logged_in = False
    # Enable Horovod distributor for multi-GPU training.
    distribution.set_distributor(distribution.HorovodDistributor())
    is_master = distribution.get_distributor().is_master()

    try:
        if is_master:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            wandb_logged_in = check_wandb_logged_in()
        # Writing out status file for TLT.
        status_file = os.path.join(results_dir, "status.json")
        status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=is_master,
                verbosity=logger.getEffectiveLevel(),
                append=True
            )
        )
        status_logging.get_status_logger().write(
            data=None,
            status_level=status_logging.Status.STARTED,
            message="Starting DetectNet_v2 Training job"
        )
        if args.enable_determinism:
            logger.info("Enabling deterministic training.")
            enable_deterministic_training()
        run_experiment(
            config_path=args.experiment_spec_file,
            results_dir=results_dir,
            model_name=args.model_name,
            key=args.key,
            wandb_logged_in=wandb_logged_in
        )
    except (KeyboardInterrupt, SystemExit) as e:
        logger.info("Training was interrupted.")
        status_logging.get_status_logger().write(
            data={"Error": "{}".format(e)},
            message="Training was interrupted",
            status_level=status_logging.Status.FAILURE
        )
    finally:
        if distribution.get_distributor().is_master():
            if wandb_logged_in:
                wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if type(e) == tf.errors.ResourceExhaustedError:
            logger = logging.getLogger(__name__)
            logger.error(
                "Ran out of GPU memory, please lower the batch size, use a smaller input "
                "resolution, or use a smaller backbone."
            )
            status_logging.get_status_logger().write(
                message="Ran out of GPU memory, please lower the batch size, use a smaller input "
                        "resolution, or use a smaller backbone.",
                verbosity_level=status_logging.Verbosity.INFO,
                status_level=status_logging.Status.FAILURE
            )
            exit(1)
        else:
            # throw out the error as-is if they are not OOM error
            status_logging.get_status_logger().write(
                message=str(e),
                status_level=status_logging.Status.FAILURE
            )
            raise e
