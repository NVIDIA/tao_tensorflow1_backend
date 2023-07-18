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

"""Perform continuous RetinaNet training on a tfrecords dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
from math import ceil
from multiprocessing import cpu_count
import os

from google.protobuf.json_format import MessageToDict
from keras import backend as K
from keras.callbacks import EarlyStopping, TerminateOnNaN
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.core.export._quantized import check_for_quantized_layers
from nvidia_tao_tf1.cv.common.callbacks.enc_model_saver_callback import KerasModelSaver
from nvidia_tao_tf1.cv.common.callbacks.loggers import TAOStatusLogger
from nvidia_tao_tf1.cv.common.evaluator.ap_evaluator import APEvaluator
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.common.utils import ap_mode_dict
from nvidia_tao_tf1.cv.common.utils import build_class_weights
from nvidia_tao_tf1.cv.common.utils import build_lrs_from_config
from nvidia_tao_tf1.cv.common.utils import (
    build_optimizer_from_config,
    build_regularizer_from_config
)
from nvidia_tao_tf1.cv.common.utils import check_tf_oom, hvd_keras, initialize
from nvidia_tao_tf1.cv.common.utils import OneIndexedCSVLogger as CSVLogger
from nvidia_tao_tf1.cv.common.utils import parse_model_load_from_config

from nvidia_tao_tf1.cv.retinanet.box_coder.input_encoder import InputEncoder
from nvidia_tao_tf1.cv.retinanet.box_coder.input_encoder_tf import InputEncoderTF
from nvidia_tao_tf1.cv.retinanet.builders import eval_builder, input_builder
from nvidia_tao_tf1.cv.retinanet.callbacks.retinanet_metric_callback import RetinaMetricCallback
from nvidia_tao_tf1.cv.retinanet.losses.focal_loss import FocalLoss
from nvidia_tao_tf1.cv.retinanet.utils.helper import eval_str
from nvidia_tao_tf1.cv.retinanet.utils.model_io import load_model_as_pretrain
from nvidia_tao_tf1.cv.retinanet.utils.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.ssd.callbacks.tb_callback import SSDTensorBoard, SSDTensorBoardImage

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)
verbose = 0


def run_experiment(config_path, results_dir, key, root_path=None, initial_epoch=0):
    """
    Launch experiment that trains the model.

    NOTE: Do not change the argument names without verifying that cluster submission works.

    Args:
        config_path (str): Path to a text file containing a complete experiment configuration.
        results_dir (str): Path to a folder where various training outputs will be written.
        If the folder does not already exist, it will be created.
        key (str): encryption key.
        root_path (str): for AVDC training, INTERNAL only.
    """
    hvd = hvd_keras()
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    K.set_session(sess)
    verbose = 1 if hvd.rank() == 0 else 0
    is_master = hvd.rank() == 0
    if is_master and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=is_master,
            verbosity=1,
            append=True
        )
    )
    # Load experiment spec.
    if config_path is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", config_path)
        # The spec in config_path has to be complete.
        # Default spec is not merged into experiment_spec.
        experiment_spec = load_experiment_spec(config_path, merge_from_default=False)
    else:
        logger.info("Loading default experiment spec.")
        experiment_spec = load_experiment_spec()

    initialize(experiment_spec.random_seed, hvd)
    training_config = experiment_spec.training_config
    if is_master:
        if training_config.HasField("visualizer"):
            if training_config.visualizer.HasField("clearml_config"):
                clearml_config = training_config.visualizer.clearml_config
                get_clearml_task(clearml_config, "retinanet")
            if training_config.visualizer.HasField("wandb_config"):
                wandb_config = training_config.visualizer.wandb_config
                wandb_logged_in = check_wandb_logged_in()
                wandb_name = f"{wandb_config.name}" if wandb_config.name else \
                    "retinanet_training"
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

    # Load training parameters
    num_epochs = experiment_spec.training_config.num_epochs
    ckpt_interval = experiment_spec.training_config.checkpoint_interval or 1
    train_bs = experiment_spec.training_config.batch_size_per_gpu
    # Class mapping
    cls_mapping = experiment_spec.dataset_config.target_class_mapping
    classes = sorted({str(x) for x in cls_mapping.values()})
    # n_classes + 1 for background class
    n_classes = len(classes) + 1
    # Choose DALI
    use_dali = False
    if experiment_spec.dataset_config.data_sources[0].tfrecords_path != "":
        use_dali = True
        logger.info("Using DALI dataloader...")
    # build dataset
    train_dataset = input_builder.build(experiment_spec,
                                        training=True,
                                        device_id=hvd.local_rank(),
                                        root_path=root_path,
                                        shard_id=hvd.rank(),
                                        num_shards=hvd.size(),
                                        use_dali=use_dali)
    val_dataset = input_builder.build(experiment_spec,
                                      training=False,
                                      root_path=root_path,
                                      use_dali=False)
    # config regularizer
    kr = build_regularizer_from_config(experiment_spec.training_config.regularizer)
    # configure optimizer
    optim = build_optimizer_from_config(experiment_spec.training_config.optimizer,
                                        clipnorm=2.0)
    focal_loss = FocalLoss(loc_loss_weight=experiment_spec.retinanet_config.loss_loc_weight,
                           alpha=experiment_spec.retinanet_config.focal_loss_alpha,
                           gamma=experiment_spec.retinanet_config.focal_loss_gamma)
    # config model loading
    load_path, load_graph, reset_optim, init_epoch = \
        parse_model_load_from_config(experiment_spec.training_config)
    if initial_epoch > 0:
        init_epoch = initial_epoch
    model_train, model_eval, optim_load = load_model_as_pretrain(
        load_path,
        load_graph,
        n_classes,
        experiment_spec=experiment_spec,
        input_tensor=train_dataset.images if use_dali else None,
        kernel_regularizer=kr,
        key=key,
        resume_training=not reset_optim)

    # check if the loaded model is QAT
    if not experiment_spec.training_config.enable_qat and check_for_quantized_layers(model_eval):
        raise ValueError("QAT training is disabled but the pretrained model is a QAT model.")
    if experiment_spec.training_config.enable_qat and not check_for_quantized_layers(model_eval):
        raise ValueError("QAT training is enabled but the pretrained model is not a QAT model.")

    if optim_load is not None:
        optim = optim_load

    # set encoder for data sequences
    predictor_sizes = [model_train.get_layer('P3_relu').output_shape[2:],
                       model_train.get_layer('P4_relu').output_shape[2:],
                       model_train.get_layer('P5_relu').output_shape[2:],
                       model_train.get_layer('P6_relu').output_shape[2:],
                       model_train.get_layer('P7_relu').output_shape[2:]]
    # encoder parameters
    img_height = experiment_spec.augmentation_config.output_height
    img_width = experiment_spec.augmentation_config.output_width
    scales = eval_str(experiment_spec.retinanet_config.scales)
    aspect_ratios_global = eval_str(experiment_spec.retinanet_config.aspect_ratios_global)
    aspect_ratios_per_layer = eval_str(experiment_spec.retinanet_config.aspect_ratios)
    steps = eval_str(experiment_spec.retinanet_config.steps)
    offsets = eval_str(experiment_spec.retinanet_config.offsets)
    variances = eval_str(experiment_spec.retinanet_config.variances)
    min_scale = experiment_spec.retinanet_config.min_scale
    max_scale = experiment_spec.retinanet_config.max_scale
    two_boxes_for_ar1 = experiment_spec.retinanet_config.two_boxes_for_ar1
    clip_boxes = experiment_spec.retinanet_config.clip_boxes
    pos_iou_thresh = experiment_spec.retinanet_config.pos_iou_thresh or 0.5
    neg_iou_thresh = experiment_spec.retinanet_config.neg_iou_thresh or 0.4
    n_anchor_levels = experiment_spec.retinanet_config.n_anchor_levels or 3

    # set the background weights
    cls_weights = [1.0]
    cls_weights.extend(build_class_weights(experiment_spec))
    # encoder for keras seq training
    input_encoder = InputEncoder(
        img_height=img_height,
        img_width=img_width,
        n_classes=n_classes,
        predictor_sizes=predictor_sizes,
        scales=scales,
        min_scale=min_scale,
        max_scale=max_scale,
        aspect_ratios_global=aspect_ratios_global,
        aspect_ratios_per_layer=aspect_ratios_per_layer,
        two_boxes_for_ar1=two_boxes_for_ar1,
        steps=steps,
        n_anchor_levels=n_anchor_levels,
        offsets=offsets,
        clip_boxes=clip_boxes,
        variances=variances,
        pos_iou_threshold=pos_iou_thresh,
        neg_iou_limit=neg_iou_thresh,
        class_weights=cls_weights)

    input_encoder_tf = InputEncoderTF(
        img_height=img_height,
        img_width=img_width,
        n_classes=n_classes,
        predictor_sizes=predictor_sizes,
        scales=scales,
        min_scale=min_scale,
        max_scale=max_scale,
        aspect_ratios_global=aspect_ratios_global,
        aspect_ratios_per_layer=aspect_ratios_per_layer,
        two_boxes_for_ar1=two_boxes_for_ar1,
        steps=steps,
        n_anchor_levels=n_anchor_levels,
        offsets=offsets,
        clip_boxes=clip_boxes,
        variances=variances,
        pos_iou_threshold=pos_iou_thresh,
        neg_iou_limit=neg_iou_thresh,
        gt_normalized=True,
        class_weights=cls_weights)

    # encoder for eval.
    def eval_encode_fn(gt_label):
        bboxes = gt_label[:, -4:]
        cls_id = gt_label[:, 0:1]
        gt_label_without_diff = np.concatenate((cls_id, bboxes), axis=-1)
        return (input_encoder(gt_label_without_diff), gt_label)

    # set encode_fn
    train_dataset.set_encoder(input_encoder_tf if use_dali else input_encoder)
    val_dataset.set_encoder(eval_encode_fn)

    # configure LR scheduler
    iters_per_epoch = int(ceil(train_dataset.n_samples / hvd.size() / train_bs))
    max_iterations = num_epochs * iters_per_epoch
    lr_scheduler = build_lrs_from_config(experiment_spec.training_config.learning_rate,
                                         max_iterations, hvd.size())
    init_step = init_epoch * iters_per_epoch
    lr_scheduler.reset(init_step)
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.MetricAverageCallback(),
                 lr_scheduler,
                 TerminateOnNaN()]

    model_train.compile(optimizer=hvd.DistributedOptimizer(optim),
                        loss=focal_loss.compute_loss,
                        target_tensors=[train_dataset.labels] if use_dali else None)

    if hvd.rank() == 0:
        model_train.summary()
        logger.info("Number of samples in the training dataset:\t{:>6}"
                    .format(train_dataset.n_samples))
        logger.info("Number of samples in the validation dataset:\t{:>6}"
                    .format(val_dataset.n_samples))
        if not os.path.exists(os.path.join(results_dir, 'weights')):
            os.mkdir(os.path.join(results_dir, 'weights'))

        arch_name = experiment_spec.retinanet_config.arch
        if arch_name in ['resnet', 'darknet', 'vgg']:
            # append nlayers into meta_arch_name
            arch_name = arch_name + str(experiment_spec.retinanet_config.nlayers)

        ckpt_path = str(os.path.join(results_dir, 'weights',
                                     'retinanet_' + arch_name + '_epoch_{epoch:03d}.hdf5'))
        # This callback will update model and save the model.
        model_checkpoint = KerasModelSaver(ckpt_path, key, ckpt_interval, last_epoch=num_epochs,
                                         verbose=1)
        callbacks.append(model_checkpoint)

    if len(val_dataset) > 0:
        # Load evaluation parameters
        validation_interval = experiment_spec.eval_config.validation_period_during_training
        ap_mode = experiment_spec.eval_config.average_precision_mode
        matching_iou = experiment_spec.eval_config.matching_iou_threshold
        # Load NMS parameters
        conf_th = experiment_spec.nms_config.confidence_threshold
        clustering_iou = experiment_spec.nms_config.clustering_iou_threshold
        top_k = experiment_spec.nms_config.top_k
        nms_max_output = top_k
        # build eval graph
        K.set_learning_phase(0)
        built_eval_model = eval_builder.build(model_eval, conf_th,
                                              clustering_iou, top_k,
                                              nms_max_output,
                                              include_encoded_pred=True)

        evaluator = APEvaluator(n_classes,
                                conf_thres=experiment_spec.nms_config.confidence_threshold,
                                matching_iou_threshold=matching_iou,
                                average_precision_mode=ap_mode_dict[ap_mode])

        focal_loss_val = FocalLoss(
            loc_loss_weight=experiment_spec.retinanet_config.loss_loc_weight,
            alpha=experiment_spec.retinanet_config.focal_loss_alpha,
            gamma=experiment_spec.retinanet_config.focal_loss_gamma)
        n_box, n_attr = model_eval.layers[-1].output_shape[1:]
        op_pred = tf.placeholder(tf.float32, shape=(None, n_box, n_attr))
        # +1 for class weights
        op_true = tf.placeholder(tf.float32, shape=(None, n_box, n_attr+1))
        loss_ops = [op_true, op_pred,
                    focal_loss_val.compute_loss(op_true, op_pred)]

        eval_callback = RetinaMetricCallback(
            ap_evaluator=evaluator,
            built_eval_model=built_eval_model,
            eval_sequence=val_dataset,
            loss_ops=loss_ops,
            eval_model=model_eval,
            metric_interval=validation_interval or 10,
            last_epoch=num_epochs,
            verbose=verbose)
        K.set_learning_phase(1)
        callbacks.append(eval_callback)
        # K.set_learning_phase(1)

    if hvd.rank() == 0:
        # This callback logs loss and mAP
        csv_path = os.path.join(results_dir, 'retinanet_training_log_' + arch_name + '.csv')
        csv_logger = CSVLogger(filename=csv_path,
                               separator=',',
                               append=True)

        callbacks.append(csv_logger)
        status_logger = TAOStatusLogger(
            results_dir,
            append=True,
            num_epochs=num_epochs,
            is_master=is_master,
        )
        callbacks.append(status_logger)

    # init EarlyStopping callback:
    if experiment_spec.training_config.HasField("early_stopping"):
        es_config = experiment_spec.training_config.early_stopping
        # align the validation name
        if es_config.monitor == "val_loss":
            es_config.monitor = "validation_loss"
        if es_config.monitor == "validation_loss":
            if len(val_dataset) <= 0:
                raise ValueError("Validation dataset is needed for "
                                 "using validation_loss as the early stopping monitor")
            if experiment_spec.eval_config.validation_period_during_training != 1:
                raise ValueError("validation_period_during_training should be 1 for "
                                 "using validation_loss as the early stopping monitor")
        es_cb = EarlyStopping(monitor=es_config.monitor,
                              min_delta=es_config.min_delta,
                              patience=es_config.patience,
                              verbose=True)
        callbacks.append(es_cb)

    if hvd.rank() == 0:
        if experiment_spec.training_config.visualizer.enabled:
            tb_log_dir = os.path.join(results_dir, "events")
            tb_cb = SSDTensorBoard(log_dir=tb_log_dir, write_graph=False)
            callbacks.append(tb_cb)
            tbimg_cb = SSDTensorBoardImage(tb_log_dir, experiment_spec, variances,
                                           experiment_spec.training_config.visualizer.num_images)
            fetches = [tf.assign(tbimg_cb.img, model_train.inputs[0], validate_shape=False),
                       tf.assign(tbimg_cb.label, model_train.targets[0], validate_shape=False)]
            model_train._function_kwargs = {'fetches': fetches}
            callbacks.append(tbimg_cb)

    if use_dali:
        model_train.fit(
            steps_per_epoch=iters_per_epoch,
            epochs=num_epochs,
            callbacks=callbacks,
            initial_epoch=init_epoch,
            verbose=verbose)
    else:
        model_train.fit_generator(
            generator=train_dataset,
            steps_per_epoch=iters_per_epoch,
            epochs=num_epochs,
            callbacks=callbacks,
            initial_epoch=init_epoch,
            workers=experiment_spec.training_config.n_workers or (cpu_count()-1),
            shuffle=False,
            use_multiprocessing=experiment_spec.training_config.use_multiprocessing,
            max_queue_size=experiment_spec.training_config.max_queue_size or 20,
            verbose=verbose)

    status_logging.get_status_logger().write(
        status_level=status_logging.Status.SUCCESS,
        message="Training finished successfully."
    )


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism.

    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='train', description='Train a RetinaNet model.')

    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        required=True,
        help='Path to spec file. Absolute path or relative to working directory. \
            If not specified, default spec from spec_loader.py is used.')
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        required=True,
        help='Path to a folder where experiment outputs should be written.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        default="",
        required=False,
        help='Key to save or load a .tlt model.'
    )
    parser.add_argument(
        '--root_path',
        type=str,
        required=False,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--initial_epoch',
        type=int,
        default=0,
        help=argparse.SUPPRESS
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


@check_tf_oom
def main(args=None):
    """Run the training process."""
    args = parse_command_line_arguments(args)
    try:
        run_experiment(config_path=args.experiment_spec_file,
                       results_dir=args.results_dir,
                       key=args.key,
                       root_path=args.root_path,
                       initial_epoch=args.initial_epoch)
        logger.info("Training finished successfully.")
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
        logger.info("Training was interrupted.")
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == "__main__":
    main()
