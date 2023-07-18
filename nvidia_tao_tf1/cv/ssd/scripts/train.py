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

"""Perform continuous SSD training on a tfrecords dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
from math import ceil
from multiprocessing import cpu_count
import os
import tempfile

from google.protobuf.json_format import MessageToDict
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.optimizers import SGD
import tensorflow as tf

from nvidia_tao_tf1.core.export._quantized import check_for_quantized_layers
from nvidia_tao_tf1.cv.common.callbacks.enc_model_saver_callback import KerasModelSaver
from nvidia_tao_tf1.cv.common.callbacks.loggers import TAOStatusLogger
from nvidia_tao_tf1.cv.common.evaluator.ap_evaluator import APEvaluator
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.common.utils import check_tf_oom, hvd_keras, initialize, reg_dict
from nvidia_tao_tf1.cv.common.utils import OneIndexedCSVLogger as CSVLogger
from nvidia_tao_tf1.cv.common.utils import SoftStartAnnealingLearningRateScheduler as LRS
from nvidia_tao_tf1.cv.ssd.architecture.ssd_loss import SSDLoss
from nvidia_tao_tf1.cv.ssd.builders import dataset_builder
from nvidia_tao_tf1.cv.ssd.builders import eval_builder
from nvidia_tao_tf1.cv.ssd.builders import model_builder
from nvidia_tao_tf1.cv.ssd.callbacks.detection_metric_callback_bg import DetectionMetricCallbackBG
from nvidia_tao_tf1.cv.ssd.callbacks.tb_callback import SSDTensorBoard, SSDTensorBoardImage
import nvidia_tao_tf1.cv.ssd.models.patch_keras
from nvidia_tao_tf1.cv.ssd.utils.model_io import CUSTOM_OBJS, load_model
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import (
    eval_str,
    load_experiment_spec,
    validate_train_spec
)
from nvidia_tao_tf1.cv.ssd.utils.tensor_utils import get_init_ops

nvidia_tao_tf1.cv.ssd.models.patch_keras.patch()

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)
verbose = 0


def load_model_as_pretrain(model_path, experiment_spec, is_dssd, use_dali=False,
                           local_rank=0, shard_id=0, num_shards=1,
                           key=None, kernel_regularizer=None,
                           resume_from_training=False):
    """
    Load a model as pretrained weights.

    If the model is pruned, just return the model.

    Always return two models, first is for training, last is a template with input placeholder.
    """

    train_dataset, val_dataset = dataset_builder.build_dataset(experiment_spec, is_dssd,
                                                               device_id=local_rank,
                                                               shard_id=shard_id,
                                                               num_shards=num_shards)

    if resume_from_training:
        if use_dali:
            model_load = load_model(model_path, experiment_spec, is_dssd, None, key)
            model_load_train = load_model(model_path, experiment_spec, is_dssd,
                                          train_dataset.images, key)
            optimizer = model_load.optimizer
            return model_load_train, model_load, train_dataset, val_dataset, optimizer

        model_load = load_model(model_path, experiment_spec, is_dssd, None, key)
        return model_load, model_load, train_dataset, val_dataset, model_load.optimizer

    if use_dali:
        input_tensor = train_dataset.images
    else:
        input_tensor = None

    model_train, model_eval = \
        model_builder.build(experiment_spec, is_dssd,
                            input_tensor=input_tensor,
                            kernel_regularizer=kernel_regularizer)

    model_load = load_model(model_path, experiment_spec, is_dssd, None, key)

    strict_mode = True
    error_layers = []
    loaded_layers = []
    for layer in model_train.layers[1:]:
        # The layer must match up to ssd layers.
        if layer.name.find('ssd_') != -1:
            strict_mode = False
        try:
            l_return = model_load.get_layer(layer.name)
        except ValueError:
            if layer.name[-3:] != 'qdq' and strict_mode:
                error_layers.append(layer.name)
            # Some layers are not there
            continue
        try:
            wts = l_return.get_weights()
            if len(wts) > 0:
                layer.set_weights(wts)
                loaded_layers.append(layer.name)
        except ValueError:
            if strict_mode:
                # This is a pruned model
                print('The shape of this layer does not match original model:', layer.name)
                print('Loading the model as a pruned model.')
                model_config = model_load.get_config()
                for layer, layer_config in zip(model_load.layers, model_config['layers']):
                    if hasattr(layer, 'kernel_regularizer'):
                        layer_config['config']['kernel_regularizer'] = kernel_regularizer
                reg_model = keras.models.Model.from_config(model_config, custom_objects=CUSTOM_OBJS)
                reg_model.set_weights(model_load.get_weights())

                os_handle, temp_file_name = tempfile.mkstemp(suffix='.hdf5')
                os.close(os_handle)
                reg_model.save(temp_file_name, overwrite=True, include_optimizer=False)

                if use_dali:
                    train_model = load_model(temp_file_name, experiment_spec, is_dssd,
                                             train_dataset.images, None)
                else:
                    train_model = load_model(temp_file_name, experiment_spec, is_dssd,
                                             None, None)

                os.remove(temp_file_name)
                return train_model, model_load, train_dataset, val_dataset, None
            error_layers.append(layer.name)
    if len(error_layers) > 0:
        print('Weights for those layers can not be loaded:', error_layers)
        print('STOP trainig now and check the pre-train model if this is not expected!')

    print("Layers that load weights from the pretrained model:", loaded_layers)

    return model_train, model_eval, train_dataset, val_dataset, None


def run_experiment(config_path, results_dir, resume_weights,
                   key, check_arch=None, init_epoch=1, use_multiprocessing=False):
    """
    Launch experiment that trains the model.

    NOTE: Do not change the argument names without verifying that cluster submission works.

    Args:
        config_path (str): Path to a text file containing a complete experiment configuration.
        results_dir (str): Path to a folder where various training outputs will be written.
        If the folder does not already exist, it will be created.
        resume_weights (str): Optional path to a pretrained model file.
        init_epoch (int): The number of epoch to resume training.
        check_arch (enum): choose from [None, 'ssd', 'dssd']. If not None, raise error if spec file
            says otherwise.
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
    experiment_spec, is_dssd = load_experiment_spec(config_path, check_arch)
    validate_train_spec(experiment_spec)

    initialize(experiment_spec.random_seed, hvd)
    cls_mapping = experiment_spec.dataset_config.target_class_mapping
    classes = sorted({str(x) for x in cls_mapping.values()})

    if is_master:
        if experiment_spec.training_config.HasField("visualizer"):
            network_name = "dssd" if is_dssd else "ssd"
            visualizer_config = experiment_spec.training_config.visualizer
            if visualizer_config.HasField("clearml_config"):
                clearml_config = visualizer_config.clearml_config
                get_clearml_task(clearml_config, network_name)
            if visualizer_config.HasField("wandb_config"):
                wandb_config = visualizer_config.wandb_config
                wandb_logged_in = check_wandb_logged_in()
                wandb_name = f"{wandb_config.name}" if wandb_config.name else \
                    f"{network_name}_training"
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
    batch_size_per_gpu = experiment_spec.training_config.batch_size_per_gpu
    lrconfig = experiment_spec.training_config.learning_rate.soft_start_annealing_schedule
    # config kernel regularizer
    reg_type = experiment_spec.training_config.regularizer.type
    reg_weight = experiment_spec.training_config.regularizer.weight
    kr = None
    if reg_type:
        if reg_type > 0:
            assert 0 < reg_weight < 1, \
                "Weight decay should be no less than 0 and less than 1"
            kr = reg_dict[reg_type](reg_weight)

    if experiment_spec.ssd_config.alpha != 0.0:
        alpha = experiment_spec.ssd_config.alpha
    else:
        alpha = 1.0

    if experiment_spec.ssd_config.neg_pos_ratio != 0.0:
        neg_pos_ratio = experiment_spec.ssd_config.neg_pos_ratio
    else:
        neg_pos_ratio = 3

    use_dali = False
    # @TODO(tylerz): if there is tfrecord, then use dali.
    if experiment_spec.dataset_config.data_sources[0].tfrecords_path != "":
        use_dali = True

    # build train/val data and model, configure optimizer && loss
    sgd = SGD(lr=0, decay=0, momentum=0.9, nesterov=False)
    ssd_loss = SSDLoss(neg_pos_ratio=neg_pos_ratio, alpha=alpha)
    if resume_weights is not None:
        if init_epoch == 1:
            resume_from_training = False
        else:
            resume_from_training = True

        logger.info("Loading pretrained weights. This may take a while...")
        model, model_eval, train_dataset, val_dataset, optimizer = \
            load_model_as_pretrain(model_path=resume_weights,
                                   experiment_spec=experiment_spec,
                                   is_dssd=is_dssd,
                                   use_dali=use_dali,
                                   local_rank=hvd.local_rank(),
                                   shard_id=hvd.rank(),
                                   num_shards=hvd.size(),
                                   key=key,
                                   kernel_regularizer=kr,
                                   resume_from_training=resume_from_training)

        if use_dali:
            label_tensor = [train_dataset.labels]
        else:
            label_tensor = None

        # check if the loaded model is QAT
        qat_flag = experiment_spec.training_config.enable_qat
        if not qat_flag and check_for_quantized_layers(model_eval):
            raise ValueError("QAT training is disabled but the pretrained model is a QAT model.")
        if qat_flag and not check_for_quantized_layers(model_eval):
            raise ValueError("QAT training is enabled but the pretrained model is not a QAT model.")

        if init_epoch == 1:
            print("Initialize optimizer")
            model.compile(optimizer=hvd.DistributedOptimizer(sgd),
                          loss=ssd_loss.compute_loss,
                          target_tensors=label_tensor)
        else:
            print("Resume optimizer from pretrained model")
            model.compile(optimizer=hvd.DistributedOptimizer(optimizer),
                          loss=ssd_loss.compute_loss,
                          target_tensors=label_tensor)
    else:
        train_dataset, val_dataset = dataset_builder.build_dataset(experiment_spec, is_dssd,
                                                                   device_id=hvd.local_rank(),
                                                                   shard_id=hvd.rank(),
                                                                   num_shards=hvd.size())
        if use_dali:
            input_tensor = train_dataset.images
        else:
            input_tensor = None

        model, model_eval = \
            model_builder.build(experiment_spec, is_dssd,
                                input_tensor=input_tensor,
                                kernel_regularizer=kr)

        print("Initialize optimizer")
        if use_dali:
            label_tensor = [train_dataset.labels]
        else:
            label_tensor = None

        model.compile(optimizer=hvd.DistributedOptimizer(sgd),
                      loss=ssd_loss.compute_loss,
                      target_tensors=label_tensor)

    # configure LR scheduler
    total_num = train_dataset.n_samples
    iters_per_epoch = int(ceil(total_num / batch_size_per_gpu / hvd.size()))
    max_iterations = num_epochs * iters_per_epoch
    lr_scheduler = LRS(base_lr=lrconfig.max_learning_rate * hvd.size(),
                       min_lr_ratio=lrconfig.min_learning_rate / lrconfig.max_learning_rate,
                       soft_start=lrconfig.soft_start,
                       annealing_start=lrconfig.annealing,
                       max_iterations=max_iterations)

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.MetricAverageCallback(),
                 lr_scheduler,
                 TerminateOnNaN()]
    init_step = (init_epoch - 1) * iters_per_epoch
    lr_scheduler.reset(init_step)

    sess.run(get_init_ops())

    if hvd.rank() == 0:
        model.summary()
        logger.info("Number of images in the training dataset:\t{:>6}"
                    .format(train_dataset.n_samples))
        logger.info("Number of images in the validation dataset:\t{:>6}"
                    .format(val_dataset.n_samples))
        if not os.path.exists(os.path.join(results_dir, 'weights')):
            os.mkdir(os.path.join(results_dir, 'weights'))

        if experiment_spec.ssd_config.arch in ['resnet', 'darknet', 'vgg']:
            # append nlayers into meta_arch_name
            arch_name = experiment_spec.ssd_config.arch + \
                str(experiment_spec.ssd_config.nlayers)
        else:
            arch_name = experiment_spec.ssd_config.arch

        meta_arch_name = 'dssd_' if is_dssd else 'ssd_'
        ckpt_path = str(os.path.join(results_dir, 'weights',
                                     meta_arch_name + arch_name + '_epoch_{epoch:03d}.hdf5'))
        save_period = experiment_spec.training_config.checkpoint_interval or 1
        # This callback will update model_eval and save the model.
        model_checkpoint = KerasModelSaver(ckpt_path, key, save_period, verbose=1)

        csv_path = os.path.join(
            results_dir, meta_arch_name + 'training_log_' + arch_name + '.csv')
        csv_logger = CSVLogger(filename=csv_path,
                               separator=',',
                               append=False)
        callbacks.append(model_checkpoint)
    if len(val_dataset) > 0:
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

        # Load evaluation parameters
        every_k = experiment_spec.eval_config.validation_period_during_training
        ap_mode = experiment_spec.eval_config.average_precision_mode
        matching_iou = experiment_spec.eval_config.matching_iou_threshold
        matching_iou = matching_iou if matching_iou > 0 else 0.5
        ap_mode_dict = {0: "sample", 1: "integrate"}
        average_precision_mode = ap_mode_dict[ap_mode]

        evaluator = APEvaluator(len(classes)+1,
                                conf_thres=experiment_spec.nms_config.confidence_threshold,
                                matching_iou_threshold=matching_iou,
                                average_precision_mode=average_precision_mode)

        ssd_loss_val = SSDLoss(neg_pos_ratio=neg_pos_ratio, alpha=alpha)
        n_box, n_attr = model_eval.layers[-1].output_shape[1:]
        op_pred = tf.placeholder(tf.float32, shape=(None, n_box, n_attr))
        op_true = tf.placeholder(tf.float32, shape=(None, n_box, n_attr))
        loss_ops = [op_true, op_pred,
                    ssd_loss_val.compute_loss(op_true, op_pred)]

        eval_callback = DetectionMetricCallbackBG(ap_evaluator=evaluator,
                                                  built_eval_model=built_eval_model,
                                                  eval_sequence=val_dataset,
                                                  loss_ops=loss_ops,
                                                  eval_model=model_eval,
                                                  metric_interval=every_k,
                                                  verbose=verbose)
        K.set_learning_phase(1)
        callbacks.append(eval_callback)

    if hvd.rank() == 0:
        callbacks.append(csv_logger)
        status_logger = TAOStatusLogger(
            results_dir,
            append=True,
            num_epochs=num_epochs,
            is_master=hvd.rank() == 0,
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
            variances = eval_str(experiment_spec.ssd_config.variances)
            tbimg_cb = SSDTensorBoardImage(tb_log_dir, experiment_spec, variances,
                                           experiment_spec.training_config.visualizer.num_images)
            fetches = [tf.assign(tbimg_cb.img, model.inputs[0], validate_shape=False),
                       tf.assign(tbimg_cb.label, model.targets[0], validate_shape=False)]
            model._function_kwargs = {'fetches': fetches}
            callbacks.append(tbimg_cb)

    if use_dali:
        model.fit(steps_per_epoch=iters_per_epoch,
                  epochs=num_epochs,
                  callbacks=callbacks,
                  initial_epoch=init_epoch - 1,
                  verbose=verbose)
    else:
        # @TODO(tylerz): run into deadlock on P40 with use_multiprocessing = True on small dataset
        # So enable multi-thread mode if dataset is small.
        # https://github.com/keras-team/keras/issues/10340.
        workers = experiment_spec.training_config.n_workers or (cpu_count()-1)
        model.fit_generator(generator=train_dataset,
                            steps_per_epoch=iters_per_epoch,
                            epochs=num_epochs,
                            callbacks=callbacks,
                            initial_epoch=init_epoch - 1,
                            use_multiprocessing=use_multiprocessing,
                            max_queue_size=experiment_spec.training_config.max_queue_size or 20,
                            shuffle=False,
                            workers=workers,
                            verbose=verbose)

    status_logging.get_status_logger().write(
        status_level=status_logging.Status.SUCCESS,
        message="Training finished successfully."
    )


def build_command_line_parser(parser=None):
    '''build parser.'''
    if parser is None:
        parser = argparse.ArgumentParser(
            prog='train', description='Train an SSD model.')

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
        '-m',
        '--resume_model_weights',
        type=str,
        default=None,
        help='Path to a model to continue training.'
    )
    parser.add_argument(
        '--initial_epoch',
        type=int,
        default=1,
        help='Set resume epoch'
    )
    parser.add_argument(
        '--arch',
        choices=[None, 'ssd', 'dssd'],
        default=None,
        help='Which architecture to uses'
    )
    parser.add_argument(
        '--use_multiprocessing',
        action="store_true",
        default=False
    )
    return parser


def parse_command_line(args):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


@check_tf_oom
def main(args=None):
    """Run the training process."""
    args = parse_command_line(args)
    try:
        run_experiment(config_path=args.experiment_spec_file,
                       results_dir=args.results_dir,
                       resume_weights=args.resume_model_weights,
                       init_epoch=args.initial_epoch,
                       key=args.key,
                       check_arch=args.arch,
                       use_multiprocessing=args.use_multiprocessing)
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
