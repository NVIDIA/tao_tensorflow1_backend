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
"""BpNet Trainer."""

import glob
import logging
import os
import shutil

import keras

import tensorflow as tf

from nvidia_tao_tf1.blocks.trainer import Trainer
import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.core.utils import mkdir_p, set_random_seed
from nvidia_tao_tf1.cv.common.utilities.serialization_listener import \
    EpochModelSerializationListener
from nvidia_tao_tf1.cv.common.utilities.tlt_utils \
    import get_latest_checkpoint, get_step_from_ckzip, get_tf_ckpt, load_pretrained_weights
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import get_latest_tlt_model, load_model
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.task_progress_monitor_hook import (
    TaskProgressMonitorHook
)
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.utils import get_common_training_hooks

logger = logging.getLogger(__name__)

MODEL_EXTENSION = ".hdf5"


class BpNetTrainer(Trainer):
    """BpNet Trainer class for building training graph and graph execution."""

    @tao_core.coreobject.save_args
    def __init__(self,
                 checkpoint_dir,
                 log_every_n_secs=1,
                 checkpoint_n_epoch=1,
                 num_epoch=2,
                 summary_every_n_steps=1,
                 infrequent_summary_every_n_steps=0,
                 validation_every_n_epoch=20,
                 max_ckpt_to_keep=5,
                 pretrained_weights=None,
                 load_graph=False,
                 inference_spec=None,
                 finetuning_config=None,
                 use_stagewise_lr_multipliers=False,
                 evaluator=None,
                 random_seed=42,
                 key=None,
                 final_model_name='bpnet_model',
                 **kwargs):
        """__init__ method.

        Args:
            checkpoint_dir (str): path to directory containing checkpoints.
            log_every_n_secs (int): log every n secs.
            checkpoint_n_epoch (int): how often to save checkpoint.
            num_epoch (int): Number of epochs to train for.
            summary_every_n_steps (int): summary every n steps.
            infrequent_summary_every_n_steps (int): infrequent summary every n steps.
            validation_every_n_steps (int): run evaluation every n steps.
            max_ckpt_to_keep (int): How many checkpoints to keep.
            pretrained_weighted (str): Pretrained weights path.
            use_stagewise_lr_multipliers (bool): Option to enable use of
                stagewise learning rate multipliers using WeightedMomentumOptimizer.
            evaluator (TAOObject): evaluate predictions and save statistics.
            random_seed (int): random seed.
        """
        super(BpNetTrainer, self).__init__(**kwargs)
        assert checkpoint_n_epoch <= num_epoch, "Checkpoint_n_epochs must be \
        <= num_epochs"
        assert (num_epoch % checkpoint_n_epoch) == 0, "Checkpoint_n_epoch should\
        be a divisor of num_epoch"
        self._checkpoint_dir = checkpoint_dir
        self._pretrained_weights = pretrained_weights
        self._load_graph = load_graph
        self._log_every_n_secs = log_every_n_secs
        self._checkpoint_n_epoch = checkpoint_n_epoch
        self._num_epoch = num_epoch
        self._infrequent_summary_every_n_steps = infrequent_summary_every_n_steps
        self._evaluator = evaluator
        self._random_seed = random_seed
        self._summary_every_n_steps = summary_every_n_steps
        self._max_ckpt_to_keep = max_ckpt_to_keep
        self._validation_every_n_epoch = validation_every_n_epoch
        self._steps_per_epoch = self._total_loss = self._train_op = None
        self.inference_spec = inference_spec
        self._finetuning_config = finetuning_config
        if self._finetuning_config is None:
            self._finetuning_config = {
                'is_finetune_exp': False,
                'checkpoint_path': None,
            }
        self.use_stagewise_lr_multipliers = use_stagewise_lr_multipliers
        self._key = key
        self._generate_output_sequence()
        self.final_model_name = final_model_name
        # Checks
        if self._load_graph:
            assert self._pretrained_weights is not None, "Load graph is True,\
            please specify pretrained model to use to load the graph."
        assert self.inference_spec is not None, "Please specify inference spec\
            path in the config file."

    @property
    def train_op(self):
        """Return train op of Trainer."""
        return self._train_op

    def _check_if_first_run(self):

        files = [
            file for file in glob.glob(
                self._checkpoint_dir +
                '/model.epoch-*')]
        return (not bool(len(files)))

    def _generate_output_sequence(self):
        """Generates required output sequence."""
        stages = self._model._stages
        cmaps = [('cmap', i) for i in range(1, stages + 1)]
        pafs = [('paf', i) for i in range(1, stages + 1)]

        output_seq = []
        output_seq.extend(cmaps)
        output_seq.extend(pafs)

        self.output_seq = output_seq

    def update_regularizers(self, keras_model, kernel_regularizer=None,
                            bias_regularizer=None):
        """Update regularizers for models that are being loaded."""
        model_config = keras_model.get_config()
        for layer, layer_config in zip(keras_model.layers, model_config['layers']):
            # Updating regularizer parameters for conv2d, depthwise_conv2d and dense layers.
            if type(layer) in [keras.layers.convolutional.Conv2D,
                               keras.layers.core.Dense,
                               keras.layers.DepthwiseConv2D]:
                if hasattr(layer, 'kernel_regularizer'):
                    layer_config['config']['kernel_regularizer'] = kernel_regularizer
                if hasattr(layer, 'bias_regularizer'):
                    layer_config['config']['bias_regularizer'] = bias_regularizer
        prev_model = keras_model
        keras_model = keras.models.Model.from_config(model_config)
        keras_model.set_weights(prev_model.get_weights())

        return keras_model

    def _build_distributed(self):
        """Build the training and validation graph, with Horovod Distributer enabled."""

        # Use Horovod distributor for multi-gpu training.
        self._ngpus = distribution.get_distributor().size()
        # Set random seed for distributed training.
        seed = distribution.get_distributor().distributed_seed(self._random_seed)
        set_random_seed(seed)

        # Must set the correct learning phase, `1` is training mode.
        keras.backend.set_learning_phase(1)
        with tf.name_scope("DataLoader"):
            # Prepare data for training and validation.
            data = self._dataloader()
            # Total training samples and steps per epoch.
            self._samples_per_epoch = self._dataloader.num_samples
            self._steps_per_epoch = \
                self._samples_per_epoch // (self._dataloader.batch_size * self._ngpus)
            self._last_step = self._num_epoch * self._steps_per_epoch

        with tf.name_scope("Model"):
            if self._load_graph:
                logger.info(("Loading pretrained model graph as is from {}...").format(
                            self._pretrained_weights))
                # Load the model
                loaded_model = load_model(self._pretrained_weights, self._key)
                logger.warning("Ignoring regularization factors for pruning exp..!")
                # WAR is to define an input layer explicitly with data.images as
                # tensor. This resolves the input type/shape mismatch error.
                # But this creates a submodel within the model. And currently,
                # there are two input layers.
                # TODO: See if the layers can be expanded or better solution.
                input_layer = keras.layers.Input(
                    tensor=data.images,
                    shape=(None, None, 3),
                    name='input_1')
                # TODO: Enable once tested.
                # loaded_model = self.update_regularizers(
                #     loaded_model, self._model._kernel_regularizer, self._model._bias_regularizer
                # )
                loaded_model = self.update_regularizers(loaded_model)
                predictions = loaded_model(input_layer)
                self._model._keras_model = keras.models.Model(
                    inputs=input_layer, outputs=predictions)
            else:
                logger.info("Building model graph from model defintion ...")
                predictions = self._model(inputs=data.images)
            # Print out model summary.
            # print_model_summary(self._model._keras_model) # Disable for TLT

            if self._check_if_first_run(
            ) and not self._finetuning_config["is_finetune_exp"]:
                logger.info("First run ...")
                # Initialize model with pre-trained weights
                if self._pretrained_weights is not None and not self._load_graph:
                    logger.info(
                        ("Intializing model with pre-trained weights {}...").format(
                            self._pretrained_weights))
                    load_pretrained_weights(
                        self._model._keras_model,
                        self._pretrained_weights,
                        key=self._key,
                        logger=None)
            elif self._finetuning_config["is_finetune_exp"]:
                logger.info(
                    ("Finetuning started -> Loading from {} checkpoint...").format(
                        self._finetuning_config["checkpoint_path"]))
                # NOTE: The last step here might be different because of the difference in
                # dataset sizes - steps_per_epoch might be small for a smaller
                # dataset
                current_step = get_step_from_ckzip(self._finetuning_config["checkpoint_path"])
                if "epoch" in self._finetuning_config["checkpoint_path"]:
                    current_step *= self._steps_per_epoch

                self._last_step = current_step + (
                    self._num_epoch - self._finetuning_config["ckpt_epoch_num"]
                ) * self._steps_per_epoch
                logger.info("Updated last_step: {}".format(self._last_step))
            else:
                logger.info(
                    "Not first run and not finetuning experiment -> \
                        Loading from latest checkpoint...")

        if self.use_stagewise_lr_multipliers:
            lr_mult = self._model.get_lr_multipiers()
        else:
            lr_mult = {}

        with tf.name_scope("Loss"):
            label_slice_indices = self._dataloader.pose_config.label_slice_indices
            self._losses = self._loss(data.labels,
                                      predictions,
                                      data.masks,
                                      self.output_seq,
                                      label_slice_indices)

            self._model_loss = self._model.regularization_losses()
            self._total_loss = tf.reduce_sum(
                self._losses) / self._dataloader.batch_size + self._model_loss

            tf.summary.scalar(name='total_loss', tensor=self._total_loss)

        with tf.name_scope("Optimizer"):
            # Update decay steps
            _learning_rate_scheduler_type = type(self._optimizer._learning_rate_schedule).__name__
            if 'SoftstartAnnealingLearningRateSchedule' in _learning_rate_scheduler_type:
                self._optimizer._learning_rate_schedule.last_step = self._last_step
            elif 'BpNetExponentialDecayLRSchedule' in _learning_rate_scheduler_type:
                self._optimizer._learning_rate_schedule.update_decay_steps(
                    self._steps_per_epoch)
            self._optimizer.build()
            self._optimizer.set_grad_weights_dict(lr_mult)

            self._train_op = self._optimizer.minimize(
                loss=self._total_loss,
                global_step=tf.compat.v1.train.get_global_step())[0]

    def build(self):
        """Build the training and validation graph."""

        self._build_distributed()

    def train(self):
        """Run training."""
        is_master = distribution.get_distributor().is_master()
        if not is_master:
            checkpoint_dir = None
            checkpoint_path = None
        else:
            checkpoint_dir = self._checkpoint_dir
            checkpoint_path = self._finetuning_config["checkpoint_path"]

            mkdir_p(checkpoint_dir)

        # TODO: tensorboard visualization of sample outputs at each stage
        # TODO: CSV Logger like in Keras for epoch wise loss summary
        # TODO: Add more log_tensors: stagewise_loss etc.
        log_tensors = {
            'step': tf.compat.v1.train.get_global_step(),
            'loss': self._total_loss,
            'epoch': tf.compat.v1.train.get_global_step() / self._steps_per_epoch}

        serialization_listener = EpochModelSerializationListener(
            checkpoint_dir=checkpoint_dir,
            model=self._model,
            key=self._key,
            steps_per_epoch=self._steps_per_epoch,
            max_to_keep=None
        )
        listeners = [serialization_listener]

        common_hooks = get_common_training_hooks(
            log_tensors=log_tensors,
            log_every_n_secs=self._log_every_n_secs,
            checkpoint_n_steps=self._checkpoint_n_epoch *
            self._steps_per_epoch,
            model=None,
            last_step=self._last_step,
            checkpoint_dir=checkpoint_dir,
            scaffold=self.scaffold,
            steps_per_epoch=self._steps_per_epoch,
            summary_every_n_steps=self._summary_every_n_steps,
            infrequent_summary_every_n_steps=0,
            listeners=listeners,
            key=self._key,
        )

        # Add hook to stop training if the loss becomes nan
        self._hooks = self._hooks + [tf.train.NanTensorHook(
           self._total_loss, fail_on_nan_loss=True
        )]

        if is_master:
            self._hooks.append(TaskProgressMonitorHook(log_tensors,
                                                       checkpoint_dir,
                                                       self._num_epoch,
                                                       self._steps_per_epoch))

        hooks = common_hooks + self._hooks
        # If specific checkpoint path provided, then pick up the params from that
        # Otherwise, use the latest checkpoint from the checkpoint dir
        if self._finetuning_config["is_finetune_exp"]:
            latest_step = get_step_from_ckzip(checkpoint_path)
            if "epoch" in checkpoint_path:
                latest_step *= self._steps_per_epoch
            checkpoint_filename = get_tf_ckpt(checkpoint_path, self._key, latest_step)
        else:
            checkpoint_filename = get_latest_checkpoint(checkpoint_dir, self._key)

        self.run_training_loop(
            train_op=self._train_op,
            hooks=hooks,
            checkpoint_filename_with_path=checkpoint_filename
        )

        # Once training is completed, copy the lastest model to weights directory
        if is_master:
            latest_tlt_model_path = get_latest_tlt_model(checkpoint_dir, extension=MODEL_EXTENSION)
            if latest_tlt_model_path and os.path.exists(latest_tlt_model_path):
                final_model_path = os.path.join(checkpoint_dir, self.final_model_name + MODEL_EXTENSION)
                logger.info("Saving the final step model to {}".format(final_model_path))
                shutil.copyfile(latest_tlt_model_path, final_model_path)
