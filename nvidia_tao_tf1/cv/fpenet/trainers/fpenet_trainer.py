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
"""FpeNet Trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import logging
import os

from keras import backend as K

import tensorflow as tf

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.blocks.trainer import Trainer
from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.core.hooks.sample_counter_hook import SampleCounterHook
from nvidia_tao_tf1.core.utils import set_random_seed
from nvidia_tao_tf1.cv.common.utilities.serialization_listener import (
    EpochModelSerializationListener
)
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import get_tf_ckpt
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.task_progress_monitor_hook import (
    TaskProgressMonitorHook
)
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.utils import get_common_training_hooks
from nvidia_tao_tf1.cv.fpenet.evaluation.fpenet_evaluator import FpeNetEvaluator
from nvidia_tao_tf1.cv.fpenet.losses.fpenet_loss import FpeNetEltLoss
from nvidia_tao_tf1.cv.fpenet.visualization import FpeNetVisualizer

logger = logging.getLogger(__name__)

MODEL_EXTENSION=".hdf5"


class FpeNetTrainer(Trainer):
    """FpeNet Trainer object that builds the training graph and execution."""

    @tao_core.coreobject.save_args
    def __init__(self,
                 checkpoint_dir=None,
                 random_seed=42,
                 log_every_n_secs=60,
                 checkpoint_n_epoch=1,
                 num_epoch=40,
                 infrequent_summary_every_n_steps=0,
                 enable_visualization=True,
                 visualize_num_images=3,
                 num_keypoints=80,
                 key=None,
                 log_resources=False,
                 **kwargs):
        """__init__ method.

        Args:
            checkpoint_dir (str): Path to directory containing checkpoints.
            random_seed (int): Set random seed.
            log_every_n_secs (int): Log every n secs.
            checkpoint_n_epoch (int): Checkpoint every n epochs.
            num_epoch (int): Number of training epochs.
            infrequent_summary_every_n_epoch (int): Infrequent summary every n epoch.
            enable_visualization (bool): Toggle to enable visualization.
            visualize_num_images (int): Number of data images to show on Tensorboard.
            num_keypoints (int): Number of facial keypoints.
            key (str): Key to decode the model.
            log_resources (bool): Toggle to log GPU usage resources in tensorboard.
        """
        super(FpeNetTrainer, self).__init__(**kwargs)

        self._random_seed = random_seed
        self._checkpoint_dir = checkpoint_dir
        self._log_every_n_secs = log_every_n_secs
        self._checkpoint_n_epoch = checkpoint_n_epoch
        self._num_epoch = num_epoch
        self._infrequent_summary_every_n_steps = infrequent_summary_every_n_steps
        self._summary_every_n_steps = self._steps_per_epoch = \
            self._validation_every_n_steps = self._train_op = self._evaluator =\
            self._eyelids_loss = self._mouth_loss = None

        self._total_loss = 0.0
        self._enable_visualization = enable_visualization
        self._visualize_num_images = visualize_num_images
        self._visualizer = FpeNetVisualizer(
            self._checkpoint_dir, num_images=self._visualize_num_images)

        self._num_keypoints = num_keypoints
        self._key = key
        self._log_resources = log_resources

        self._worker_count = distribution.get_distributor().size()
        self._worker_index = distribution.get_distributor().rank()

    def build(
            self, 
            eval_mode='validation', 
            eval_model_path=None):
        """
        Build the training and validation graph.

        Args:
            eval_mode (str): Evaluation mode- 'validation' or 'kpi_testing'
                'validation'- Validation step durng training.
                'kpi_testing'- KPI data testing.
            eval_model_path (str): Path to the model file to be evaluated.
        """

        # Set random seeds.
        seed = distribution.get_distributor().distributed_seed(
            self._random_seed)
        set_random_seed(seed)
        # Set learning phase to 1 for building the train graph.
        K.set_learning_phase(1)

        # Prepare data for training.
        images, ground_truth_labels, num_samples, occ_masking_info = \
            self._dataloader(phase='training')
        # Compute num_samples per gpu.
        num_samples = num_samples // self._worker_count
        self._batch_size = self._dataloader.batch_size
        self._steps_per_epoch = num_samples // self._batch_size
        self._last_step = self._steps_per_epoch * self._num_epoch

        # Visualization of images and data distribution.
        if self._enable_visualization:
            # Add images to Tensorboard.
            self._visualizer.visualize_images(
                images, ground_truth_labels[0], viz_phase='training')

        # Summary and validate at the end of every epoch.
        self._summary_every_n_steps = self._steps_per_epoch / 10

        # Build model.
        predictions = self._model.build(images,
                                        num_keypoints=self._num_keypoints,
                                        enc_key=self._key)

        predictions_coord = K.reshape(predictions['landmarks'],
                                      (self._dataloader.batch_size,
                                       self._num_keypoints, 2))

        # Add images to Tensorboard.
        if self._enable_visualization:
            self._visualizer.visualize_images(images,
                                              predictions_coord,
                                              viz_phase='training_predictions')

        # For freezing parts of the model.
        trainable_weights = self._model.keras_model.trainable_weights

        # Build optimizer.
        if hasattr(self._optimizer, '_learning_rate_schedule') and \
           hasattr(self._optimizer._learning_rate_schedule, '_last_step'):
            self._optimizer._learning_rate_schedule._last_step = self._last_step
        self._optimizer.build()

        # Compute loss.
        self._landmarks_loss, self._mouth_loss, self._eyelids_loss = \
            self._loss(y_true=ground_truth_labels[0],
                       y_pred=predictions_coord,
                       occ_true=ground_truth_labels[1],
                       occ_masking_info=occ_masking_info,
                       num_keypoints=self._num_keypoints)
        self._total_loss += self._landmarks_loss

        # Compute ELT loss.
        if not eval_mode == 'kpi_testing':
            elt_loss = FpeNetEltLoss(self._loss.elt_loss_info,
                                     image_height=self._dataloader.image_height,
                                     image_width=self._dataloader.image_width,
                                     num_keypoints=self._num_keypoints)
            if elt_loss.enable_elt_loss:
                # apply random transform to images and also retrieve transformation matrix
                images_tm, mapMatrix = elt_loss.transform_images(images)
                # make predictions on the transformed images using current model
                predictions_tm = self._model.keras_model(images_tm)
                predictions_tm_coord = K.reshape(predictions_tm[0],
                                                 (self._dataloader.batch_size,
                                                  self._num_keypoints, 2))

                # apply same transformation to predicted/ground truth labels
                ground_truth_labels_tm = elt_loss.transform_points(ground_truth_labels[0],
                                                                   mapMatrix)
                # compute elt loss
                self._elt_loss, _, _ = self._loss(y_true=ground_truth_labels_tm,
                                                  y_pred=predictions_tm_coord,
                                                  occ_true=ground_truth_labels[1],
                                                  occ_masking_info=occ_masking_info,
                                                  num_keypoints=self._num_keypoints,
                                                  loss_name='elt')
                # scale the elt loss term
                self._total_loss += elt_loss.elt_alpha * self._elt_loss

                # Add images to Tensorboard.
                if self._enable_visualization:
                    self._visualizer.visualize_images(images_tm,
                                                      ground_truth_labels_tm,
                                                      viz_phase='training_elt')

        # Create optimizer.
        self._train_op = self._optimizer.minimize(loss=self._total_loss,
                                                  var_list=trainable_weights)
        if eval_model_path is None:
            logger.info(
                "Evaluation model file path wasn't provided. "
                "Getting the latest checkpoint in {checkpoint_dir}".format(
                    checkpoint_dir=self._checkpoint_dir
                )
            )
            eval_model_path = self.get_latest_checkpoint(
                self._checkpoint_dir,
                self._key,
                extension=MODEL_EXTENSION)
            logger.info("Evaluating using the model at {eval_model_path}".format(
                eval_model_path=eval_model_path
            ))

        # Build evaluator.
        self._evaluator = FpeNetEvaluator(
            self._model, self._dataloader, self._checkpoint_dir, eval_mode,
            self._visualizer, self._enable_visualization, self._num_keypoints,
            self._loss, key=self._key, model_path=eval_model_path,
            steps_per_epoch=self._steps_per_epoch
        )
        self._evaluator.build()
        self._validation_every_n_steps = self._steps_per_epoch * self._checkpoint_n_epoch

    @property
    def train_op(self):
        """Return train optimizer of Trainer."""
        return self._train_op

    def get_latest_checkpoint(
            self,
            results_dir,
            key,
            extension=".ckzip"):
        """Get the latest checkpoint path from a given results directory.

        Parses through the directory to look for the latest checkpoint file
        and returns the path to this file.

        Args:
            results_dir (str): Path to the results directory.
            key (str): Key to load .tlt model
            extension (str): Extension of the file to be filtered.

        Returns:
            ckpt_path (str): Path to the latest checkpoint.
        """
        print(f"Checkpoint results dir {results_dir}")
        checkpoint_glob_string = os.path.join(
            results_dir, f"model.epoch-*{extension}"
        )
        trainable_ckpts = [
            int(os.path.basename(item).split('.')[1].split('-')[1])
            for item in glob.glob(checkpoint_glob_string)
        ]
        num_ckpts = len(trainable_ckpts)
        if num_ckpts == 0:
            return None
        latest_step = sorted(trainable_ckpts, reverse=True)[0]
        latest_checkpoint = os.path.join(
            results_dir,
            f"model.epoch-{latest_step}{extension}"
        )
        if extension in [".tlt", ".hdf5"]:
            return latest_checkpoint
        return get_tf_ckpt(latest_checkpoint, key, latest_step)

    def train(self):
        """Run the training."""
        checkpoint_dir = self._checkpoint_dir \
            if distribution.get_distributor().is_master() else None

        log_tensors = {
            'step': tf.train.get_global_step(),
            'loss': self._total_loss,
            'epoch': tf.train.get_global_step() / self._steps_per_epoch,
            'landmarks_loss': self._landmarks_loss
        }
        if self._loss.elt_loss_info['enable_elt_loss']:
            log_tensors['elt_loss'] = self._elt_loss

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
            checkpoint_n_steps=self._checkpoint_n_epoch * self._steps_per_epoch,
            model=None,
            last_step=self._last_step,
            checkpoint_dir=checkpoint_dir,
            scaffold=self.scaffold,
            steps_per_epoch=self._steps_per_epoch,
            summary_every_n_steps=self._summary_every_n_steps,
            infrequent_summary_every_n_steps=self._infrequent_summary_every_n_steps,
            validation_every_n_steps=self._validation_every_n_steps,
            evaluator=self._evaluator,
            listeners=listeners,
            key=self._key
        )

        if self._worker_index == 0:
            self._hooks = [SampleCounterHook(batch_size=self._worker_count * self._batch_size,
                                             name="Train")]
        else:
            self._hooks = []

        # if self._log_resources:
        #     self._hooks = self._hooks + [ResourceHook(checkpoint_dir, write_interval=1)]

        if self._worker_index == 0:
            self._hooks.append(TaskProgressMonitorHook(log_tensors,
                                                       checkpoint_dir,
                                                       self._num_epoch,
                                                       self._steps_per_epoch))

        hooks = self._hooks + common_hooks
        checkpoint_filename = self.get_latest_checkpoint(self._checkpoint_dir, self._key)

        self.run_training_loop(
            train_op=self._train_op,
            hooks=hooks,
            checkpoint_dir=checkpoint_filename)

    def run_testing(self):
        """Run testing on test and KPI data after training is done."""
        self._evaluator.evaluate()

    def run_training_loop(self, train_op, hooks, checkpoint_dir=None):
        """Run the training loop in a tensorflow session.

        Args:
            train_op (tensor): Tensorflow op to be evaluated to take a training step.
            hooks (list of Hooks): List of Tensorflow Hooks to be used as callbacks while running
                training.
            checkpoint_dir (str): for resuming from a checkpoint. If this value is `None` it will
                not restore variables. If it points to a directory, it will find the latest variable
                snapshot and resume from there.
        """
        # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
        # The SingularMonitoredSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        # Notice we are not using the `MonitoredTrainingSession` variant because that automatically
        # adds unwanted hooks if a `checkpoint_dir` is provided: and if we do not provide it,
        # we cannot resume out checkpoint.
        config = distribution.get_distributor().get_config()

        ignore_keras_values = checkpoint_dir is not None
        if hooks is None:
            hooks = []
        if self._model.keras_model is not None:
            # KerasModelHook takes care of initializing model variables.
            hooks.insert(0, tao_core.hooks.KerasModelHook(
                self._model.keras_model,
                ignore_keras_values)
            )

        with tf.compat.v1.train.SingularMonitoredSession(
                                                hooks=hooks,
                                                scaffold=self.scaffold,
                                                config=config,
                                                checkpoint_filename_with_path=checkpoint_dir
        ) as sess:
            try:
                while not sess.should_stop():
                    # Run training ops with the wrapped session.
                    sess.run(train_op)

            except (KeyboardInterrupt, SystemExit):
                logger.info("Training interrupted.")
