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

"""Tests for FPENet Trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K

from nvidia_tao_tf1.blocks import learning_rate_schedules
from nvidia_tao_tf1.blocks import optimizers

import tensorflow as tf

from nvidia_tao_tf1.core.utils import get_all_simple_values_from_event_file
from nvidia_tao_tf1.cv.fpenet.losses.fpenet_loss import FpeLoss
from nvidia_tao_tf1.cv.fpenet.models.fpenet_basemodel import FpeNetBaseModel
from nvidia_tao_tf1.cv.fpenet.trainers.fpenet_trainer import FpeNetTrainer


class _synthetic_dataloader():
    """Create synthetic dataloader for test."""

    def __init__(self, phase='training'):
        self.phase = phase
        self.batch_size = 4
        self.image_width = 80
        self.image_height = 80
        self.images = tf.fill((4, 1, 80, 80), 255.0)

    def __call__(self, repeat=True, phase='validation'):

        images = self.images
        label = (tf.zeros([4, 80, 2], dtype='float32'), tf.zeros([4, 80], dtype='float32'))
        masking_occ_info = tf.zeros([4], dtype='float32')
        num_samples = 4

        return images, label, num_samples, masking_occ_info


def _create_trainer(phase, checkpoint_dir):
    """
    Create trainer object.

    Args:
        phase (str): phase for dataloader- 'training' or 'validation'
        checkpoint_dir (str): folder path for model.
    """

    dataloader = _synthetic_dataloader(phase=phase)

    model_parameters = {
        'beta': 0.01,
        'dropout_rate': 0.5,
        'freeze_Convlayer': None,
        'pretrained_model_path': None,
        'regularizer_type': 'l2',
        'regularizer_weight': 1.0e-05,
        'type': 'FpeNet_base',
        'use_upsampling_layer': False,
        'visualization_parameters': None,
    }

    model = FpeNetBaseModel(model_parameters)

    learning_rate_schedule = learning_rate_schedules.SoftstartAnnealingLearningRateSchedule(
            annealing=0.5,
            base_learning_rate=0.0005,
            min_learning_rate=1.0e-07,
            soft_start=0.3
            )

    optimizer = optimizers.AdamOptimizer(
            learning_rate_schedule=learning_rate_schedule,
            )

    elt_loss_info = {
        'elt_alpha': 0.5,
        'enable_elt_loss': True,
        'modulus_spatial_augmentation': {}}
    loss = FpeLoss('l1', elt_loss_info=elt_loss_info)

    trainer = FpeNetTrainer(
            dataloader=dataloader,
            model=model,
            optimizer=optimizer,
            loss=loss,
            checkpoint_dir=checkpoint_dir,
            random_seed=42,
            log_every_n_secs=5,
            checkpoint_n_epoch=1,
            num_epoch=10,
            infrequent_summary_every_n_steps=0,
            enable_visualization=False,
            visualize_num_images=3,
            num_keypoints=80,
            key="0"
    )

    return trainer


def test_trainer_train(tmpdir):
    """Test whether trainer trains correctly."""
    K.clear_session()
    trainer = _create_trainer('training', str(tmpdir))
    trainer.build()
    trainer.train()

    # Test on trainable weights. Need to update if freezing a part of model.
    assert len(trainer._model.keras_model.trainable_weights) == 38
    tensorboard_log_dir = os.path.join(str(tmpdir), "events")
    assert os.path.isdir(tensorboard_log_dir), (
        f"Tensorboard log directory not found at {tensorboard_log_dir}"
    )

    values_dict = get_all_simple_values_from_event_file(tensorboard_log_dir)

    loss_key = 'l1_net_loss'
    assert loss_key in values_dict.keys()

    # Get loss values as a list for all steps.
    loss_values = [loss_tuple[1] for loss_tuple in values_dict[loss_key].items()]

    # Form a list to determine whether loss has decreased across each step.
    is_loss_reduced = [loss_values[i] >= loss_values[i+1]
                       for i in range(len(loss_values)-1)]

    loss_reduced_percentage = sum(is_loss_reduced) / len(is_loss_reduced)

    assert loss_reduced_percentage >= 0.5


def test_trainer_evaluator(tmpdir):
    """Test whether trainer passes variables to evaluator correctly."""
    K.clear_session()
    trainer = _create_trainer('vaidation', str(tmpdir))
    trainer.build()

    evaluator = trainer._evaluator

    # Assert that instance variables of evaluator match with values in trainer spec.
    assert evaluator.save_dir == tmpdir
    assert evaluator.mode == "validation"
