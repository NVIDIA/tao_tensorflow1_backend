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
"""Test BpNet Trainer."""

from collections import namedtuple
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.core.utils import get_all_simple_values_from_event_file
from nvidia_tao_tf1.cv.bpnet.dataloaders.pose_config import BpNetPoseConfig
from nvidia_tao_tf1.cv.bpnet.dataloaders.processors.augmentation import AugmentationConfig
from nvidia_tao_tf1.cv.bpnet.learning_rate_schedules.exponential_decay_schedule import \
    BpNetExponentialDecayLRSchedule
from nvidia_tao_tf1.cv.bpnet.losses.bpnet_loss import BpNetLoss
from nvidia_tao_tf1.cv.bpnet.models.bpnet_model import BpNetModel
from nvidia_tao_tf1.cv.bpnet.optimizers.weighted_momentum_optimizer import \
    WeightedMomentumOptimizer
from nvidia_tao_tf1.cv.bpnet.trainers.bpnet_trainer import BpNetTrainer

BpData = namedtuple('BpData', ['images', 'masks', 'labels'])


class SyntheticDataloader:
    def __init__(self, batch_size, image_shape, label_shape):
        """init funtion for Synthetic Dataloader

        Args:
            batch_size (int): batch size to use for training
            image_shape (list): HWC ordering
            label_shape (list): HWC ordering
        """
        self.images = tf.convert_to_tensor(np.random.randn(
            batch_size, image_shape[0], image_shape[1], image_shape[2]
        ),
                                          dtype=tf.float32)
        self.masks = tf.convert_to_tensor(np.random.randn(
            batch_size, label_shape[0], label_shape[1], label_shape[2]
        ),
                                         dtype=tf.float32)
        self.labels = tf.convert_to_tensor(np.random.randn(
            batch_size, label_shape[0], label_shape[1], label_shape[2]
        ),
                                          dtype=tf.float32)

        self.num_samples = batch_size
        self.batch_size = batch_size
        self.pose_config = create_pose()

    def __call__(self):
        return BpData(self.images, self.masks, self.labels)


def create_pose():
    """
    Create bpnet pose config object.

    Returns:
        (BpNetPoseConfig)
    """
    pose_config_root = "nvidia_tao_tf1/cv/bpnet/dataloaders"
    pose_config_path = os.path.join(
        pose_config_root,
        "pose_configurations/bpnet_18joints.json"
    )
    pose_config_spec = {
        'target_shape': [32, 32],
        'pose_config_path': pose_config_path
    }

    return BpNetPoseConfig(**pose_config_spec)


def create_augmentation():
    """
    Create bpnet augmentation config object.

    Returns:
        (AugmentationConfig)
    """

    augmentation_config_spec = {'spatial_aug_params': {
        'flip_lr_prob': 0.5,
        'rotate_deg_max': 40.0,
        'rotate_deg_min': 15.0,
        'zoom_prob': 0.0,
        'zoom_ratio_min': 0.5,
        'zoom_ratio_max': 1.1,
        'translate_max_x': 40.0,
        'translate_min_x': 10,
        'translate_max_y': 40.0,
        'translate_min_y': 10,
        'target_person_scale': 0.7},
        'identity_spatial_aug_params': None,
        'spatial_augmentation_mode': 'person_centric'
    }

    return AugmentationConfig(**augmentation_config_spec)


def create_optimizer():
    """
    Create bpnet weighted momentum optimizer object.

    Returns:
        (WeightedMomentumOptimizer)
    """

    learning_rate_spec = {
        'learning_rate': 2e-5,
        'decay_epochs': 17,
        'decay_rate': 0.333,
        'min_learning_rate': 8.18938e-08
    }

    lr_scheduler = BpNetExponentialDecayLRSchedule(**learning_rate_spec)

    optimizer_spec = {
        'learning_rate_schedule': lr_scheduler,
        'grad_weights_dict': None,
        'weight_default_value': 1.0,
        'momentum': 0.9,
        'use_nesterov': False
    }

    optimizer = WeightedMomentumOptimizer(**optimizer_spec)

    return optimizer


def create_model():
    """
    Create bpnet model object.

    Returns:
        (BpNetModel)
    """

    backbone_attr = {
        'architecture': 'vgg',
        'mtype': 'default',
        'use_bias': False
    }
    model_spec = {
        'backbone_attributes': backbone_attr,
        'stages': 3,
        'heat_channels': 19,
        'paf_channels': 38,
        'use_self_attention': False,
        'data_format': 'channels_last',
        'use_bias': True,
        'regularization_type': 'l2',
        'kernel_regularization_factor': 5e-4,
        'bias_regularization_factor': 0.0
    }

    return BpNetModel(**model_spec)


def create_trainer(checkpoint_dir):
    """
    Create trainer object.

    Args:
        checkpoint_dir (str): folder path for model.
    """

    optimizer = create_optimizer()
    model = create_model()
    dataloader = SyntheticDataloader(2, [256, 256, 3], [32, 32, 57])
    loss = BpNetLoss()
    inference_spec = "nvidia_tao_tf1/cv/bpnet/experiment_specs/infer_default.yaml"

    trainer_specs = {
        'checkpoint_dir': checkpoint_dir,
        'optimizer': optimizer,
        'model': model,
        'dataloader': dataloader,
        'loss': loss,
        'key': '0',
        "inference_spec": inference_spec,
        "num_epoch": 5
    }

    trainer = BpNetTrainer(**trainer_specs)

    return trainer


def test_trainer_train(tmpdir):
    """Test whether trainer trains correctly."""

    trainer = create_trainer(str(tmpdir))
    trainer.build()
    trainer.train()

    train_op = trainer.train_op

    assert train_op is not None
    assert isinstance(train_op, tf.Operation)

    tensorboard_log_dir = os.path.join(str(tmpdir), "events")
    assert os.path.isdir(tensorboard_log_dir), (
        f"Tensorboard log directory not found at {tensorboard_log_dir}"
    )

    values_dict = get_all_simple_values_from_event_file(tensorboard_log_dir)

    loss_key = 'Loss/total_loss'
    assert loss_key in values_dict.keys()

    # Get loss values as a list for all steps.
    loss_values = [loss_tuple[1] for loss_tuple in values_dict[loss_key].items()]

    # Form a list to determine whether loss has decreased across each step.
    is_loss_reduced = [loss_values[i] >= loss_values[i+1]
                       for i in range(len(loss_values)-1)]

    loss_reduced_percentage = sum(is_loss_reduced) / len(is_loss_reduced)

    assert loss_reduced_percentage >= 0.5
