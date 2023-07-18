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

"""Tests for BpNet dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.graph import get_init_ops
from nvidia_tao_tf1.cv.bpnet.dataloaders.bpnet_dataloader import BpData
from nvidia_tao_tf1.cv.bpnet.dataloaders.bpnet_dataloader import BpNetDataloader
from nvidia_tao_tf1.cv.bpnet.dataloaders.pose_config import BpNetPoseConfig
from nvidia_tao_tf1.cv.bpnet.dataloaders.processors.augmentation import AugmentationConfig

TEST_DATA_ROOT_PATH = os.getenv("CI_DATA_DIR", "/media/scratch.metropolis2/tao_ci/tao_tf1/data/bpnet")


def build_augmentation_config(augmentation_mode='person_centric', augmentation_dict=None):
    """Initialize and return object of type AugmentationConfig."""

    if augmentation_dict is None:
        augmentation_dict = {
            'spatial_aug_params': {
                'flip_lr_prob': 0.5,
                'flip_tb_prob': 0.0,
                'rotate_deg_max': 40.0,
                'rotate_deg_min': None,
                'zoom_prob': 0.0,
                'zoom_ratio_min': 0.5,
                'zoom_ratio_max': 1.1,
                'translate_max_x': 40.0,
                'translate_min_x': None,
                'translate_max_y': 40.0,
                'translate_min_y': None,
                'use_translate_ratio': False,
                'translate_ratio_max': 0.2,
                'translate_ratio_min': 0.2,
                'target_person_scale': 0.6
            },
            'spatial_augmentation_mode': augmentation_mode
        }

    augmentation_config = AugmentationConfig(**augmentation_dict)
    return augmentation_config


def build_pose_config(pose_config_path, target_shape=(46, 46)):
    """Initialize and return object of type BpNetPoseConfig."""

    pose_config = BpNetPoseConfig(target_shape, pose_config_path)
    return pose_config


def build_image_config(image_shape=None, image_encoding='jpg'):
    """Initialize and return dict with image related parameters."""

    if image_shape is None:
        image_dims = {
            "channels": 3,
            "height": 368,
            "width": 368
        }
    else:
        image_dims = {
            "channels": image_shape[2],
            "height": image_shape[0],
            "width": image_shape[1]
        }

    image_config = {
        'image_dims': image_dims,
        'image_encoding': image_encoding
    }
    return image_config


def build_normalization_config():
    """Initialize and return dict with normalization related parameters."""

    normalization_config = {
        'image_scale': [256.0, 256.0, 256.0],
        'image_offset': [0.5, 0.5, 0.5],
        'mask_scale': [255.0],
        'mask_offset': [0.0]
    }
    return normalization_config


def build_dataset_config(
    train_records_path=None,
    val_records_folder_path=None,
    val_records_path=None
):
    """Initialize and return dict with dataset related parameters."""

    root_data_path = os.path.join(TEST_DATA_ROOT_PATH, 'test_data/')
    train_records_folder_path = os.path.join(TEST_DATA_ROOT_PATH, 'test_data/')

    if train_records_path is None:
        train_records_path = ['coco/coco_sample.tfrecords']

    dataset_config = {
        'root_data_path': root_data_path,
        'train_records_folder_path': train_records_folder_path,
        'train_records_path': train_records_path,
        'val_records_folder_path': val_records_folder_path,
        'val_records_path': val_records_path,
        'dataset_specs': {
            'coco': 'nvidia_tao_tf1/cv/bpnet/dataio/dataset_specs/coco_spec.json'
        }
    }
    return dataset_config


def build_label_processor_config():
    """Initialize and return dict with label processor related parameters."""

    label_processor_config = {
        'paf_gaussian_sigma': 0.03,
        'heatmap_gaussian_sigma': 0.15,
        'paf_ortho_dist_thresh': 1.0
    }
    return label_processor_config


def build_dataloader(
    batch_size=3,
    image_shape=None,
    target_shape=None,
    train_records_path=None,
    pose_config_path=None,
    normalization_params=None,
    augmentation_mode=None
):
    """Initialize and return object of type BpNetDataloader."""

    # Set default values
    if image_shape is None:
        image_shape = [368, 368, 3]
    if target_shape is None:
        target_shape = [46, 46]
    if train_records_path is None:
        train_records_path = ['coco/sample.tfrecords']
    if pose_config_path is None:
        pose_config_path = \
            'nvidia_tao_tf1/cv/bpnet/dataloaders/pose_configurations/bpnet_18joints.json'

    # Build BpNetPoseConfig
    pose_config = build_pose_config(
        pose_config_path=pose_config_path,
        target_shape=target_shape
    )

    # Build image config
    image_config = build_image_config(image_shape)

    # Build dataset config
    dataset_config = build_dataset_config(train_records_path)

    # Build augmentation config with default params
    augmentation_config = build_augmentation_config(augmentation_mode=augmentation_mode)

    # Build label processor config with default params
    label_processor_config = build_label_processor_config()

    # Build normalization params
    normalization_params = build_normalization_config()

    dataloader = BpNetDataloader(
        batch_size=batch_size,
        pose_config=pose_config,
        image_config=image_config,
        dataset_config=dataset_config,
        augmentation_config=augmentation_config,
        label_processor_config=label_processor_config,
        normalization_params=normalization_params
    )

    return dataloader, pose_config


def test_dataloader_return_type():
    """Test for correct type."""

    dataloader, _ = build_dataloader()
    fetches = dataloader()
    sess = tf.compat.v1.Session()
    sess.run(get_init_ops())
    example = sess.run(fetches)

    assert type(example) == BpData


@pytest.mark.parametrize(
    "pose_config_path",
    ['nvidia_tao_tf1/cv/bpnet/dataloaders/pose_configurations/bpnet_18joints.json']
)
@pytest.mark.parametrize("image_shape", [[368, 368, 3], [256, 256, 3], [224, 320, 3]])
@pytest.mark.parametrize("target_shape", [[46, 46], [32, 32], [28, 40]])
@pytest.mark.parametrize(
    "augmentation_mode", ['person_centric', 'standard', 'standard_with_fixed_aspect_ratio'])
def test_dataloader_shapes(pose_config_path, image_shape, target_shape, augmentation_mode):
    """Test for correct shape of dataloder objects."""

    batch_size = 2

    # Check if the dataloader should throw an exception for the given parameters
    exception_expected = (image_shape[0] // target_shape[0]) != (image_shape[1] // target_shape[1])

    # If an exception is expected, use `pytest.raises(Exception)` to assert
    if exception_expected:
        with pytest.raises(Exception):
            dataloader, pose_config = build_dataloader(
                batch_size=batch_size,
                image_shape=image_shape,
                target_shape=target_shape,
                pose_config_path=pose_config_path,
                augmentation_mode=augmentation_mode
            )
    else:
        dataloader, pose_config = build_dataloader(
            batch_size=batch_size,
            image_shape=image_shape,
            target_shape=target_shape,
            pose_config_path=pose_config_path,
            augmentation_mode=augmentation_mode
        )

        fetches = dataloader()
        sess = tf.compat.v1.Session()
        sess.run(get_init_ops())
        example = sess.run(fetches)

        # Assert that the shapes of all tensors are as expected.
        label_tensor_shape = pose_config.label_tensor_shape
        assert np.shape(example.images) == \
            (batch_size, image_shape[0], image_shape[1], image_shape[2])
        assert np.shape(example.masks) == \
            (batch_size, label_tensor_shape[0], label_tensor_shape[1], label_tensor_shape[2])
        assert np.shape(example.labels) == \
            (batch_size, label_tensor_shape[0], label_tensor_shape[1], label_tensor_shape[2])

        # Assert that the images and mask values are within the range after normalization
        assert np.equal(np.sum(example.images > 0.500001), 0)
        assert np.equal(np.sum(example.images < -0.500001), 0)
        assert np.equal(np.sum(example.masks > 1.000001), 0)
        assert np.equal(np.sum(example.masks < -0.00001), 0)
