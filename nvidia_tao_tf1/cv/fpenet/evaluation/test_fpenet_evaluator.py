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
"""Tests for FpeNet Evaluator Class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras.backend as K
import tensorflow as tf

from nvidia_tao_tf1.cv.fpenet.evaluation.fpenet_evaluator import FpeNetEvaluator
from nvidia_tao_tf1.cv.fpenet.losses.fpenet_loss import FpeLoss
from nvidia_tao_tf1.cv.fpenet.models.fpenet_basemodel import FpeNetBaseModel


class _synthetic_dataloader():
    """Create synthetic dataloader for test."""

    def __init__(self, phase='validation'):
        self.phase = phase
        self.batch_size = 4
        self.image_width = 80
        self.images = tf.fill((4, 1, 80, 80), 255.0)

    def __call__(self, repeat=True, phase='validation'):
        '''
        Synthetic dataloader call.

        Args:
            repeat (bool): Whether the dataset can be looped over multiple times or only once.
            phase (str): Evaluation phase. Options- 'validation' or 'kpi_testing'.
        Returns:
            images (Tensor): Image tensors.
            label (Tensor): Ground truth keypoints tensor.
            num_samples (int): Number of samples.
            masking_occ_info (Tensor): Keypoints masking info.
        '''

        images = self.images
        label = (tf.zeros([4, 80, 2], dtype='float32'), tf.zeros([4, 80], dtype='float32'))
        masking_occ_info = tf.zeros([4], dtype='float32')
        face_bbox = tf.stack([tf.zeros([4, 4], dtype='float32')])
        image_names = tf.stack(['tmp'])
        num_samples = 4

        if phase == 'kpi_testing':
            return images, label, num_samples, masking_occ_info, face_bbox, image_names
        return images, label, num_samples, masking_occ_info


def _create_evaluator(phase, checkpoint_dir):
    """
    Create evaluator object.

    Args:
        phase (str): Evaluation phase. Options- 'validation' or 'kpi_testing'.
        checkpoint_dir (str): Checkpoint directory path with model.
    Returns:
        evaluator (FpeNetEvaluator): Instance of FpeNetEvaluator to evaluate with.
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
    model.build(input_images=dataloader.images)

    # save temporary files for test case purpose since doing only eval
    model.save_model(os.path.join(checkpoint_dir, 'model.epoch-1.hdf5'), 'test')
    open(os.path.join(checkpoint_dir, 'checkpoint'), 'a').close()

    loss = FpeLoss('l1')

    evaluator = FpeNetEvaluator(model=model,
                                dataloader=dataloader,
                                save_dir=checkpoint_dir,
                                mode='validation',
                                visualizer=None,
                                enable_viz=False,
                                num_keypoints=80,
                                loss=loss,
                                steps_per_epoch=1,
                                model_path=checkpoint_dir)

    return evaluator


def test_evaluator_validation(tmpdir):
    """
    Test whether evaluator instantiates correctly.

    Args:
        tmpdir (str): Temporary path for checkpoint directory.
    Returns:
        None
    """
    # Test: 'validation' phase
    K.clear_session()
    evaluator = _create_evaluator('validation', str(tmpdir))
    evaluator.build()
    evaluation_cost = evaluator.evaluate(global_step=1)
    assert isinstance(evaluation_cost, float)


def test_evaluator_kpi(tmpdir):
    """
    Test whether evaluator instantiates correctly.

    Args:
        tmpdir (str): Temporary path for checkpoint directory.
    Returns:
        None
    """
    # Test: 'kpi_testing' phase
    K.clear_session()
    evaluator = _create_evaluator('kpi_testing', str(tmpdir))
    evaluator.build()
    evaluation_cost = evaluator.evaluate(global_step=1)
    assert isinstance(evaluation_cost, float)
