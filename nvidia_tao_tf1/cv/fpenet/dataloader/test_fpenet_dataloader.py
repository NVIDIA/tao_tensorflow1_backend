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
"""Test FpeNet default dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import pytest
import tensorflow as tf
from nvidia_tao_tf1.cv.fpenet.dataloader.fpenet_dataloader import FpeNetDataloader


def get_data_source():
    """
    Returns a data source dict for FpeNetDataloader constructor.
    """
    image_info_dict = {}

    # Set input information for eyes and face input.
    image_info_dict['image'] = {
        'height': 80,
        'width': 80,
        'channel': 1
    }

    # Setup dataset info.
    dataset_info_dict = {
        'use_extra_dataset': False,
        'root_path': '',
        'tfrecords_directory_path': 'nvidia_tao_tf1/cv/fpenet/dataloader',
        'tfrecords_set_id_train': 'testdata',
        'no_occlusion_masking_sets': '',
        'tfrecords_set_id_val': 'testdata',
        'image_extension': 'png',
        'ground_truth_folder_name': '',
        'tfrecord_folder_name': '',
        'tfrecord_file_name': 'data.tfrecords',
    }
    # Setup KPI set info.
    kpiset_info_dict = {
        'tfrecords_set_id_kpi': 'testdata',
    }

    # Setup augmentation info
    augmentation_info_dict = {
        'augmentation_resize_probability': 0.5,
        'augmentation_resize_scale': 1.6,
        'enable_occlusion_augmentation': True,
        'enable_online_augmentation': True,
        'enable_resize_augmentation': True,
        'patch_probability': 0.5,
        'size_to_image_ratio': 0.5,
        'mask_augmentation_patch': True
    }

    # Assemble all information needed together
    data_source = {
        'batch_size': 2,
        'num_keypoints': 80,
        'image_info': image_info_dict,
        'dataset_info': dataset_info_dict,
        'kpiset_info': kpiset_info_dict,
        'augmentation_config': augmentation_info_dict
    }
    return data_source


class FpeNetDataloaderTest(tf.test.TestCase):
    """
    Test FpeNet dataloader returning correct instance image and keypoint pairs.
    """

    @parameterized.expand([
        ("training", 2),
        ("validation", 2),
        ("kpi_testing", 2)])
    def test_tfrecords_loading(self, phase, expected_num_samples):
        """
        Test dataloader on loading multiple instances and augmentation enabled.
        """
        data_source = get_data_source()
        img_shape = (1, 80, 80)
        repeat = True

        with self.test_session() as session:
            dataloader = FpeNetDataloader(batch_size=data_source['batch_size'],
                                          image_info=data_source['image_info'],
                                          dataset_info=data_source['dataset_info'],
                                          kpiset_info=data_source['kpiset_info'],
                                          augmentation_info=data_source['augmentation_config'],
                                          num_keypoints=data_source['num_keypoints'])

            # Get the tensor object from test data
            results = dataloader(repeat, phase)
            images = results[0]
            ground_truth_landmarks = results[1]
            num_samples = results[2]

            iterator_init = tf.get_collection(
                dataloader.ITERATOR_INIT_OP_NAME)
            session.run(iterator_init)

            assert num_samples == expected_num_samples
            assert tf.shape(images).eval().tolist() == \
                [dataloader.batch_size, img_shape[0], img_shape[1], img_shape[2]]
            assert tf.shape(ground_truth_landmarks[0]).eval().tolist() == \
                [dataloader.batch_size, data_source['num_keypoints'], 2]
            assert tf.shape(ground_truth_landmarks[1]).eval().tolist() == \
                [dataloader.batch_size, data_source['num_keypoints']]
