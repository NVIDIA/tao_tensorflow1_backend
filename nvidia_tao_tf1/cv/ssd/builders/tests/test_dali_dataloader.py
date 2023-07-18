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
"""test ssd dali dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from keras import backend as K
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _bytes_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _float_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _int64_feature

from nvidia_tao_tf1.cv.ssd.builders.dalipipeline_builder import SSDDALIDataset
from nvidia_tao_tf1.cv.ssd.scripts.dataset_convert import create_tfrecord_idx
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import load_experiment_spec


def create_sample_example():
    example = tf.train.Example(features=tf.train.Features(feature={
        'frame/id': _bytes_feature("001.jpg".encode('utf-8')),
        'frame/height': _int64_feature(375),
        'frame/width': _int64_feature(500),
    }))
    img = np.random.randint(low=0, high=255, size=(375, 500, 3), dtype=np.uint8)
    tmp_im = Image.fromarray(img)
    tmp_im.save("fake_img.jpg")
    with open("fake_img.jpg", "rb") as f:
        image_string = f.read()
    f = example.features.feature
    f['frame/encoded'].MergeFrom(_bytes_feature(image_string))
    os.remove("fake_img.jpg")
    truncation = [0]
    occlusion = [0]
    observation_angle = [0]
    coordinates_x1 = [0.1]
    coordinates_y1 = [0.1]
    coordinates_x2 = [0.8]
    coordinates_y2 = [0.8]
    world_bbox_h = [0]
    world_bbox_w = [0]
    world_bbox_l = [0]
    world_bbox_x = [0]
    world_bbox_y = [0]
    world_bbox_z = [0]
    world_bbox_rot_y = [0]
    object_class_ids = [0]

    f['target/object_class_id'].MergeFrom(_float_feature(*object_class_ids))
    f['target/truncation'].MergeFrom(_float_feature(*truncation))
    f['target/occlusion'].MergeFrom(_int64_feature(*occlusion))
    f['target/observation_angle'].MergeFrom(_float_feature(*observation_angle))
    f['target/coordinates_x1'].MergeFrom(_float_feature(*coordinates_x1))
    f['target/coordinates_y1'].MergeFrom(_float_feature(*coordinates_y1))
    f['target/coordinates_x2'].MergeFrom(_float_feature(*coordinates_x2))
    f['target/coordinates_y2'].MergeFrom(_float_feature(*coordinates_y2))
    f['target/world_bbox_h'].MergeFrom(_float_feature(*world_bbox_h))
    f['target/world_bbox_w'].MergeFrom(_float_feature(*world_bbox_w))
    f['target/world_bbox_l'].MergeFrom(_float_feature(*world_bbox_l))
    f['target/world_bbox_x'].MergeFrom(_float_feature(*world_bbox_x))
    f['target/world_bbox_y'].MergeFrom(_float_feature(*world_bbox_y))
    f['target/world_bbox_z'].MergeFrom(_float_feature(*world_bbox_z))
    f['target/world_bbox_rot_y'].MergeFrom(_float_feature(*world_bbox_rot_y))

    return example.SerializeToString()


@pytest.fixture
def _test_experiment_spec():
    serialized_example = create_sample_example()
    experiment_spec, _ = load_experiment_spec()
    record_path = "tmp_tfrecord"
    idx_path = "idx-tmp_tfrecord"
    with tf.io.TFRecordWriter(record_path) as writer:
        writer.write(serialized_example)
    create_tfrecord_idx(record_path, idx_path)
    experiment_spec.dataset_config.data_sources[0].tfrecords_path = record_path
    experiment_spec.training_config.batch_size_per_gpu = 1
    yield experiment_spec
    os.remove(record_path)
    os.remove(idx_path)


def test_dali_dataset(_test_experiment_spec):
    dali_dataset = SSDDALIDataset(experiment_spec=_test_experiment_spec,
                                  device_id=0,
                                  shard_id=0,
                                  num_shards=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(0)
    sess = tf.Session(config=config)
    K.set_session(sess)
    imgs, tf_labels = K.get_session().run([dali_dataset.images, dali_dataset.labels])
    assert imgs.shape == (1, 3, 300, 300)
    assert tf_labels[0].shape == (1, 5)
