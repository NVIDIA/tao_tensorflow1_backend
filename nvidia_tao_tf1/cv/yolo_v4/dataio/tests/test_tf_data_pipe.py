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
"""test YOLOv4 tf.data dataloader."""

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

from nvidia_tao_tf1.cv.yolo_v3.proto.dataset_config_pb2 import (
    YOLOv3DatasetConfig,
    YOLOv3DataSource,
)
from nvidia_tao_tf1.cv.yolo_v3.utils.tensor_utils import get_init_ops
from nvidia_tao_tf1.cv.yolo_v4.dataio.tf_data_pipe import YOLOv4TFDataPipe
from nvidia_tao_tf1.cv.yolo_v4.proto.augmentation_config_pb2 import AugmentationConfig
from nvidia_tao_tf1.cv.yolo_v4.proto.experiment_pb2 import Experiment


def generate_dummy_labels(path, num_samples, height=0, width=0, labels=None):
    """Generate num_samples dummy labels and store them to a tfrecords file in the given path.

    Args:
        path: Path to the generated tfrecords file.
        num_samples: Labels will be generated for this many images.
        height, width: Optional image shape.
        labels: Optional, list of custom labels to write into the tfrecords file. The user is
            expected to provide a label for each sample. Each label is dictionary with the label
            name as the key and value as the corresponding tf.train.Feature.
    """
    if labels is None:
        labels = [{'target/object_class': _bytes_feature('bicycle'),
                   'target/coordinates_x1': _float_feature(1.0),
                   'target/coordinates_y1': _float_feature(45.0),
                   'target/coordinates_x2': _float_feature(493.0),
                   'target/coordinates_y2': _float_feature(372.0),
                   'target/truncation': _float_feature(0.0),
                   'target/occlusion': _int64_feature(0),
                   'target/front': _float_feature(0.0),
                   'target/back': _float_feature(0.5),
                   'frame/id': _bytes_feature(str(i)),
                   'frame/height': _int64_feature(height),
                   'frame/width': _int64_feature(width)} for i in range(num_samples)]
    else:
        num_custom_labels = len(labels)
        assert num_custom_labels == num_samples, \
            "Expected %d custom labels, got %d." % (num_samples, num_custom_labels)
    writer = tf.python_io.TFRecordWriter(str(path))
    for label in labels:
        features = tf.train.Features(feature=label)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()


@pytest.fixture
def dataset_config():
    """dataset config."""
    source = YOLOv3DataSource()
    curr_dir = os.path.dirname(__file__)
    source.tfrecords_path = os.path.join(curr_dir, "tmp_tfrecord")
    source.image_directory_path = curr_dir
    dataset = YOLOv3DatasetConfig()
    dataset.data_sources.extend([source])
    dataset.target_class_mapping.update({"bicycle": "bicycle"})
    dataset.validation_data_sources.extend([source])
    dataset.image_extension = "jpg"
    return dataset


@pytest.fixture
def augmentation_config():
    """augmentation config."""
    aug_config = AugmentationConfig()
    aug_config.output_width = 320
    aug_config.output_height = 320
    aug_config.output_channel = 3
    return aug_config


@pytest.fixture
def _test_experiment_spec(dataset_config, augmentation_config):
    curr_dir = os.path.dirname(__file__)
    record_path = os.path.join(curr_dir, "tmp_tfrecord")
    img_path = os.path.join(curr_dir, "0.jpg")
    # write dummy tfrecord
    generate_dummy_labels(record_path, 1, height=375, width=500)
    # write dummy image
    img = np.random.randint(low=0, high=255, size=(375, 500, 3), dtype=np.uint8)
    tmp_im = Image.fromarray(img)
    tmp_im.save(img_path)
    # instantiate a dummy spec
    spec = Experiment()
    spec.dataset_config.data_sources.extend(dataset_config.data_sources)
    spec.dataset_config.target_class_mapping.update(dataset_config.target_class_mapping)
    spec.dataset_config.validation_data_sources.extend(dataset_config.validation_data_sources)
    spec.dataset_config.image_extension = dataset_config.image_extension
    spec.augmentation_config.output_width = augmentation_config.output_width
    spec.augmentation_config.output_height = augmentation_config.output_height
    spec.augmentation_config.output_channel = augmentation_config.output_channel
    spec.training_config.batch_size_per_gpu = 1
    spec.eval_config.batch_size = 1
    yield spec
    os.remove(record_path)
    os.remove(img_path)


def test_tf_data_pipe(_test_experiment_spec):
    h_tensor = tf.constant(
        _test_experiment_spec.augmentation_config.output_height,
        dtype=tf.int32
    )
    w_tensor = tf.constant(
        _test_experiment_spec.augmentation_config.output_width,
        dtype=tf.int32
    )
    tf_dataset = YOLOv4TFDataPipe(
        experiment_spec=_test_experiment_spec,
        training=False,
        h_tensor=h_tensor,
        w_tensor=w_tensor
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(0)
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(get_init_ops())
    imgs, tf_labels = K.get_session().run(
        [tf_dataset.images, tf_dataset.gt_labels]
    )
    assert imgs.shape == (1, 3, 320, 320)
    assert tf_labels[0].shape == (1, 6)
