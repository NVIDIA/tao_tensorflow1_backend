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

"""Tests for the COCO converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import inspect
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pytest
import six

import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.dataio.build_converter import build_converter
import nvidia_tao_tf1.cv.detectnet_v2.proto.coco_config_pb2 as coco_config_pb2
import nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_export_config_pb2 as\
    dataset_export_config_pb2


def _dataset_export_config(num_partitions=1, coco_label_path="coco.json"):
    """Return a COCO dataset export configuration with given number of partitions."""
    dataset_export_config = dataset_export_config_pb2.DatasetExportConfig()

    coco_config = coco_config_pb2.COCOConfig()
    root_dir = os.path.dirname(os.path.abspath(
        inspect.getsourcefile(lambda: None)))
    root_dir += "/test_data"
    coco_config.root_directory_path = root_dir
    coco_config.num_partitions = num_partitions
    coco_config.num_shards[:] = [0]
    coco_config.img_dir_names[:] = ["image_2"]
    coco_config.annotation_files[:] = [coco_label_path]

    get_mock_images(coco_config)

    dataset_export_config.coco_config.CopyFrom(coco_config)
    return dataset_export_config


def get_mock_images(coco_config):
    """Generate mock images from the image_ids."""
    image_root = os.path.join(
        coco_config.root_directory_path, coco_config.img_dir_names[0])
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    image_file = {'000000552775': (375, 500),
                  '000000394940': (426, 640),
                  '000000015335': (640, 480)}
    for idx, sizes in six.iteritems(image_file):
        image_file_name = os.path.join(image_root,
                                       '{}{}'.format(idx, ".jpg"))
        image = Image.new("RGB", sizes)
        image.save(image_file_name)
    return image_file


def _mock_open_image(image_file):
    """Mock image open()."""
    # the images are opened to figure out their dimensions so mock the image size
    mock_image = namedtuple('mock_image', ['size'])
    images = {'000000552775': mock_image((375, 500)),
              '000000394940': mock_image((426, 640)),
              '000000015335': mock_image((640, 480))}

    # return the mocked image corresponding to frame_id in the image_file path
    for frame_id in images.keys():
        if frame_id in image_file:
            return images[frame_id]
    return mock_image


def _mock_converter(mocker, tmpdir, coco_label_path):
    """Return a COCO converter with a mocked sequence to frame map."""
    output_filename = os.path.join(str(tmpdir), 'coco_test.tfrecords')

    # Mock image open().
    mocker.patch.object(Image, 'open', _mock_open_image)

    # Convert a few COCO labels to TFrecords.
    dataset_export_config = _dataset_export_config(coco_label_path=coco_label_path)
    converter = build_converter(dataset_export_config, output_filename)

    return converter, output_filename, dataset_export_config


@pytest.mark.parametrize("coco_label_path", ['coco.json'])
@pytest.mark.parametrize("use_dali", [False, True])
def test_tfrecords_roundtrip(mocker, tmpdir, coco_label_path, use_dali):
    """Test converting a few labels to TFRecords and parsing them back to Python."""
    converter, output_filename, dataset_export_config = _mock_converter(mocker,
                                                                        tmpdir,
                                                                        coco_label_path)
    converter.use_dali = use_dali
    converter.convert()

    class2idx = converter.class2idx

    tfrecords = tf.python_io.tf_record_iterator(output_filename + '-fold-000-of-001')

    # Load small coco example annotation
    coco_local_path = os.path.join(dataset_export_config.coco_config.root_directory_path,
                                   dataset_export_config.coco_config.annotation_files[0])

    coco = COCO(coco_local_path)

    frame_ids = ['000000552775', '000000394940', '000000015335']
    expected_frame_ids = [os.path.join("image_2", frame_id) for frame_id in frame_ids]

    # Specific to each test case
    for i, record in enumerate(tfrecords):
        example = tf.train.Example()
        example.ParseFromString(record)

        features = example.features.feature

        # Load expected values from the local JSON COCO annotation
        frame_id = os.path.basename(expected_frame_ids[i])
        expected_img = coco.loadImgs(int(frame_id))[0]
        expected_annIds = coco.getAnnIds(imgIds=[int(frame_id)])
        expected_anns = coco.loadAnns(expected_annIds)

        expected_cats = []
        expected_x1, expected_y1, expected_x2, expected_y2 = [], [], [], []
        for ann in expected_anns:
            cat = coco.loadCats(ann['category_id'])[0]['name']
            if use_dali:
                expected_x1.append(ann['bbox'][0] / expected_img['width'])
                expected_y1.append(ann['bbox'][1] / expected_img['height'])
                expected_x2.append((ann['bbox'][0] + ann['bbox'][2]) / expected_img['width'])
                expected_y2.append((ann['bbox'][1] + ann['bbox'][3]) / expected_img['height'])
                expected_cats.append(class2idx[cat.replace(" ", "").lower()])
            else:
                expected_x1.append(ann['bbox'][0])
                expected_y1.append(ann['bbox'][1])
                expected_x2.append(ann['bbox'][0] + ann['bbox'][2])
                expected_y2.append(ann['bbox'][1] + ann['bbox'][3])
                expected_cats.append(cat.replace(" ", "").encode('utf-8'))

        bbox_values = [expected_x1, expected_y1,
                       expected_x2, expected_y2]

        assert features['frame/id'].bytes_list.value[0] == bytes(expected_frame_ids[i], 'utf-8')
        assert features['frame/width'].int64_list.value[0] == expected_img['width']
        assert features['frame/height'].int64_list.value[0] == expected_img['height']

        if use_dali:
            assert features['target/object_class_id'].float_list.value[:] == expected_cats
        else:
            assert features['target/object_class'].bytes_list.value[:] == expected_cats

        bbox_features = ['target/coordinates_' +
                         x for x in ('x1', 'y1', 'x2', 'y2')]

        for feature, expected_value in zip(bbox_features, bbox_values):
            np.testing.assert_allclose(
                features[feature].float_list.value[:], expected_value)

    return converter
