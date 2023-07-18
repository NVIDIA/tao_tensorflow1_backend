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

"""Tests for the KITTI converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import inspect
import json
import os
import tempfile

import numpy as np
from PIL import Image
import pytest
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.dataio.build_converter import build_converter
from nvidia_tao_tf1.cv.detectnet_v2.dataio.kitti_converter_lib import KITTIConverter
import nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_export_config_pb2 as\
    dataset_export_config_pb2
import nvidia_tao_tf1.cv.detectnet_v2.proto.kitti_config_pb2 as kitti_config_pb2


def _dataset_export_config(num_partitions=0):
    """Return a KITTI dataset export configuration with given number of partitions."""
    dataset_export_config = dataset_export_config_pb2.DatasetExportConfig()

    kitti_config = kitti_config_pb2.KITTIConfig()
    root_dir = os.path.dirname(os.path.abspath(
        inspect.getsourcefile(lambda: None)))
    root_dir += "/test_data"
    kitti_config.root_directory_path = root_dir
    kitti_config.num_partitions = num_partitions
    kitti_config.num_shards = 0
    kitti_config.partition_mode = "sequence"
    kitti_config.image_dir_name = "image_2"
    kitti_config.image_extension = ".jpg"
    kitti_config.point_clouds_dir = "velodyne"
    kitti_config.calibrations_dir = "calib"

    kitti_config.kitti_sequence_to_frames_file = generate_sequence_map_file()
    get_mock_images(kitti_config)

    dataset_export_config.kitti_config.CopyFrom(kitti_config)
    return dataset_export_config


def get_mock_images(kitti_config):
    """Generate mock images from the image_ids."""
    image_root = os.path.join(
        kitti_config.root_directory_path, kitti_config.image_dir_name)
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    image_file = {'000012': (1242, 375),
                  '000000': (1224, 370),
                  '000001': (1242, 375)}
    for idx, sizes in six.iteritems(image_file):
        image_file_name = os.path.join(image_root,
                                       '{}{}'.format(idx, kitti_config.image_extension))
        image = Image.new("RGB", sizes)
        image.save(image_file_name)
    return image_file


def generate_sequence_map_file():
    """Generate a sequence map file for sequence wise partitioning kitti."""
    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)

    mock_sequence_to_frames_map = {'0': ['000000', '000001', '000012']}

    with open(temp_file_name, 'w') as tfile:
        json.dump(mock_sequence_to_frames_map, tfile)

    return temp_file_name


def _mock_open_image(image_file):
    """Mock image open()."""
    # the images are opened to figure out their dimensions so mock the image size
    mock_image = namedtuple('mock_image', ['size'])
    images = {'000012': mock_image((1242, 375)),
              '000000': mock_image((1224, 370)),
              '000001': mock_image((1242, 375))}

    # return the mocked image corresponding to frame_id in the image_file path
    for frame_id in images.keys():
        if frame_id in image_file:
            return images[frame_id]
    return mock_image


def _mock_converter(mocker, tmpdir):
    """Return a KITTI converter with a mocked sequence to frame map."""
    output_filename = os.path.join(str(tmpdir), 'kitti_test.tfrecords')

    # Mock image open().
    mocker.patch.object(Image, 'open', _mock_open_image)

    # Instead of using all KITTI labels, use only a few samples.
    mock_sequence_to_frames_map = {'0': ['000000', '000001', '000012']}
    # Convert a few KITTI labels to TFrecords.
    dataset_export_config = _dataset_export_config()
    mocker.patch.object(KITTIConverter, '_read_sequence_to_frames_file',
                        return_value=mock_sequence_to_frames_map)

    converter = build_converter(dataset_export_config, output_filename)
    converter.labels_dir = ""
    return converter, output_filename


@pytest.mark.parametrize("num_partitions,expected_partitions",
                         [(1, [list(range(1)) + list(range(2))
                               + list(range(3)) + list(range(4)) + list(range(5))]),
                             (2, [list(range(5)) + list(range(3)) +
                                  list(range(1)), list(range(4)) + list(range(2))]),
                             (3, [list(range(5)) + list(range(2)),
                                  list(range(4)) + list(range(1)), list(range(3))]),
                             (5, [list(range(num_items)) for num_items in
                                  range(5, 0, -1)])])
def test_partition(mocker, num_partitions, expected_partitions):
    """KITTI partitioning loops sequences starting from the longest one.

    Frames corresponding to sequences are added to partitions one-by-one.
    """
    dataset_export_config = _dataset_export_config(num_partitions)

    # Create a dummy mapping in which num_items maps to a list of length num_items.
    mock_sequence_to_frames_map = {num_items: list(range(num_items)) for
                                   num_items in range(1, 6)}
    mocker.patch.object(KITTIConverter, '_read_sequence_to_frames_file',
                        return_value=mock_sequence_to_frames_map)

    os_handle, output_filename = tempfile.mkstemp()
    os.close(os_handle)
    output_filename = os.path.join(str(output_filename), "kitti.testrecords")

    # Create a converter and run partitioning.
    converter = build_converter(dataset_export_config,
                                output_filename=output_filename)
    partitions = converter._partition()

    assert partitions == expected_partitions[::-1]  # reverse to match Rumpy
    return dataset_export_config


expected_objects = [[b'truck', b'car', b'cyclist', b'dontcare', b'dontcare',
                    b'dontcare', b'dontcare'],
                    [b'pedestrian'],
                    [b'car', b'van', b'dontcare', b'dontcare', b'dontcare']]
expected_truncation = [[0.0, 0.0, -1.0, -1.0, -1.0],
                       [0.0],
                       [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]]
expected_occlusion = [[0, 0, -1, -1, -1], [0], [0, 0, 3, -1, -1, -1, -1]]
expected_x1 = [[662.20, 448.07, 610.5, 582.969971, 600.359985],
               [712.4],
               [599.41, 387.63, 676.60, 503.89, 511.35, 532.37, 559.62]]
expected_y1 = [[185.85, 177.14, 179.95, 182.70, 185.59],
               [143.0],
               [156.40, 181.54, 163.95, 169.71, 174.96, 176.35, 175.83]]
expected_x2 = [[690.21, 481.60, 629.68, 594.78, 608.36],
               [810.73],
               [629.75, 423.81, 688.98, 590.61, 527.81, 542.68, 575.40]]
expected_y2 = [[205.03, 206.41, 196.31, 191.05, 192.69],
               [307.92],
               [189.25, 203.12, 193.93, 190.13, 187.45, 185.27, 183.15]]

expected_truncation = expected_truncation[::-1]
expected_occlusion = expected_occlusion[::-1]
expected_x1 = expected_x1[::-1]
expected_y1 = expected_y1[::-1]
expected_x2 = expected_x2[::-1]
expected_y2 = expected_y2[::-1]
expected_point_cloud_channels = [[4], [4], [4]]

expected_T_lidar_to_camera1 = \
    np.array([1.0, 0.0, 0.1, 4.0, 0.0, 0.0, -1.0, 0.0,
              0.0, 1.0, 0.2, 6.0, 0.0, 0.0, 0.0, 1.0]).T
expected_T_lidar_to_camera2 = np.array(
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).T
expected_T_lidar_to_camera3 = np.array(
    [0, 0, 1, 0, 0, 1, 0, 6, -1, 0, 0, -4, 0, 0, 0, 1]).T
expected_T_lidar_to_camera = [expected_T_lidar_to_camera1, expected_T_lidar_to_camera2,
                              expected_T_lidar_to_camera3]
expected_T_lidar_to_camera = expected_T_lidar_to_camera[::-1]
expected_P_lidar_to_image1 = np.array(
    [10., 2., 1.4, 57., 0., 3., -9.4, 18., 0., 1., 0.2, 6.]).T
expected_P_lidar_to_image2 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]).T
expected_P_lidar_to_image3 = np.array(
    [0, 0, 2, 3, 0, 2, 0, 12, -1, 0, 0, -4]).T
expected_P_lidar_to_image = [expected_P_lidar_to_image1, expected_P_lidar_to_image2,
                             expected_P_lidar_to_image3]
expected_P_lidar_to_image = expected_P_lidar_to_image[::-1]
single_class_expected_values = (expected_objects, expected_truncation, expected_occlusion,
                                expected_x1, expected_y1, expected_x2, expected_y2,
                                expected_point_cloud_channels, expected_T_lidar_to_camera,
                                expected_P_lidar_to_image)

expected_objects = [[b'truck', b'car', b'cyclist', b'dontcare', b'dontcare',
                    b'dontcare', b'dontcare'],
                    [b'pedestrian'],
                    [b'car', b'van', b'dontcare', b'dontcare', b'dontcare']]
expected_truncation = [[0.0, 0.0, -1.0, -1.0, -1.0],
                       [0.0],
                       [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]]
expected_occlusion = [[0, 0, -1, -1, -1], [0], [0, 0, 3, -1, -1, -1, -1]]
expected_x1 = [[662.20, 448.07, 610.5, 582.97, 600.36],
               [712.40],
               [599.41, 387.63, 676.60, 503.89, 511.35, 532.37, 559.62]]
expected_y1 = [[185.85, 177.14, 179.95, 182.70, 185.59],
               [143.0],
               [156.4, 181.54, 163.95, 169.71, 174.96, 176.35, 175.83]]
expected_x2 = [[690.21, 481.60, 629.68, 594.78, 608.36],
               [810.73],
               [629.75, 423.81, 688.98, 590.61, 527.81, 542.68, 575.40]]
expected_y2 = [[205.03, 206.41, 196.31, 191.05, 192.69],
               [307.92],
               [189.25, 203.12, 193.93, 190.13, 187.45, 185.27, 183.15]]

expected_truncation = expected_truncation[::-1]
expected_occlusion = expected_occlusion[::-1]
expected_x1 = expected_x1[::-1]
expected_y1 = expected_y1[::-1]
expected_x2 = expected_x2[::-1]
expected_y2 = expected_y2[::-1]

multi_class_expected_values = (expected_objects, expected_truncation, expected_occlusion,
                               expected_x1, expected_y1, expected_x2, expected_y2,
                               expected_point_cloud_channels, expected_T_lidar_to_camera,
                               expected_P_lidar_to_image)


@pytest.mark.parametrize("expected_values",
                         [single_class_expected_values, multi_class_expected_values])
def test_tfrecords_roundtrip(mocker, tmpdir, expected_values):
    """Test converting a few labels to TFRecords and parsing them back to Python."""
    converter, output_filename = _mock_converter(mocker, tmpdir)
    converter.convert()

    tfrecords = tf.python_io.tf_record_iterator(output_filename)

    # Common to all test cases
    frame_ids = ['000001', '000000', '000012']
    expected_point_cloud_ids = [
        'velodyne/' + frame_id for frame_id in frame_ids]
    expected_frame_ids = ['image_2/' + frame_id for frame_id in frame_ids]
    expected_img_widths = [1242, 1224, 1242]
    expected_img_heights = [375, 370, 375]

    # Specific to each test case
    expected_objects, expected_truncation, expected_occlusion, \
        expected_x1, expected_y1, expected_x2, expected_y2, \
        expected_point_cloud_channels, expected_T_lidar_to_camera, \
        expected_P_lidar_to_image = expected_values

    for i, record in enumerate(tfrecords):
        example = tf.train.Example()
        example.ParseFromString(record)

        features = example.features.feature

        assert features['frame/id'].bytes_list.value[0] == bytes(expected_frame_ids[i], 'utf-8')
        assert features['target/object_class'].bytes_list.value[:] == expected_objects[i]
        assert features['frame/width'].int64_list.value[0] == expected_img_widths[i]
        assert features['frame/height'].int64_list.value[0] == expected_img_heights[i]

        assert features['target/truncation'].float_list.value[:] == expected_truncation[i]
        assert features['target/occlusion'].int64_list.value[:] == expected_occlusion[i]

        bbox_features = ['target/coordinates_' +
                         x for x in ('x1', 'y1', 'x2', 'y2')]
        bbox_values = [expected_x1[i], expected_y1[i],
                       expected_x2[i], expected_y2[i]]

        for feature, expected_value in zip(bbox_features, bbox_values):
            np.testing.assert_allclose(
                features[feature].float_list.value[:], expected_value)

        assert features['point_cloud/id'].bytes_list.value[0].decode() \
            == expected_point_cloud_ids[i]

        assert features['point_cloud/num_input_channels'].int64_list.value[:] == \
            expected_point_cloud_channels[i]

        np.testing.assert_allclose(
            features['calibration/T_lidar_to_camera'].float_list.value[:],
            expected_T_lidar_to_camera[i])

        np.testing.assert_allclose(
            features['calibration/P_lidar_to_image'].float_list.value[:],
            expected_P_lidar_to_image[i])
    return converter


def test_count_targets(mocker, tmpdir):
    """Test that count_targets counts objects correctly."""
    # Take a few examples from the KITTI dataset.
    object_counts = {
        '000000': {
            b'pedestrian': 1
        },
        '000001': {
            b'car': 1,
            b'truck': 1,
            b'cyclist': 1,
            b'dontcare': 4
        },
        '000012': {
            b'car': 1,
            b'van': 1,
            b'dontcare': 3
        }
    }

    converter, _ = _mock_converter(mocker, tmpdir)

    # Check the counts.
    for frame_id, object_count in six.iteritems(object_counts):
        example = converter._create_example_proto(frame_id)
        returned_count = converter._count_targets(example)
        assert returned_count == object_count

    return converter
