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

"""Tests for exporting .tfrecords based on dataset export config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

from PIL import Image
import six
from six.moves import range
from nvidia_tao_tf1.cv.detectnet_v2.dataio.export import export_tfrecords, TEMP_TFRECORDS_DIR
from nvidia_tao_tf1.cv.detectnet_v2.dataio.kitti_converter_lib import KITTIConverter

import nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_export_config_pb2 as\
    dataset_export_config_pb2
import nvidia_tao_tf1.cv.detectnet_v2.proto.experiment_pb2 as experiment_pb2
import nvidia_tao_tf1.cv.detectnet_v2.proto.kitti_config_pb2 as kitti_config_pb2


def get_mock_images(kitti_config):
    """Generate mock images from the image_ids."""
    image_root = os.path.join(kitti_config.root_directory_path, kitti_config.image_dir_name)
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


def generate_sequence_map_file():
    """Generate a sequence map file for sequence wise partitioning kitti."""
    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)

    mock_sequence_to_frames_map = {'0': ['000000', '000001', '000012']}

    with open(temp_file_name, 'w') as tfile:
        json.dump(mock_sequence_to_frames_map, tfile)

    return temp_file_name


def test_export_tfrecords(mocker):
    """Create a set of dummy dataset export config and test exporting to .tfrecords."""
    kitti_config1 = kitti_config_pb2.KITTIConfig()
    kitti_config2 = kitti_config_pb2.KITTIConfig()

    export_config1 = dataset_export_config_pb2.DatasetExportConfig()
    export_config1.kitti_config.CopyFrom(kitti_config1)
    export_config1.image_directory_path = "images0"
    export_config1.kitti_config.partition_mode = "sequence"
    export_config1.kitti_config.image_dir_name = "image_2"
    export_config1.kitti_config.image_extension = ".jpg"
    export_config1.kitti_config.point_clouds_dir = "velodyne"
    export_config1.kitti_config.calibrations_dir = "calib"
    export_config1.kitti_config.kitti_sequence_to_frames_file = generate_sequence_map_file()
    get_mock_images(export_config1.kitti_config)

    export_config2 = dataset_export_config_pb2.DatasetExportConfig()
    export_config2.kitti_config.CopyFrom(kitti_config2)
    export_config2.image_directory_path = "images1"
    export_config2.kitti_config.partition_mode = "sequence"
    export_config2.kitti_config.image_dir_name = "image_2"
    export_config2.kitti_config.image_extension = ".jpg"
    export_config2.kitti_config.point_clouds_dir = "velodyne"
    export_config2.kitti_config.calibrations_dir = "calib"
    export_config2.kitti_config.kitti_sequence_to_frames_file = generate_sequence_map_file()
    get_mock_images(export_config2.kitti_config)

    experiment_config = experiment_pb2.Experiment()
    experiment_config.dataset_export_config.extend([export_config1, export_config2])

    # Mock these functions that take a lot of time in the constructor.
    mocker.patch.object(KITTIConverter, "_read_sequence_to_frames_file", return_value=None)

    # Mock the export step.
    kitti_converter = mocker.patch.object(KITTIConverter, 'convert')
    data_sources = export_tfrecords(experiment_config.dataset_export_config, 0)
    tfrecords_paths = [data_source.tfrecords_path for data_source in data_sources]
    image_dir_paths = [data_source.image_directory_path for data_source in data_sources]

    # Check that the expected path was returned and the converters were called as expected.
    assert tfrecords_paths == [TEMP_TFRECORDS_DIR + '/' + str(i) + '*' for i in range(2)]
    assert image_dir_paths == ['images' + str(i) for i in range(2)]
    assert kitti_converter.call_count == 2
