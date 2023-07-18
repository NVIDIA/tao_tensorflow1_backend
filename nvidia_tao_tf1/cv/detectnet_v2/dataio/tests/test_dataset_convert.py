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

"""Tests for the dataset converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf.text_format import Merge as merge_text_proto
import pytest

from nvidia_tao_tf1.cv.detectnet_v2.dataio.build_converter import build_converter
from nvidia_tao_tf1.cv.detectnet_v2.dataio.dataset_converter_lib import DatasetConverter
from nvidia_tao_tf1.cv.detectnet_v2.dataio.kitti_converter_lib import KITTIConverter
import nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_export_config_pb2 as\
    dataset_export_config_pb2
import nvidia_tao_tf1.cv.detectnet_v2.scripts.dataset_convert as dataset_converter


@pytest.mark.parametrize("dataset_export_spec", ['kitti_spec.txt'])
def test_dataset_converter(mocker, dataset_export_spec):
    """Check that dataset_converter.py can be loaded and run."""
    dataset_export_spec_path = os.path.dirname(os.path.realpath(__file__))
    dataset_export_spec_path = \
        os.path.join(dataset_export_spec_path, '..', 'dataset_specs', dataset_export_spec)

    # Mock the slow functions of converters that are part of constructor.
    mocker.patch.object(KITTIConverter, "_read_sequence_to_frames_file", return_value=None)
    mock_converter = mocker.patch.object(DatasetConverter, "convert")
    args = ['-d', dataset_export_spec_path, '-o', './tmp']
    dataset_converter.main(args)
    mock_converter.assert_called_once()


@pytest.mark.parametrize("dataset_export_spec,converter_type", [('kitti_spec.txt', KITTIConverter)])
def test_build_converter(mocker, dataset_export_spec, converter_type):
    """Check that build_converter returns a DatasetConverter of the desired type."""
    # Mock the slow functions of converters that are part of constructor.
    mocker.patch.object(KITTIConverter, "_read_sequence_to_frames_file", return_value=None)

    dataset_export_spec_path = os.path.dirname(os.path.realpath(__file__))
    dataset_export_spec_path = \
        os.path.join(dataset_export_spec_path, '..', 'dataset_specs', dataset_export_spec)
    dataset_export_config = dataset_export_config_pb2.DatasetExportConfig()
    with open(dataset_export_spec_path, "r") as f:
        merge_text_proto(f.read(), dataset_export_config)

    converter = build_converter(dataset_export_config=dataset_export_config,
                                output_filename=dataset_export_spec_path)

    assert isinstance(converter, converter_type)
