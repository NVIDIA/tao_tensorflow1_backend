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

"""Export datasets to .tfrecords files based on dataset export config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from nvidia_tao_tf1.cv.detectnet_v2.dataio.build_converter import build_converter
from nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config_pb2 import DataSource

TEMP_TFRECORDS_DIR = os.path.join(tempfile.gettempdir(), 'temp-tfrecords')


def _clear_temp_dir():
    """If the temporary directory exists, remove its contents. Otherwise, create the directory."""
    temp_dir = TEMP_TFRECORDS_DIR

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir)


def export_tfrecords(dataset_export_config, validation_fold):
    """A helper function to export .tfrecords files based on a list of dataset export config.

    Args:
        dataset_export_config: A list of DatasetExportConfig objects.
        validation_fold (int): Validation fold number (0-based). Indicates which fold from the
            training data to use as validation. Can be None.
    Return:
        data_sources: A list of DataSource proto objects.
    """
    dataset_path = TEMP_TFRECORDS_DIR

    _clear_temp_dir()

    data_sources = []

    for dataset_idx, config in enumerate(dataset_export_config):
        temp_filename = str(dataset_idx)
        temp_filename = os.path.join(dataset_path, temp_filename)

        converter = build_converter(config, temp_filename, validation_fold)
        converter.convert()

        # Create a DataSource message for this dataset
        data_source = DataSource()
        data_source.tfrecords_path = temp_filename + '*'
        data_source.image_directory_path = config.image_directory_path

        data_sources.append(data_source)

    return data_sources
