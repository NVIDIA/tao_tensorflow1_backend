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

import json
import os
import pytest

from nvidia_tao_tf1.cv.bpnet.dataio.build_converter import build_converter
from nvidia_tao_tf1.cv.bpnet.dataio.coco_converter import COCOConverter

BPNET_ROOT = os.getenv(
    "CI_PROJECT_DIR",
    "/workspace/tao-tf1"
)

TEST_DATA_ROOT_PATH = os.path.join(
    BPNET_ROOT,
    "nvidia_tao_tf1/cv/bpnet"
)


@pytest.mark.parametrize("dataset_spec,converter_type", [
    ('coco_spec.json', COCOConverter),
])
def test_build_converter(dataset_spec, converter_type):
    """Check that build_converter returns a DatasetConverter of the desired type."""

    dataset_spec_path = \
        os.path.join('nvidia_tao_tf1/cv/bpnet/dataio/dataset_specs', dataset_spec)

    # Load config file
    with open(dataset_spec_path, "r") as f:
        dataset_spec_json = json.load(f)

    # Update the paths based on test data
    dataset_spec_json["train_data"] = {
        "images_root_dir_path": "coco/images",
        "mask_root_dir_path": "coco/masks",
        "annotation_root_dir_path": "coco/person_keypoints_test.json"
    }
    dataset_spec_json["test_data"] = {
        "images_root_dir_path": "coco/images",
        "mask_root_dir_path": "coco/masks",
        "annotation_root_dir_path": "coco/person_keypoints_test.json"
    }

    dataset_spec_json['root_directory_path'] = os.path.join(
        TEST_DATA_ROOT_PATH, "dataio/tests/test_data")
    print(f"Test data root path {TEST_DATA_ROOT_PATH}")
    converter = build_converter(dataset_spec_json, 'test/coco')

    assert isinstance(converter, converter_type)
