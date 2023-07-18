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

"""Build a dataset converter object based on dataset spec config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.bpnet.dataio.coco_converter import COCOConverter


def build_converter(
    dataset_spec,
    output_filename,
    mode='train',
    num_partitions=1,
    num_shards=0,
    generate_masks=False,
    check_if_images_and_masks_exist=False
):
    """Build a DatasetConverter object.

    Build and return an object of desired subclass of DatasetConverter based on
    given dataset export configuration.

    Args:
        dataset_spec (dict): Dataset export configuration spec
        output_filename (string): Path for the output file.
        mode (string): train/test.
        num_partitions (int): Number of folds
        num_shards (int): Number of shards
        generate_masks (bool): Flag to generate and save masks of regions with unlabeled people
        check_if_images_and_masks_exist (bool): check if the required files exist in data location.

    Return:
        converter (DatasetConverter): An object of desired subclass of DatasetConverter.
    """
    # Fetch the dataset configuration object first
    # TODO: Handle more datasets.
    # dataset_type = dataset_spec["dataset"]

    root_directory_path = dataset_spec["root_directory_path"]
    converter = COCOConverter(
        root_directory_path=root_directory_path,
        dataset_spec=dataset_spec,
        num_partitions=num_partitions,
        num_shards=num_shards,
        output_filename=output_filename,
        mode=mode,
        generate_masks=generate_masks,
        check_if_images_and_masks_exist=check_if_images_and_masks_exist)

    return converter
