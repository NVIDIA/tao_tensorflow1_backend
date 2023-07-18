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

"""Command line interface for converting pose datasets to TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging

from nvidia_tao_tf1.cv.bpnet.dataio.build_converter import build_converter


def main(args=None):
    """
    Convert an object detection dataset to TFRecords.

    Args:
        args(list): list of arguments to be parsed if called from another module.
    """
    parser = argparse.ArgumentParser(prog='dataset_converter',
                                     description='Convert pose datasets to TFRecords')
    parser.add_argument(
        '-d',
        '--dataset_spec',
        required=True,
        help='Path to the dataset spec containing config for exporting .tfrecords.')
    parser.add_argument(
        '-o',
        '--output_filename',
        required=True,
        help='Output file name.')
    parser.add_argument(
        '-m',
        '--mode',
        required=False,
        default='train',
        help='Converter mode: train/test.')
    parser.add_argument(
        '-p',
        '--num_partitions',
        required=False,
        default=1,
        help='Number of partitions (folds).')
    parser.add_argument(
        '-s',
        '--num_shards',
        required=False,
        default=0,
        help='Number of shards.')
    parser.add_argument(
        '--generate_masks',
        action='store_true',
        help='Generate and save masks of regions with unlabeled people - used for training.')
    parser.add_argument(
        '--check_files',
        action='store_true',
        help='Check if the files including images and masks exist in the given root data dir.')

    args = parser.parse_args(args)

    # Load config file
    with open(args.dataset_spec, "r") as f:
        dataset_spec_json = json.load(f)

    converter = build_converter(
        dataset_spec_json,
        args.output_filename,
        mode=args.mode,
        num_partitions=args.num_partitions,
        num_shards=args.num_shards,
        generate_masks=args.generate_masks,
        check_if_images_and_masks_exist=args.check_files)

    converter.convert()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    main()
