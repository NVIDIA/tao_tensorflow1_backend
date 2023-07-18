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

"""Main file of pipeline script which generates tfrecords and related debug files."""

import argparse
import sys
from nvidia_tao_tf1.cv.common.dataio.tfrecord_manager import TfRecordManager
from nvidia_tao_tf1.cv.common.dataio.utils import is_kpi_set


def main(args=None):
    """Generate tfrecords based on user arguments."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
            description="Generate TFRecords from json post-labels or nvhelnet sdk labels")
    parser.add_argument('-folder-suffix', '--ground_truth_experiment_folder_suffix',
                        type=str, required=True,
                        help='Suffix of folder which will include generated tfrecords')
    parser.add_argument('-unique', '--use_unique', type=lambda x: (str(x).lower() == 'true'),
                        default=False, help='Take the first frame of every image')
    parser.add_argument('-filter', '--use_filtered', type=lambda x: (str(x).lower() == 'true'),
                        default=False, help='Use filtered frames on nvhelnet sets')
    parser.add_argument('-undistort', '--undistort', type=lambda x: (str(x).lower() == 'true'),
                        default=False, help='Read undistort frames')
    parser.add_argument('-landmarks-folder', '--landmarks_folder_name', type=str, default="",
                        help='Source folder to obtain predicted fpe landmarks from, '
                        'or "" for default landmarks from JSON')
    parser.add_argument('-sdklabels-folder', '--sdklabels_folder_name', type=str, default="",
                        help='Source folder to obtain nvhelnet sdk labels from, '
                        'or "" for sdk labels from JSON (with fallback to Nvhelnet_v11.2)')
    parser.add_argument('-norm_folder_name', '--norm_folder_name', type=str, required=True,
                        help='Folder to generate normalized data')
    parser.add_argument('-data_root_path', '--data_root_path', type=str, default="",
                        help='root path of the data')
    parser.add_argument('-sets', '--set_ids', type=str,
                        required=True, help='Setid (ex. s427-gaze-2)')
    args = parser.parse_args(args)

    set_ids = args.set_ids
    is_unique = args.use_unique

    if is_unique and is_kpi_set(set_ids):
        raise ValueError('Do not use unique for kpi sets.')

    is_filtered = args.use_filtered
    folder_suffix = args.ground_truth_experiment_folder_suffix
    if is_filtered:
        folder_suffix = 'filtered_' + folder_suffix

    if args.landmarks_folder_name == "":
        args.landmarks_folder_name = None
    elif not args.landmarks_folder_name.startswith('fpenet_results'):
        raise ValueError('Landmarks folder has to start with "fpenet_results".')
    if args.sdklabels_folder_name == "":
        args.sdklabels_folder_name = None
    # The check to see if these folders actually exists depends on the cosmos choice
    # and is therefore done in set_strategy.py's _get_landmarks_path(),
    # and set_label_sorter.py's _get_sdklabels_folder(), respectively.

    tfrecord_manager = TfRecordManager(
        set_ids,
        folder_suffix,
        is_unique,
        is_filtered,
        args.undistort,
        args.landmarks_folder_name,
        args.sdklabels_folder_name,
        args.norm_folder_name,
        args.data_root_path)
    tfrecord_manager.generate_tfrecords()
    tfrecord_manager.generate_gt()
    tfrecord_manager.split_tfrecords()


if __name__ == '__main__':
    main()
