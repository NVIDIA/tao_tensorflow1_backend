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

"""Stand-alone script that can read the generated tfrecords."""

import argparse
import os
import sys
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.common.dataio.data_converter import DataConverter


def main(args=None):
    """Generate tfrecords based on user arguments."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Read TfRecords")
    parser.add_argument('-folder-name', '--ground_truth_experiment_folder_name',
                        type=str, required=True,
                        help='Name of ground truth experiment folder containing tfrecords')
    parser.add_argument('-nfs', '--nfs_storage', type=str,
                        required=True, help='NFS cosmos storage name')
    # This script uses -pred instead of -landmarks-folder, not caring about the location.
    parser.add_argument('-pred', '--use_landmark_predictions',
                        type=lambda x: (str(x).lower() == 'true'),
                        default=False, help='Read landmarks predictions')
    parser.add_argument('-set', '--set_id', type=str,
                        required=True, help='Setid (ex. s427-gaze-2)')
    args = parser.parse_args(args)

    ground_truth_factory = args.ground_truth_experiment_folder_name
    nfs_name = args.nfs_storage
    use_lm_pred = args.use_landmark_predictions
    set_id = args.set_id

    tfrecord_folder_name = 'TfRecords_combined'

    if nfs_name == 'driveix.cosmos639':
        spec_set_path = expand_path(f"/home/{nfs_name}/GazeData/postData/{set_id}")
        set_gt_factory = expand_path(f"{spec_set_path}/{ground_truth_factory}")
    elif nfs_name in ('copilot.cosmos10', 'projects1_copilot'):
        set_gt_factory = expand_path(f"/home/{nfs_name}/RealTimePipeline/set/{set_id}/{ground_truth_factory}")
    else:
        raise IOError('No such NFS cosmos storage')

    tfrecords_path = os.path.join(set_gt_factory, tfrecord_folder_name)

    if os.path.isdir(tfrecords_path):
        for filename in os.listdir(tfrecords_path):
            if filename.endswith('.tfrecords'):
                DataConverter.read_tfrecords([os.path.join(tfrecords_path, filename)], use_lm_pred)


if __name__ == "__main__":
    main()
