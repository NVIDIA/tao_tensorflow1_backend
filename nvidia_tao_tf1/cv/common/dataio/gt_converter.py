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

"""Write debug ground truth txt files."""

import os
import subprocess
import numpy as np
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.common.dataio.utils import mkdir


class GtConverter(object):
    """Converts a dataset to GT txt files."""

    gt_features = [
        'train/facebbx_x',
        'train/facebbx_y',
        'train/facebbx_w',
        'train/facebbx_h',
        'train/lefteyebbx_x',
        'train/lefteyebbx_y',
        'train/lefteyebbx_w',
        'train/lefteyebbx_h',
        'train/righteyebbx_x',
        'train/righteyebbx_y',
        'train/righteyebbx_w',
        'train/righteyebbx_h',
        'label/gaze_cam_x',
        'label/gaze_cam_y',
        'label/gaze_cam_z',
        'label/gaze_screen_x',
        'label/gaze_screen_y',
        'label/hp_pitch',
        'label/hp_yaw',
        'label/hp_roll',
        'label/theta',
        'label/phi',
        'label/mid_cam_x',
        'label/mid_cam_y',
        'label/mid_cam_z',
        'label/lpc_cam_x',
        'label/lpc_cam_y',
        'label/lpc_cam_z',
        'label/rpc_cam_x',
        'label/rpc_cam_y',
        'label/rpc_cam_z',
        'label/head_pose_theta',
        'label/head_pose_phi',
        'label/theta_mid',
        'label/phi_mid',
        'label/theta_le',
        'label/phi_le',
        'label/theta_re',
        'label/phi_re',
        'train/valid_theta_phi',
        'train/num_keypoints',
        'train/source',
        'train/norm_frame_path',
        'label/norm_face_gaze_theta',
        'label/norm_face_gaze_phi',
        'label/norm_face_hp_theta',
        'label/norm_face_hp_phi',
        'label/norm_leye_gaze_theta',
        'label/norm_leye_gaze_phi',
        'label/norm_leye_hp_theta',
        'label/norm_leye_hp_phi',
        'label/norm_reye_gaze_theta',
        'label/norm_reye_gaze_phi',
        'label/norm_reye_hp_theta',
        'label/norm_reye_hp_phi',
        'train/norm_facebb_x',
        'train/norm_facebb_y',
        'train/norm_facebb_w',
        'train/norm_facebb_h',
        'train/norm_leyebb_x',
        'train/norm_leyebb_y',
        'train/norm_leyebb_w',
        'train/norm_leyebb_h',
        'train/norm_reyebb_x',
        'train/norm_reyebb_y',
        'train/norm_reyebb_w',
        'train/norm_reyebb_h',
        'train/norm_per_oof',
        'train/norm_face_cnv_mat',
        'train/norm_leye_cnv_mat',
        'train/norm_reye_cnv_mat',
        'label/face_cam_x',
        'label/face_cam_y',
        'label/face_cam_z'
    ]

    lm_pred_gt_features = [feature for feature in gt_features
                           if (
                               'lefteyebbx' not in feature and
                               'righteyebbx' not in feature and
                               'facebbx' not in feature and
                               'num_eyes_detected' not in feature
                           )]

    def __init__(self, gt_files_path, use_lm_pred):
        """Initialize file paths and features.

        Args:
            gt_files_path (path): Path to dump ground truth files.
            use_lm_pred (bool): True if using predicted landmarks.
        """
        self._gt_files_path = gt_files_path

        # Overwrite the gt txt files, cleanup
        if os.path.exists(expand_path(gt_files_path)):
            subprocess.run('rm -r ' + gt_files_path, shell=False)
        mkdir(gt_files_path)

        # Get different set of features if using predicted landmarks
        self._gt_features = self.gt_features
        if use_lm_pred:
            self._gt_features = self.lm_pred_gt_features

    @staticmethod
    def _get_frame_gt(gt_features, frame_dict):
        split_frame = frame_dict['train/image_frame_name'].rsplit('/', 1)
        path_to_frame = split_frame[0]
        frame_name = split_frame[1]

        features_list = [
            path_to_frame,
            frame_name
        ]
        for feature in gt_features:
            val = ''
            if feature == 'train/lefteyebbx_w':
                val = str(frame_dict[feature] + frame_dict['train/lefteyebbx_x'])
            elif feature == 'train/lefteyebbx_h':
                val = str(frame_dict[feature] + frame_dict['train/lefteyebbx_y'])
            elif feature == 'train/righteyebbx_w':
                val = str(frame_dict[feature] + frame_dict['train/righteyebbx_x'])
            elif feature == 'train/righteyebbx_h':
                val = str(frame_dict[feature] + frame_dict['train/righteyebbx_y'])
            elif isinstance(frame_dict[feature], np.ndarray):
                val = ' '.join(map(str, frame_dict[feature].flatten()))
            else:
                val = str(frame_dict[feature])
            features_list.append(val)

        return ' '.join(features_list) + '\n'

    def write_landmarks(self, users_dict):
        """Write landmarks GT files to separate it from other features."""
        for user in users_dict.keys():
            user_files_path = expand_path(f"{self._gt_files_path}/{user}_landmarks.txt")
            user_writer = open(user_files_path, 'a')

            for region in users_dict[user].keys():
                for frame in users_dict[user][region].keys():
                    frame_data_dict = users_dict[user][region][frame]

                    user_writer.write(self._get_frame_gt(
                        ['train/landmarks', 'train/landmarks_occ'],
                        frame_data_dict))

            user_writer.close()

    def write_gt_files(self, users_dict):
        """Write GT files."""
        for user in users_dict.keys():
            user_files_path = expand_path(f"{self._gt_files_path}/{user}.txt")
            user_writer = open(user_files_path, 'a')

            for region in users_dict[user].keys():
                for frame in users_dict[user][region].keys():
                    frame_data_dict = users_dict[user][region][frame]

                    user_writer.write(self._get_frame_gt(
                        self._gt_features,
                        frame_data_dict))

            user_writer.close()

    def write_combined_landmarks(self, combined_dict, test_users, validation_users, train_users):
        """Write combined landmarks GT files after splitting into test, train, validation."""
        def _write_category_files(combined_dict, category_users, gt_file_path):
            gt_writer = open(gt_file_path, 'a')
            for user in category_users:
                for region in combined_dict[user].keys():
                    for frame in combined_dict[user][region].keys():
                        frame_data_dict = combined_dict[user][region][frame]

                        gt_writer.write(self._get_frame_gt(
                            ['train/landmarks', 'train/landmarks_occ'],
                            frame_data_dict))
            gt_writer.close()

        test_path = os.path.join(self._gt_files_path, 'test_landmarks.txt')
        validation_path = os.path.join(self._gt_files_path, 'validate_landmarks.txt')
        train_path = os.path.join(self._gt_files_path, 'train_landmarks.txt')

        _write_category_files(combined_dict, test_users, test_path)
        _write_category_files(combined_dict, validation_users, validation_path)
        _write_category_files(combined_dict, train_users, train_path)

    def write_combined_gt_files(self, combined_dict, test_users, validation_users, train_users):
        """Write combined GT files after splitting into test, train, validation."""
        def _write_category_files(combined_dict, category_users, gt_file_path):
            gt_writer = open(gt_file_path, 'a')
            for user in category_users:
                for region in combined_dict[user].keys():
                    for frame in combined_dict[user][region].keys():
                        frame_data_dict = combined_dict[user][region][frame]

                        gt_writer.write(self._get_frame_gt(
                            self._gt_features,
                            frame_data_dict))
            gt_writer.close()

        test_path = os.path.join(self._gt_files_path, 'test.txt')
        validation_path = os.path.join(self._gt_files_path, 'validate.txt')
        train_path = os.path.join(self._gt_files_path, 'train.txt')

        _write_category_files(combined_dict, test_users, test_path)
        _write_category_files(combined_dict, validation_users, validation_path)
        _write_category_files(combined_dict, train_users, train_path)
