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

"""Manage generation of tfrecords and related output files."""

from collections import defaultdict
import json
import os
import random
import numpy as np
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.common.dataio.data_converter import DataConverter
# from nvidia_tao_tf1.cv.common.dataio.desq_converter import write_desq
from nvidia_tao_tf1.cv.common.dataio.gt_converter import GtConverter
from nvidia_tao_tf1.cv.common.dataio.jsonlabels_strategy import JsonLabelsStrategy
from nvidia_tao_tf1.cv.common.dataio.sdklabels_strategy import SdkLabelsStrategy
from nvidia_tao_tf1.cv.common.dataio.set_strategy_generator import SetStrategyGenerator
from nvidia_tao_tf1.cv.common.dataio.tfrecord_generator import TfRecordGenerator
from nvidia_tao_tf1.cv.common.dataio.utils import is_kpi_set, PipelineReporter


class TfRecordManager(object):
    """Manage the flow of data source files to tfrecord generation."""

    def __init__(
        self,
        set_id,
        experiment_folder_suffix,
        use_unique,
        use_filtered,
        use_undistort,
        landmarks_folder_name,
        sdklabels_folder_name,
        norm_folder_name,
        set_root_path,
        save_images=True
    ):
        """Initialize parameters.

        Args:
            set_id (str): Set for which to generate tfrecords.
            experiment_folder_suffix (str): Suffix of experiment folder containing tfrecords.
            use_unique (bool): Only create records for first frame in a series if true.
            use_filtered (bool): Filter frames if true.
            use_undistort (bool): Use undistorted/not calibrated pts and frames if true.
            landmarks_folder_name (str): Folder name to obtain predicted fpe landmarks from.
            sdklabels_folder_name (str): Folder name to obtain predicted nvhelnet sdk labels from.
            norm_folder_name (str): Folder name to save normalized face, eyes and frame images.
            save_images (bool): Flag to determine if want to save normalized faces and images.
        """
        self._set_id = set_id
        self._experiment_folder_suffix = experiment_folder_suffix
        self._use_unique = use_unique
        self._use_filtered = use_filtered
        self._use_undistort = use_undistort
        self._landmarks_folder_name = landmarks_folder_name
        self._sdklabels_folder_name = sdklabels_folder_name
        self._norm_folder_name = norm_folder_name
        self._save_images = save_images
        self._set_root_path = set_root_path

        self._extract_cosmos_paths()

        strategy = None
        if self._strategy_type == 'sdk':
            strategy = SdkLabelsStrategy(
                set_id,
                self._use_unique,
                self._logger,
                self._set_strategy,
                self._norm_folder_name,
                self._save_images)
        elif self._strategy_type == 'json':
            strategy = JsonLabelsStrategy(
                set_id,
                self._use_unique,
                self._logger,
                self._set_strategy,
                self._norm_folder_name,
                self._save_images)
        else:
            self._logger.add_error('No such strategy {}'.format(self._strategy_type))
            raise ValueError('Invalid strategy.')

        self._generator = TfRecordGenerator(strategy)
        self._generate_data()

    def _extract_cosmos_paths(self):
        tfrecord_folder_name = 'TfRecords'
        gt_folder_name = 'GT'
        if self._use_unique:
            tfrecord_folder_name += '_unique'
            gt_folder_name += '_unique'

        self._set_strategy = SetStrategyGenerator(
            self._set_id,
            self._experiment_folder_suffix,
            tfrecord_folder_name,
            gt_folder_name,
            self._use_filtered,
            self._use_undistort,
            self._landmarks_folder_name,
            self._sdklabels_folder_name,
            self._set_root_path)
        self._strategy_type, self._paths = self._set_strategy.get_source_paths()

        self._logger = PipelineReporter(self._paths.error_path, 'tfrecord_generation', self._set_id)

    def _generate_data(self):
        self._tfrecord_data = self._generator.get_tfrecord_data()

        # Generate joint records if not kpi set
        if not is_kpi_set(self._set_id):
            self._joint_data = defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict())))
            for user in self._tfrecord_data.keys():
                for region in self._tfrecord_data[user].keys():
                    for frame in self._tfrecord_data[user][region].keys():
                        frame_data_dict = self._tfrecord_data[user][region][frame]
                        if not frame_data_dict['train/valid_theta_phi']:
                            continue
                        self._joint_data[user][region][frame] = frame_data_dict

    def _do_write_joint(self):
        if is_kpi_set(self._set_id) or 'eoc' in self._set_id.lower():
            return False
        return True

    def generate_tfrecords(self):
        """Generate tfrecords."""
        converter = DataConverter(self._paths.tfrecord_path,
                                  self._landmarks_folder_name is not None)
        converter.write_user_tfrecords(self._tfrecord_data)

        if self._do_write_joint():
            converter = DataConverter(self._paths.tfrecord_path + '_joint',
                                      self._landmarks_folder_name is not None)
            converter.write_user_tfrecords(self._joint_data)

    def generate_gt(self):
        """Generate ground truth txt files for debugging."""
        converter = GtConverter(self._paths.gt_path,
                                self._landmarks_folder_name is not None)
        converter.write_gt_files(self._tfrecord_data)
        if is_kpi_set(self._set_id):
            converter.write_landmarks(self._tfrecord_data)

        if self._do_write_joint():
            converter = GtConverter(self._paths.gt_path + '_joint',
                                    self._landmarks_folder_name is not None)
            converter.write_gt_files(self._joint_data)
            if is_kpi_set(self._set_id):
                converter.write_landmarks(self._joint_data)

    # def write_desq(self, update=False):
    #     """Write data to desq database."""
    #     write_desq(self._tfrecord_data, self._strategy_type, self._landmarks_folder_name,
    #                self._sdklabels_folder_name, update=update)

    @staticmethod
    def _summarize(data, test_users, validation_users, train_users, summarize_file_path):
        """Summarize the mean and std of points for a set."""
        def _summarize_helper(users, category, json_dict):
            landmarks_arr = []
            eye_feat_arr = []
            landmarks_arr_norm = []
            landmarks_arr_3D = []
            n_samples = 0

            for user in users:
                for region in data[user].keys():
                    for frame in data[user][region].keys():
                        n_samples += 1
                        landmarks_arr.append(data[user][region][frame]['train/landmarks']
                                             .reshape((104, 2)))
                        eye_feat_arr.append(data[user][region][frame]['train/eye_features']
                                            .reshape((56, 1)))
                        landmarks_arr_norm.append(data[user][region][frame]['train/norm_landmarks']
                                                  .reshape((104, 2)))
                        landmarks_arr_3D.append(data[user][region][frame]['train/landmarks_3D']
                                                .reshape((38, 3)))

            if landmarks_arr:
                np_landmarks_arr = np.stack(landmarks_arr, axis=0)
                json_dict[category + '_mean_lm'] = np.mean(
                    np_landmarks_arr, dtype=np.float, axis=0).tolist()
                json_dict[category + '_std_lm'] = np.std(
                    np_landmarks_arr, dtype=np.float, axis=0).tolist()
                json_dict[category + '_var_lm'] = np.var(
                    np_landmarks_arr, dtype=np.float, axis=0).tolist()
            else:
                default_lm = np.empty([104, 2], dtype=np.float)
                default_lm.fill(-1)

                json_dict[category + '_mean_lm'] = default_lm.tolist()
                json_dict[category + '_std_lm'] = default_lm.tolist()
                json_dict[category + '_var_lm'] = default_lm.tolist()

            if eye_feat_arr:
                np_eye_features_arr = np.stack(eye_feat_arr, axis=0)
                json_dict[category + '_mean_eye_feat'] = np.mean(
                    np_eye_features_arr, dtype=np.float, axis=0).tolist()
                json_dict[category + '_std_eye_feat'] = np.std(
                    np_eye_features_arr, dtype=np.float, axis=0).tolist()
                json_dict[category + '_var_eye_feat'] = np.var(
                    np_eye_features_arr, dtype=np.float, axis=0).tolist()
            else:
                default_eye_feat = np.empty([56, 1], dtype=np.float)
                default_eye_feat.fill(-1)

                json_dict[category + '_mean_eye_feat'] = default_eye_feat.tolist()
                json_dict[category + '_std_eye_feat'] = default_eye_feat.tolist()
                json_dict[category + '_var_eye_feat'] = default_eye_feat.tolist()

            if landmarks_arr_norm:
                np_landmarks_arr_norm = np.stack(landmarks_arr_norm, axis=0)
                json_dict[category + '_mean_norm_lm'] = np.mean(
                    np_landmarks_arr_norm, dtype=np.float, axis=0).tolist()
                json_dict[category + '_std_norm_lm'] = np.std(
                    np_landmarks_arr_norm, dtype=np.float, axis=0).tolist()
                json_dict[category + '_var_norm_lm'] = np.var(
                    np_landmarks_arr_norm, dtype=np.float, axis=0).tolist()
            else:
                default_lm = np.empty([104, 2], dtype=np.float)
                default_lm.fill(-1)

                json_dict[category + '_mean_norm_lm'] = default_lm.tolist()
                json_dict[category + '_std_norm_lm'] = default_lm.tolist()
                json_dict[category + '_var_norm_lm'] = default_lm.tolist()

            if landmarks_arr_3D:
                np_landmarks_arr_3D = np.stack(landmarks_arr_3D, axis=0)
                json_dict[category + '_mean_3D_lm'] = np.mean(
                    np_landmarks_arr_3D, dtype=np.float, axis=0).tolist()
                json_dict[category + '_std_3D_lm'] = np.std(
                    np_landmarks_arr_3D, dtype=np.float, axis=0).tolist()
                json_dict[category + '_var_3D_lm'] = np.var(
                    np_landmarks_arr_3D, dtype=np.float, axis=0).tolist()
            else:
                default_lm = np.empty([38, 3], dtype=np.float)
                default_lm.fill(-1)

                json_dict[category + '_mean_3D_lm'] = default_lm.tolist()
                json_dict[category + '_std_3D_lm'] = default_lm.tolist()
                json_dict[category + '_var_3D_lm'] = default_lm.tolist()

            json_dict[category + '_num_samples'] = n_samples

        json_dict = {}
        _summarize_helper(test_users, 'test', json_dict)
        _summarize_helper(validation_users, 'validate', json_dict)
        _summarize_helper(train_users, 'train', json_dict)

        data_json_path = expand_path(f"{summarize_file_path}/data.json")
        with open(data_json_path, 'w') as data_json_file:
            json.dump(json_dict, data_json_file, indent=4, sort_keys=True)

    def split_tfrecords(self):
        """Local split of users in a set."""
        def _split_users(data):
            users = data.keys()
            n_users = len(users)
            size = int(n_users / 8)
            random.seed(1)

            users = sorted(users)

            reordered_users = random.sample(users, n_users)
            if size == 0:
                size = 1

            if is_kpi_set(self._set_id):
                size = n_users

            test_users = reordered_users[:size]
            validation_users = reordered_users[size:size * 2]
            train_users = reordered_users[size * 2:]

            print('Test', test_users)
            print('Validation', validation_users)
            print('Train', train_users)

            return data, test_users, validation_users, train_users

        tfrecords_combined_path = self._paths.tfrecord_path + '_combined'
        data_converter = DataConverter(tfrecords_combined_path,
                                       self._landmarks_folder_name is not None)
        data, test_users, validation_users, train_users = _split_users(self._tfrecord_data)

        data_converter.write_combined_tfrecords(data, test_users, validation_users, train_users)
        self._summarize(data, test_users, validation_users, train_users, tfrecords_combined_path)

        gt_combined_path = self._paths.gt_path + '_combined'
        gt_converter = GtConverter(gt_combined_path, self._landmarks_folder_name is not None)
        gt_converter.write_combined_gt_files(data, test_users, validation_users, train_users)

        if is_kpi_set(self._set_id):
            gt_converter.write_combined_landmarks(data, test_users, validation_users, train_users)

        if self._do_write_joint():
            tfrecords_joint_combined_path = self._paths.tfrecord_path + '_joint_combined'
            data_joint_converter = DataConverter(tfrecords_joint_combined_path,
                                                 self._landmarks_folder_name is not None)
            data, test_users, validation_users, train_users = _split_users(self._joint_data)

            data_joint_converter.write_combined_tfrecords(
                data,
                test_users,
                validation_users,
                train_users)

            self._summarize(
                data,
                test_users,
                validation_users,
                train_users,
                tfrecords_joint_combined_path)

            gt_combined_path = self._paths.gt_path + '_joint_combined'
            gt_converter = GtConverter(gt_combined_path, self._landmarks_folder_name is not None)
            gt_converter.write_combined_gt_files(data, test_users, validation_users, train_users)

            if is_kpi_set(self._set_id):
                gt_converter.write_combined_landmarks(
                    data,
                    test_users,
                    validation_users,
                    train_users)
