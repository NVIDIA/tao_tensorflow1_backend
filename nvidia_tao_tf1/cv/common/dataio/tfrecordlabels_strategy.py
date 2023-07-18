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

"""Abstract strategy for generating tfrecords."""

import abc
from collections import defaultdict
import copy
import os
import cv2
import numpy as np
from nvidia_tao_tf1.cv.common.dataio.data_converter import DataConverter
from nvidia_tao_tf1.cv.common.dataio.eye_features_generator import (
    EyeFeaturesGenerator)
from nvidia_tao_tf1.cv.common.dataio.eye_status import EyeStatus
from nvidia_tao_tf1.cv.common.dataio.theta_phi_angle_utils import (
    populate_gaze_info,
    populate_head_norm_bbinfo,
    populate_head_norm_float,
    populate_head_norm_listinfo,
    populate_head_norm_path,
    populate_theta_phi)
from nvidia_tao_tf1.cv.common.dataio.theta_phi_calc_utils import (
    NoPupilThetaPhiStrategy, ThetaPhiLandmarksGenerator)
from nvidia_tao_tf1.cv.common.dataio.utils import (
    get_file_ext, get_file_name_noext, is_kpi_set)


class TfRecordLabelsStrategy(object):
    """Abstract class for generating tfrecords."""

    __metaclass__ = abc.ABCMeta

    # Users in KPI sets which are also in training sets are no longer considered as KPI users.
    _non_kpi_users = (
        'ZTy4vjHcYs57_4Gq',
        'TNg6PdPzFMmNKuZgr7BbSSxkNRj0ibZzbH8n8kMAnuY=',
        'VBjKmYLtjNhrjC_A')

    class Pipeline_Constants():
        """Constants of pipeline script."""

        num_fid_points = 104

    @staticmethod
    def _populate_frame_dict(frame_dict, features, vals):
        assert len(features) == len(vals)

        n_items = len(features)
        for i in range(n_items):
            frame_dict[features[i]] = vals[i]

    def __init__(
        self,
        set_id,
        use_unique,
        logger,
        set_strategy,
        norm_folder_name,
        save_images
    ):
        """Initialize parameters.

        Args:
            set_id (str): Set for which to generate tfrecords.
            use_unique (bool): Only create records for first frame in a series if true.
            logger (Logger object): Report failures and number of tfrecords lost for tracking.
            set_strategy (SetStrategy object): Strategy for set type (gaze / eoc).
            save_images (bool): Whether to generate new folders and images for face crop, eyes, etc.
        """
        self._set_id = set_id
        self._use_unique = use_unique
        self._logger = logger
        self._set_strategy = set_strategy
        self._norm_folder_name = norm_folder_name
        self._save_images = save_images
        _, self._paths = set_strategy.get_source_paths()
        self._cam_intrinsics, self._cam_extrinsics, self._screen_params = \
            self._set_strategy.get_camera_parameters()

        # 4 level dict (user_name->region_name->frame_name->frame_data)
        self._users = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

        self._frame_width = None
        self._frame_height = None
        self._landmarks_path = None

    def _read_landmarks_from_path(self):
        assert self._landmarks_path is not None

        for user_file in os.listdir(self._landmarks_path):
            user_name = get_file_name_noext(user_file)
            user_lm_path = os.path.join(self._landmarks_path, user_file)
            with open(user_lm_path, 'r') as user_landmarks:
                for line in user_landmarks:
                    line_split = line.rstrip().split(' ')
                    if len(self._paths.regions) == 1 and self._paths.regions[0] == '':
                        # On bench data collection has no regions.
                        region_name = ''
                    else:
                        region_name = line_split[0].split('/')[-2]
                    frame_name = get_file_name_noext(line_split[0].split('/')[-1])

                    frame_dict = self._users[user_name][region_name][frame_name]
                    if (
                        'train/image_frame_width' not in frame_dict or
                        'train/image_frame_height' not in frame_dict
                    ):
                        self._logger.add_error(
                            '''Could not find frame width or height.
                            User {} frame {} may not exist'''.format(user_name, frame_name))
                        continue

                    frame_w = frame_dict['train/image_frame_width']
                    frame_h = frame_dict['train/image_frame_height']

                    landmarks_arr = np.empty([208, ], dtype=np.longdouble)
                    landmarks_arr.fill(-1)

                    # No occlusion information
                    landmarks_occ_arr = np.empty([104, ], dtype=np.longdouble)
                    landmarks_occ_arr.fill(-1)
                    frame_dict['train/landmarks_occ'] = np.copy(landmarks_occ_arr).astype(int)

                    num_min_sdk_landmarks = 68
                    if len(line_split) >= (1 + num_min_sdk_landmarks * 2):
                        read_landmarks = np.array(line_split[1:])  # ignore frame at beginning
                        read_landmarks = read_landmarks.astype(np.longdouble)

                        landmarks_2D = read_landmarks.reshape(-1, 2)
                        frame_dict['internal/landmarks_2D_distort'] = np.copy(landmarks_2D)

                        num_landmarks = landmarks_2D.shape[0]
                        frame_dict['train/num_keypoints'] = num_landmarks

                        landmarks_2D[:num_landmarks] = np.asarray(
                            self._set_strategy.get_pts(
                                landmarks_2D[:num_landmarks],
                                frame_w,
                                frame_h)).reshape(-1, 2)
                        frame_dict['internal/landmarks_2D'] = landmarks_2D

                        landmarks_arr[:num_landmarks * 2] = landmarks_2D.flatten().tolist()
                        frame_dict['train/landmarks'] = landmarks_arr

                        # Eye_features only dependent on landmarks
                        frame_dict['train/eye_features'] = EyeFeaturesGenerator(
                            landmarks_2D,
                            num_landmarks).get_eye_features()

    @abc.abstractmethod
    def extract_landmarks(self):
        """Abstract method for populating landmarks."""
        pass

    @abc.abstractmethod
    def extract_bbox(self):
        """Abstract method for populating bounding boxes."""
        pass

    @abc.abstractmethod
    def extract_eye_status(self):
        """Abstract method for populating eye status."""
        pass

    def get_tfrecord_data(self):
        """Factory method which returns populated data for tfrecord generation."""
        self._extract_frame()
        self.extract_landmarks()
        self.extract_bbox()
        self.extract_eye_status()
        self._extract_gaze_vec_info()
        self._label_source()
        self._prune_data()

        self._logger.write_to_log()

        return self._users

    def _extract_frame(self):
        for user_name in os.listdir(self._paths.data_path):
            if user_name.startswith('.'):
                # Not a valid folder
                continue

            user_dir_path = os.path.join(self._paths.data_path, user_name)
            if not os.path.isdir(user_dir_path):
                continue

            for region_name in self._paths.regions:
                region_dir_path = os.path.join(user_dir_path, region_name)
                if not os.path.isdir(region_dir_path):
                    self._logger.add_warning('Could not find region data dir {}, '
                                             'skipping this region'.format(region_dir_path))
                    continue
                for img_file in os.listdir(region_dir_path):
                    frame_name = get_file_name_noext(img_file)
                    frame_path = os.path.join(region_dir_path, img_file)

                    if get_file_ext(img_file) != '.png':
                        self._logger.add_warning('{} is not an image'.format(frame_path))
                        continue

                    if not os.path.exists(frame_path):
                        self._logger.add_error('Unable to find frame {}'.format(frame_path))
                        continue

                    self._users[user_name][region_name][frame_name][
                        'train/image_frame_name'] = frame_path

                    # All images in a set have the same frame size
                    if self._frame_width is None or self._frame_height is None:
                        image_frame = cv2.imread(frame_path)
                        self._frame_width = image_frame.shape[1]
                        self._frame_height = image_frame.shape[0]

                    self._users[user_name][region_name][frame_name][
                        'train/image_frame_width'] = self._frame_width
                    self._users[user_name][region_name][frame_name][
                        'train/image_frame_height'] = self._frame_height

                    self._set_strategy.extract_gaze_info(
                        self._users[user_name][region_name][frame_name], frame_name, region_name)

    def _extract_gaze_vec_info(self):
        should_calculate = False

        if self._cam_intrinsics is not None:
            camera_matrix, _, theta_phi_distortion_coeffs = self._cam_intrinsics
            should_calculate = True

        for user_name in list(self._users.keys()):
            for region_name in list(self._users[user_name].keys()):
                # Each region has its own extrinsic parameters.
                if self._cam_extrinsics is not None:
                    R, T = self._cam_extrinsics[region_name]
                    should_calculate = True

                for frame_name in list(self._users[user_name][region_name].keys()):
                    frame_data_dict = self._users[user_name][region_name][frame_name]

                    face_bbox = None
                    try:
                        x1 = frame_data_dict['internal/facebbx_x_distort']
                        y1 = frame_data_dict['internal/facebbx_y_distort']
                        x2 = x1 + frame_data_dict['internal/facebbx_w_distort']
                        y2 = y1 + frame_data_dict['internal/facebbx_h_distort']
                        face_bbox = [x1, y1, x2, y2]
                    except KeyError:
                        # Using Shagan's landmarks will result in no face bounding boxes
                        if self._landmarks_path is None:
                            continue

                    if 'internal/landmarks_2D_distort' not in frame_data_dict:
                        continue

                    landmarks_2D_valid = ThetaPhiLandmarksGenerator(
                        frame_data_dict['internal/landmarks_2D_distort'],
                        frame_data_dict['train/landmarks_occ'],
                        frame_data_dict['train/image_frame_width'],
                        frame_data_dict['train/image_frame_height'],
                        face_bbox).get_landmarks_in_frame()

                    if not should_calculate:
                        is_valid_theta_phi = False
                        frame_data_dict['train/valid_theta_phi'] = is_valid_theta_phi
                        angle_struct_ins = NoPupilThetaPhiStrategy.empty_angle_struct
                    else:
                        gt_cam = np.zeros((3, 1), dtype=np.longdouble)
                        gt_cam[0] = frame_data_dict['label/gaze_cam_x']
                        gt_cam[1] = frame_data_dict['label/gaze_cam_y']
                        gt_cam[2] = frame_data_dict['label/gaze_cam_z']

                        # No pupils to bridge gap between json and sdk info
                        is_valid_theta_phi, angle_struct_ins = NoPupilThetaPhiStrategy(
                            self._logger,
                            gt_cam,
                            camera_matrix,
                            theta_phi_distortion_coeffs,
                            R,
                            T,
                            landmarks_2D_valid,
                            self._users[user_name][region_name]
                            [frame_name]['train/image_frame_name'],
                            self._paths.info_source_path,
                            self._norm_folder_name,
                            self._save_images).get_gaze_angles_theta_phi()
                        frame_data_dict['train/valid_theta_phi'] = is_valid_theta_phi

                    frame_data_dict['label/hp_pitch'], \
                        frame_data_dict['label/hp_yaw'], \
                        frame_data_dict['label/hp_roll'] = populate_gaze_info(
                            angle_struct_ins.euler_angles,
                            is_valid_theta_phi)
                    frame_data_dict['label/theta'] = populate_theta_phi(
                        angle_struct_ins.theta_ovr,
                        is_valid_theta_phi)
                    frame_data_dict['label/theta_le'] = populate_theta_phi(
                        angle_struct_ins.theta_le,
                        is_valid_theta_phi)
                    frame_data_dict['label/theta_re'] = populate_theta_phi(
                        angle_struct_ins.theta_re,
                        is_valid_theta_phi)
                    frame_data_dict['label/theta_mid'] = populate_theta_phi(
                        angle_struct_ins.theta_mid,
                        is_valid_theta_phi)
                    frame_data_dict['label/head_pose_theta'] = populate_theta_phi(
                        angle_struct_ins.head_pose_theta,
                        is_valid_theta_phi)
                    frame_data_dict['label/phi'] = populate_theta_phi(
                        angle_struct_ins.phi_ovr,
                        is_valid_theta_phi)
                    frame_data_dict['label/phi_le'] = populate_theta_phi(
                        angle_struct_ins.phi_le,
                        is_valid_theta_phi)
                    frame_data_dict['label/phi_re'] = populate_theta_phi(
                        angle_struct_ins.phi_re,
                        is_valid_theta_phi)
                    frame_data_dict['label/phi_mid'] = populate_theta_phi(
                        angle_struct_ins.phi_mid,
                        is_valid_theta_phi)
                    frame_data_dict['label/head_pose_phi'] = populate_theta_phi(
                        angle_struct_ins.head_pose_phi,
                        is_valid_theta_phi)
                    frame_data_dict['label/lpc_cam_x'], \
                        frame_data_dict['label/lpc_cam_y'], \
                        frame_data_dict['label/lpc_cam_z'] = populate_gaze_info(
                            angle_struct_ins.le_pc_cam_mm,
                            is_valid_theta_phi)
                    frame_data_dict['label/rpc_cam_x'], \
                        frame_data_dict['label/rpc_cam_y'], \
                        frame_data_dict['label/rpc_cam_z'] = populate_gaze_info(
                            angle_struct_ins.re_pc_cam_mm,
                            is_valid_theta_phi)
                    frame_data_dict['label/mid_cam_x'], \
                        frame_data_dict['label/mid_cam_y'], \
                        frame_data_dict['label/mid_cam_z'] = populate_gaze_info(
                            angle_struct_ins.mid_eyes_cam_mm,
                            is_valid_theta_phi)
                    frame_data_dict['label/norm_face_hp_theta'] = populate_theta_phi(
                        angle_struct_ins.norm_face_hp_theta,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_face_hp_phi'] = populate_theta_phi(
                        angle_struct_ins.norm_face_hp_phi,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_face_gaze_theta'] = populate_theta_phi(
                        angle_struct_ins.norm_face_gaze_theta,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_face_gaze_phi'] = populate_theta_phi(
                        angle_struct_ins.norm_face_gaze_phi,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_leye_hp_theta'] = populate_theta_phi(
                        angle_struct_ins.norm_leye_hp_theta,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_leye_hp_phi'] = populate_theta_phi(
                        angle_struct_ins.norm_leye_hp_phi,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_leye_gaze_theta'] = populate_theta_phi(
                        angle_struct_ins.norm_leye_gaze_theta,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_leye_gaze_phi'] = populate_theta_phi(
                        angle_struct_ins.norm_leye_gaze_phi,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_reye_hp_theta'] = populate_theta_phi(
                        angle_struct_ins.norm_reye_hp_theta,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_reye_hp_phi'] = populate_theta_phi(
                        angle_struct_ins.norm_reye_hp_phi,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_reye_gaze_theta'] = populate_theta_phi(
                        angle_struct_ins.norm_reye_gaze_theta,
                        is_valid_theta_phi)
                    frame_data_dict['label/norm_reye_gaze_phi'] = populate_theta_phi(
                        angle_struct_ins.norm_reye_gaze_phi,
                        is_valid_theta_phi)
                    frame_data_dict['train/norm_per_oof'] = populate_head_norm_float(
                        angle_struct_ins.norm_per_oof, is_valid_theta_phi)
                    frame_data_dict['train/norm_facebb_x'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_face_bb, 0, is_valid_theta_phi)
                    frame_data_dict['train/norm_facebb_y'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_face_bb, 1, is_valid_theta_phi)
                    frame_data_dict['train/norm_facebb_w'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_face_bb, 2, is_valid_theta_phi)
                    frame_data_dict['train/norm_facebb_h'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_face_bb, 3, is_valid_theta_phi)
                    frame_data_dict['train/norm_leyebb_x'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_leye_bb, 0, is_valid_theta_phi)
                    frame_data_dict['train/norm_leyebb_y'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_leye_bb, 1, is_valid_theta_phi)
                    frame_data_dict['train/norm_leyebb_w'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_leye_bb, 2, is_valid_theta_phi)
                    frame_data_dict['train/norm_leyebb_h'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_leye_bb, 3, is_valid_theta_phi)
                    frame_data_dict['train/norm_reyebb_x'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_reye_bb, 0, is_valid_theta_phi)
                    frame_data_dict['train/norm_reyebb_y'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_reye_bb, 1, is_valid_theta_phi)
                    frame_data_dict['train/norm_reyebb_w'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_reye_bb, 2, is_valid_theta_phi)
                    frame_data_dict['train/norm_reyebb_h'] = populate_head_norm_bbinfo(
                        angle_struct_ins.norm_reye_bb, 3, is_valid_theta_phi)
                    frame_data_dict['train/norm_landmarks'] = populate_head_norm_listinfo(
                        angle_struct_ins.norm_landmarks, '2D', is_valid_theta_phi)
                    frame_data_dict['train/norm_frame_path'] = populate_head_norm_path(
                        angle_struct_ins.norm_frame_path, is_valid_theta_phi)
                    frame_data_dict['train/landmarks_3D'] = populate_head_norm_listinfo(
                        angle_struct_ins.landmarks_3D, '3D', is_valid_theta_phi)
                    frame_data_dict['train/norm_face_cnv_mat'] = populate_head_norm_listinfo(
                        angle_struct_ins.norm_face_cnv_mat, 'cnv_mat', is_valid_theta_phi)
                    frame_data_dict['train/norm_leye_cnv_mat'] = populate_head_norm_listinfo(
                        angle_struct_ins.norm_leye_cnv_mat, 'cnv_mat', is_valid_theta_phi)
                    frame_data_dict['train/norm_reye_cnv_mat'] = populate_head_norm_listinfo(
                        angle_struct_ins.norm_reye_cnv_mat, 'cnv_mat', is_valid_theta_phi)
                    frame_data_dict['label/face_cam_x'], frame_data_dict['label/face_cam_y'], \
                        frame_data_dict['label/face_cam_z'] = populate_gaze_info(
                        angle_struct_ins.face_cam_mm,
                        is_valid_theta_phi)

    def _prune_data(self):
        def _get_frame_num(frame):
            return int(frame.rsplit('_')[-1])

        prune_users = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
        set_is_kpi = is_kpi_set(self._set_id)

        num_available_frames = 0
        num_tfrecord_frames = 0
        num_insufficient_features_frames = 0

        if self._landmarks_path is not None:
            num_hp_frames = 0
            num_wink_frames = 0
            num_missing_from_fpe = 0
        for user in self._users.keys():
            if set_is_kpi and user in self._non_kpi_users:
                continue

            for region in self._users[user].keys():
                for frame in self._users[user][region].keys():
                    frame_data_dict = self._users[user][region][frame]
                    num_available_frames += 1

                    # Fill missing fields
                    if 'label/left_eye_status' not in frame_data_dict:
                        frame_data_dict['label/left_eye_status'] = EyeStatus.missing_eye_status
                    if 'label/right_eye_status' not in frame_data_dict:
                        frame_data_dict['label/right_eye_status'] = EyeStatus.missing_eye_status

                    set_frame_data_dict = set(frame_data_dict)
                    if self._landmarks_path is not None:
                        if not set(DataConverter.lm_pred_feature_to_type).issubset(
                                set_frame_data_dict):
                            if 'hp' in user:
                                num_hp_frames += 1
                            elif 'wink' in user:
                                num_wink_frames += 1
                            elif 'train/landmarks' not in set_frame_data_dict:
                                num_missing_from_fpe += 1

                            continue
                    else:
                        if not set(DataConverter.feature_to_type).issubset(
                                set(frame_data_dict)):
                            self._logger.add_warning(
                                'User {} frame {} do not have sufficient features'
                                .format(user, frame))
                            num_insufficient_features_frames += 1

                            continue

                    prune_users[user][region][frame] = frame_data_dict
                    num_tfrecord_frames += 1

        self._users = copy.deepcopy(prune_users)
        self._logger.add_info('Total num of available frames: {}'.format(num_available_frames))

        if self._landmarks_path is not None:
            self._logger.add_info('Total num of frames with head pose user: {}'.format(
                num_hp_frames))
            self._logger.add_info('Total num of frames with wink user: {}'.format(
                num_wink_frames))
            self._logger.add_info('Total num of frames without fpe predictions: {}'.format(
                num_missing_from_fpe))
        else:
            self._logger.add_info(
                'Total num of frames with insufficient features: {}'.format(
                    num_insufficient_features_frames))

        if self._paths.filtered_path:
            set_filtered_frames_file = os.path.join(
                    self._paths.filtered_path,
                    self._set_id + '.txt')
            num_filtered_frames = 0
            prune_filtered = defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict())))
            if os.path.exists(set_filtered_frames_file):
                with open(set_filtered_frames_file, 'r') as set_filtered_info:
                    for line in set_filtered_info:
                        line_split = line.split('/')

                        if len(self._paths.regions) == 1 and self._paths.regions[0] == '':
                            # There are no regions (on bench).
                            user_name = line_split[-2]
                            region_name = ''
                        else:
                            user_name = line_split[-3]
                            region_name = line_split[-2]
                        frame_name = get_file_name_noext(line_split[-1])
                        num_filtered_frames += 1

                        if (user_name in self._users.keys()
                                and frame_name in self._users[user_name][region_name].keys()):
                            prune_filtered[user_name][region_name][frame_name] = \
                                self._users[user_name][region_name][frame_name]
                            num_tfrecord_frames += 1
            else:
                self._logger.add_error('Could not find filtered frames file {}'
                                       .format(set_filtered_frames_file))

            self._users = copy.deepcopy(prune_filtered)
            self._logger.add_info('Total num of filtered frames: {}'.format(num_filtered_frames))

        if self._use_unique:
            prune_unique = defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict())))

            num_tfrecord_frames = 0
            for user in self._users.keys():
                for region in self._users[user].keys():
                    img_frame_dict = defaultdict(list)
                    for frame in self._users[user][region].keys():
                        base_img = frame.rsplit('_', 1)[0]
                        img_frame_dict[base_img].append(frame)
                    for _, val in img_frame_dict.items():
                        val.sort(key=_get_frame_num)
                    frames = [val[0] for _, val in img_frame_dict.items()]

                    for frame in frames:
                        prune_unique[user][region][frame] = self._users[user][region][frame]
                        num_tfrecord_frames += 1

            self._users = copy.deepcopy(prune_unique)
        self._logger.add_info('Total num of frames in tfrecord: {}'.format(num_tfrecord_frames))

    def _label_source(self):
        for user in self._users.keys():
            for region in self._users[user].keys():
                for frame in self._users[user][region].keys():
                    self._users[user][region][frame]['train/source'] = os.path.basename(
                        os.path.normpath(self._paths.info_source_path))
