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

"""Calculate gaze vectors and their theta phi angles."""

import abc
import errno
import os
import cv2
import numpy as np
from recordclass import recordclass
from nvidia_tao_tf1.cv.common.dataio.theta_phi_angle_utils import (
    calculate_reprojection_error,
    compute_PoR_from_gaze_vector,
    compute_PoR_from_theta_phi,
    compute_theta_phi_from_gaze_vector,
    normalizeEye,
    normalizeFace,
    normalizeFullFrame)
from nvidia_tao_tf1.cv.common.dataio.theta_phi_lm_utils import (
    AnthropometicPtsGenerator,
    PnPPtsGenerator,
    projectCamera2Image,
    projectObject2Camera)


'''
Terminologies:
WCS: World (object) coordinate system
CCS: Camera coordinate system
ICS: Image coordinate system
'''


def mkdir_p(new_path):
    """Makedir, making also non-existing parent dirs."""
    try:
        os.makedirs(new_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(new_path):
            pass
        else:
            raise


class ThetaPhiLandmarksGenerator(object):
    """Generate landmarks for theta phi calculations."""

    def __init__(
        self,
        landmarks_2D,
        occlusions,
        frame_width,
        frame_height,
        face_bbox
    ):
        """Initialize landmarks, face bounding box, landmarks, and frame info."""
        if landmarks_2D is None:
            raise ValueError('Non-existent landmarks, cannot be used to compute gaze vector')

        self._landmarks_2D = landmarks_2D
        self._occlusions = occlusions
        self._face_bbox = face_bbox
        if face_bbox:
            self._x1, self._y1, self._x2, self._y2 = face_bbox
        self._frame_width = frame_width
        self._frame_height = frame_height

    def get_landmarks_in_frame(self):
        """"Get landmarks in frame."""
        valid_landmarks = []

        n_landmarks = len(self._landmarks_2D)
        for coord_index in range(n_landmarks):
            coord = self._landmarks_2D[coord_index]
            x = coord[0]
            y = coord[1]

            if x < 0 or y < 0 or x > self._frame_width or y > self._frame_height:
                valid_landmarks.append([-1, -1])
            elif (
                self._face_bbox and
                self._occlusions[coord_index] == 1 and
                not (self._x1 <= x <= self._x2 and self._y1 <= y <= self._y2)
            ):
                valid_landmarks.append([-1, -1])
            else:
                valid_landmarks.append([x, y])

        return np.asarray(valid_landmarks, dtype=np.longdouble)


class ThetaPhiCalcStrategy(object):
    """Class for calculating gaze vectors and their theta phi angles."""

    __metaclass__ = abc.ABCMeta

    validation_threshold = 50  # in mm
    min_landmark_percentile_pnp = 50.0
    max_user_depth = 2500.0
    min_user_depth = 350.0
    reproj_err_threshold = 30.0
    hp_tp_diff_threshold_theta = 90 * np.pi/180.0
    hp_tp_diff_threshold_phi = 50 * np.pi/180.0

    AngleStruct = recordclass('AngleStruct', '''
        le_pc_cam_mm, re_pc_cam_mm,
        rot_mat, tvec,
        le_gaze_vec, re_gaze_vec,
        theta_ovr, phi_ovr,
        theta_le, phi_le,
        theta_re, phi_re,
        euler_angles, err_reproj_x2,
        head_pose_theta, head_pose_phi,
        mid_eyes_cam_mm, mid_gaze_vec,
        theta_mid, phi_mid,
        norm_face_gaze_theta, norm_face_gaze_phi,
        norm_face_hp_theta, norm_face_hp_phi,
        norm_leye_gaze_theta, norm_leye_gaze_phi,
        norm_leye_hp_theta, norm_leye_hp_phi,
        norm_reye_gaze_theta, norm_reye_gaze_phi,
        norm_reye_hp_theta, norm_reye_hp_phi,
        norm_face_bb, norm_leye_bb, norm_reye_bb,
        norm_landmarks, norm_per_oof,
        norm_frame_path, landmarks_3D,
        norm_face_cnv_mat, norm_leye_cnv_mat,
        norm_reye_cnv_mat, face_cam_mm''')

    empty_angle_struct = AngleStruct(
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None)

    def __init__(self, logger, gt_cam_mm, camera_matrix, distortion_coeffs, R, T, landmarks_2D):
        """Initialize camera parameters and landmarks."""
        self._logger = logger
        self._gt_cam_mm = gt_cam_mm
        self._camera_matrix = camera_matrix
        self._distortion_coeffs = distortion_coeffs
        self._R = R
        self._T = T
        self._landmarks_2D = landmarks_2D

        self._max_theta_phi_val_err = None
        self._max_gaze_vec_val_err = None

    @abc.abstractmethod
    def _compute_gaze_angles_theta_phi(self):
        pass

    def _validate_overall_gaze(
        self,
        gaze_vec,
        phi,
        theta,
        err_reproj_x2,
        head_pose_phi,
        head_pose_theta
    ):
        is_valid_gaze = True
        is_invalid_angle_params = (
            gaze_vec is None or
            phi is None or theta is None or
            err_reproj_x2 is None or
            head_pose_phi is None or head_pose_theta is None)

        if is_invalid_angle_params:
            self._logger.add_warning('Landmark problems. Discard sample.')
            is_valid_gaze = False
        else:

            if phi >= 1.8 or phi <= -1.8:
                self._logger.add_warning(
                    'Unexpected gaze yaw or pitch: {}, {}. Discard sample.'.format(phi, theta))
                is_valid_gaze = False

            if err_reproj_x2 > self.reproj_err_threshold:
                self._logger.add_warning(
                    'High PnP reprojection error: {}. Discard sample'.format(err_reproj_x2))
                is_valid_gaze = False

            if (phi > 0 and head_pose_phi < -1 * phi) or (phi < 0 and head_pose_phi > -1 * phi):
                if abs(phi + head_pose_phi) > 0.1:
                    self._logger.add_warning(
                        'Opposite hp and gaze yaw: {}, {}'.format(phi, head_pose_phi))

                if abs(phi - head_pose_phi) > self.hp_tp_diff_threshold_phi:
                    self._logger.add_warning(
                        '''High yaw difference and opposite directions: {}, {}, {}, {}.
                        Discard sample.'''.format(phi, head_pose_phi, theta, head_pose_theta))
                    is_valid_gaze = False

            if abs(phi - head_pose_phi) > 2 * self.hp_tp_diff_threshold_phi:
                self._logger.add_warning(
                    '''High yaw difference between head pose-based gaze and real gaze: {}, {}, {}, {}.
                    Discard sample.'''.format(phi, head_pose_phi, theta, head_pose_theta))
                is_valid_gaze = False
                return is_valid_gaze

            if abs(theta - head_pose_theta) > self.hp_tp_diff_threshold_theta:
                self._logger.add_warning(
                    '''High pitch difference between head pose-based gaze and real gaze: {}, {}, {}.
                    Discard sample.'''.format(abs(theta - head_pose_theta), theta, head_pose_theta))
                is_valid_gaze = False

        return is_valid_gaze

    @abc.abstractmethod
    def _extract_error(self, info):
        pass

    def _is_valid_sample(self):
        if self._max_gaze_vec_val_err <= self.validation_threshold \
            and self._max_theta_phi_val_err <= self.validation_threshold \
                and abs(self._max_theta_phi_val_err - self._max_gaze_vec_val_err) <= 3.0:
            return True
        self._logger.add_warning('Sample discarded due to high validation error')
        self._logger.add_warning('Errors theta-phi: {} gaze_vec: {}'.format(
            self._max_theta_phi_val_err,
            self._max_gaze_vec_val_err))
        return False

    @abc.abstractmethod
    def get_gaze_angles_theta_phi(self):
        """Abstract method for getting gaze angles info."""
        pass


class NoPupilThetaPhiStrategy(ThetaPhiCalcStrategy):
    """Class for generating gaze angles without pupil landmarks."""

    def __init__(self, logger, gt_cam_mm, camera_matrix, distortion_coeffs, R, T, landmarks_2D,
                 frame_path, labels_source_path, norm_folder_name, save_images):
        """Initialize camera parameters and landmarks."""
        super(NoPupilThetaPhiStrategy, self).__init__(
            logger,
            gt_cam_mm,
            camera_matrix,
            distortion_coeffs,
            R,
            T,
            landmarks_2D)
        self._labels_source_path = labels_source_path
        self._orig_frame_path = frame_path
        self._anthro_pts = AnthropometicPtsGenerator()
        self._pnp_pts = PnPPtsGenerator(camera_matrix, distortion_coeffs)
        self._norm_folder_name = norm_folder_name
        self._save_images = save_images

    def get_gaze_angles_theta_phi(self):
        """Get gaze angles and validity of sample."""
        info = self._compute_gaze_angles_theta_phi()
        if not self._validate_overall_gaze(
            info.mid_gaze_vec,
            info.phi_mid,
            info.theta_mid,
            info.err_reproj_x2,
            info.head_pose_phi,
            info.head_pose_theta
        ):
            is_valid_sample = False
        else:
            self._extract_error(info)
            is_valid_sample = self._is_valid_sample()

        return is_valid_sample, info

    def _extract_error(self, info):
        PoR_x, PoR_y, PoR_z = compute_PoR_from_theta_phi(
            info.theta_mid,
            info.phi_mid,
            info.mid_eyes_cam_mm,
            self._R,
            self._T)
        por_arr = np.asarray([PoR_x, PoR_y, PoR_z])
        self._max_theta_phi_val_err = np.sqrt(np.sum(np.power(por_arr - self._gt_cam_mm, 2)))

        PoR_x, PoR_y, PoR_z = compute_PoR_from_gaze_vector(
            info.mid_gaze_vec,
            info.mid_eyes_cam_mm,
            self._R,
            self._T)
        por_arr = np.asarray([PoR_x, PoR_y, PoR_z])
        self._max_gaze_vec_val_err = np.sqrt(np.sum(np.power(por_arr - self._gt_cam_mm, 2)))

    def _compute_gaze_angles_theta_phi(self):
        landmarks_2D_final, landmarks_3D_final, perc_occ_landmarks = \
            self._anthro_pts.get_selected_landmarks(
                self._landmarks_2D,
                self._anthro_pts.landmarks_2D_indices_selected,
                self._anthro_pts.get_pnp_landmarks())

        # Compute PnP between the generic 3D face model landmarks (WCS) and 2D landmarks (ICS)
        if perc_occ_landmarks < self.min_landmark_percentile_pnp:
            # Rotation and translation vectors for 3D-to-2D transformation
            _, rvec, tvec = self._pnp_pts.compute_EPnP(
                landmarks_3D_final, landmarks_2D_final)
        else:
            self._logger.add_warning(
                'Too many occluded landmarks (%' + str(perc_occ_landmarks) +
                '). SolvePnP might not be reliable. Discard sample.')
            return self.empty_angle_struct

        landmarks_2D_final, landmarks_3D_final, perc_occ_landmarks = \
            self._anthro_pts.get_selected_landmarks(
                self._landmarks_2D,
                self._anthro_pts.landmarks_2D_indices_expr_inv,
                self._anthro_pts.get_robust_expr_var_landmarks())

        if perc_occ_landmarks < self.min_landmark_percentile_pnp:
            _, rvec, tvec = self._pnp_pts.compute_PnP_Iterative(
                landmarks_3D_final, landmarks_2D_final, rvec, tvec)
        else:
            self._logger.add_warning(
                'Too many occluded landmarks (%' + str(perc_occ_landmarks) +
                '). SolvePnP might not be reliable. Discard sample.')
            return self.empty_angle_struct

        # Convert rotation vector into a rotation matrix
        rot_mat = cv2.Rodrigues(rvec)[0]

        # Concatenate translation vector with rotation matrix
        proj_mat = np.hstack((rot_mat, tvec))

        # use cv with distortion coeffs
        landmarks_2D_reproj_2, _ = cv2.projectPoints(
            PnPPtsGenerator.get_cv_array(landmarks_3D_final),
            rvec,
            tvec,
            self._camera_matrix,
            self._distortion_coeffs)

        err_reproj_x2 = calculate_reprojection_error(
            np.reshape(landmarks_2D_final, (len(landmarks_2D_final), 2)),
            np.reshape(landmarks_2D_reproj_2, (len(landmarks_2D_reproj_2), 2)))

        # Head pose-based direction vector
        Zv = rot_mat[:, 2]
        head_pose_theta, head_pose_phi = \
            compute_theta_phi_from_gaze_vector(-Zv[0], -Zv[1], -Zv[2])

        # Head pose euler angles in degrees
        euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]
        if euler_angles[0] < -90:
            euler_angles[0] = -(euler_angles[0] + 180)  # correct pitch
        elif euler_angles[0] > 90:
            euler_angles[0] = -(euler_angles[0] - 180)  # correct pitch

        euler_angles[1] = -euler_angles[1]  # correct the sign of yaw
        if euler_angles[2] < -90:
            euler_angles[2] = -(euler_angles[2] + 180)  # correct roll
        elif euler_angles[2] > 90:
            euler_angles[2] = -(euler_angles[2] - 180)  # correct roll

        face_center_obj = self._anthro_pts.get_face_center()
        face_cam_mm = projectObject2Camera(face_center_obj, rot_mat, tvec)
        face_gaze_vec = self._gt_cam_mm - face_cam_mm
        face_gv_mag = np.sqrt(face_gaze_vec[0] ** 2 + face_gaze_vec[1] ** 2 + face_gaze_vec[2] ** 2)
        face_gaze_vec = face_gaze_vec / face_gv_mag

        le_center, re_center = self._anthro_pts.get_eye_centers()

        mid_eyes_obj = (re_center + le_center) / 2.0
        mid_eyes_cam_mm = projectObject2Camera(mid_eyes_obj, rot_mat, tvec)
        if mid_eyes_cam_mm[2] > self.max_user_depth or mid_eyes_cam_mm[2] < self.min_user_depth:
            self._logger.add_warning('Mid eye cam coords incorrect: {}. Discard sample'.format(
                mid_eyes_cam_mm))
        elif np.any(euler_angles > 90) or np.any(euler_angles < -90):
            self._logger.add_warning('''Head pose angle range incorrect. Discard sample.
                Euler angles - Pitch: {}, Yaw: {}, Roll: {}'''.format(
                euler_angles[0], euler_angles[1], euler_angles[2]))
            return self.empty_angle_struct

        mid_gaze_vec = self._gt_cam_mm - mid_eyes_cam_mm
        mid_gaze_vec = mid_gaze_vec / np.sqrt(np.sum(np.power(mid_gaze_vec, 2)))

        theta_mid, phi_mid = compute_theta_phi_from_gaze_vector(mid_gaze_vec[0],
                                                                mid_gaze_vec[1],
                                                                mid_gaze_vec[2])

        leye_cam_mm = projectObject2Camera(le_center, rot_mat, tvec)
        leye_gaze_vec = self._gt_cam_mm - leye_cam_mm
        leye_gaze_vec /= np.sqrt(np.sum(np.power(leye_gaze_vec, 2)))
        leye_theta, leye_phi = compute_theta_phi_from_gaze_vector(leye_gaze_vec[0],
                                                                  leye_gaze_vec[1],
                                                                  leye_gaze_vec[2])

        reye_cam_mm = projectObject2Camera(re_center, rot_mat, tvec)
        reye_gaze_vec = self._gt_cam_mm - reye_cam_mm
        reye_gaze_vec /= np.sqrt(np.sum(np.power(reye_gaze_vec, 2)))
        reye_theta, reye_phi = compute_theta_phi_from_gaze_vector(reye_gaze_vec[0],
                                                                  reye_gaze_vec[1],
                                                                  reye_gaze_vec[2])

        theta_ovr = (leye_theta + reye_theta)/2.0
        phi_ovr = (leye_phi + reye_phi)/2.0

        # Get 3D landmarks according to current 58-pt landmark model
        # 38 of 58 3D landmarks matches with 2D landmarks, therefore, considered
        no_of_3D_landmarks = 38
        landmarks_3D = -1.0 * np.ones((no_of_3D_landmarks, 3), dtype=np.float32)
        landmarks_obj_3D = self._anthro_pts.get_landmarks_export_3D()
        for ind in range(no_of_3D_landmarks):
            obj_mm = np.reshape(landmarks_obj_3D[ind, :], (1, 3))
            cam_mm = projectObject2Camera(obj_mm, rot_mat, tvec)
            landmarks_3D[ind] = cam_mm.transpose()
        landmarks_3D = landmarks_3D.reshape(-1)

        # Full Frame Normalization
        frame_gray = cv2.imread(self._orig_frame_path, 0)

        lec_px = projectCamera2Image(leye_cam_mm, self._camera_matrix)
        rec_px = projectCamera2Image(reye_cam_mm, self._camera_matrix)
        ec_pxs = np.zeros((2, 2), dtype=np.float32)
        ec_pxs[0][0] = lec_px[0]
        ec_pxs[0][1] = lec_px[1]
        ec_pxs[1][0] = rec_px[0]
        ec_pxs[1][1] = rec_px[1]

        method = 'modified'

        # For 448 x 448, scale_factor is 2
        scale_factor = 2.0
        norm_frame_warped, norm_face_gaze_theta, norm_face_gaze_phi, norm_face_hp_theta, \
            norm_face_hp_phi, norm_face_bb, norm_leye_bb, norm_reye_bb, normlandmarks, \
            norm_face_cnv_mat = normalizeFullFrame(frame_gray, face_cam_mm, ec_pxs, rot_mat,
                                                   face_gaze_vec, self._landmarks_2D,
                                                   self._camera_matrix, self._distortion_coeffs,
                                                   method, scale_factor)
        if norm_frame_warped is None:
            print('Bad normalization, warped image is flipped. Discard sample:',
                  self._orig_frame_path)
            return self.AngleStruct(
                leye_cam_mm, reye_cam_mm,
                rot_mat, tvec,
                leye_gaze_vec, reye_gaze_vec,
                theta_ovr, phi_ovr,
                leye_theta, leye_phi,
                reye_theta, reye_phi,
                euler_angles, err_reproj_x2,
                head_pose_theta, head_pose_phi,
                mid_eyes_cam_mm, mid_gaze_vec,
                theta_mid, phi_mid,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None, None,
                None, None,
                None, None,
                None, None,
                None, None)

        invalid_landmark_ind = np.argwhere(self._landmarks_2D.reshape(-1) == -1)
        no_of_2D_landmarks = 104
        norm_landmarks = -1.0 * np.ones((no_of_2D_landmarks, 2), dtype=np.float32)
        norm_landmarks[:normlandmarks.shape[0]] = normlandmarks
        norm_landmarks = norm_landmarks.reshape(-1)
        norm_landmarks[invalid_landmark_ind] = -1

        # Face/Eyes Normalization
        imageWidth = 224
        imageHeight = 224
        norm_face_warped, _, _, _, _, norm_per_oof, _ = normalizeFace(frame_gray, face_cam_mm,
                                                                      rot_mat, face_gaze_vec,
                                                                      self._camera_matrix,
                                                                      self._distortion_coeffs,
                                                                      method, imageWidth,
                                                                      imageHeight)
        norm_face_cnv_mat = norm_face_cnv_mat.reshape(-1)

        imageWidth = 120
        # For rectangular eye, use imageHeight = int(imageWidth * 36.0 / 60.0)
        imageHeight = imageWidth
        norm_leye_warped, norm_leye_gaze_theta, norm_leye_gaze_phi, norm_leye_hp_theta, \
            norm_leye_hp_phi, norm_leye_cnv_mat = normalizeEye(frame_gray, leye_cam_mm, rot_mat,
                                                               leye_gaze_vec, self._camera_matrix,
                                                               self._distortion_coeffs, method,
                                                               imageWidth, imageHeight)
        norm_leye_cnv_mat = norm_leye_cnv_mat.reshape(-1)

        norm_reye_warped, norm_reye_gaze_theta, norm_reye_gaze_phi, norm_reye_hp_theta, \
            norm_reye_hp_phi, norm_reye_cnv_mat = normalizeEye(frame_gray, reye_cam_mm, rot_mat,
                                                               reye_gaze_vec, self._camera_matrix,
                                                               self._distortion_coeffs, method,
                                                               imageWidth, imageHeight)
        norm_reye_cnv_mat = norm_reye_cnv_mat.reshape(-1)

        folder_path = ''
        for tmp in self._orig_frame_path.split('/')[:-1]:
            folder_path += (tmp + '/')

        frame_name = self._orig_frame_path.split('/')[-1]

        norm_data_name = self._norm_folder_name + '_' + \
            self._labels_source_path.split("/")[-1] + '/'

        if 'pngData' in folder_path:
            data_folder = 'pngData'
        elif 'Data' in folder_path:
            data_folder = 'Data'
        else:
            print('Cosmos path name and data folder is not recognized:', folder_path,
                  self._orig_frame_path, head_pose_phi, phi_mid, norm_face_gaze_phi)
            return self.AngleStruct(
                leye_cam_mm, reye_cam_mm,
                rot_mat, tvec,
                leye_gaze_vec, reye_gaze_vec,
                theta_ovr, phi_ovr,
                leye_theta, leye_phi,
                reye_theta, reye_phi,
                euler_angles, err_reproj_x2,
                head_pose_theta, head_pose_phi,
                mid_eyes_cam_mm, mid_gaze_vec,
                theta_mid, phi_mid,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None, None,
                None, None,
                None, None,
                None, None,
                None, None)

        new_suffix = norm_data_name + 'frame'

        # Replace only last occurrence (reverse, change first occurence, reverse back)
        norm_frame_path = (self._orig_frame_path[::-1].
                           replace(data_folder[::-1], new_suffix[::-1], 1))[::-1]

        if self._save_images:
            norm_frame_folder = folder_path[::-1]. \
                replace(data_folder[::-1], new_suffix[::-1], 1)[::-1]

            mkdir_p(norm_frame_folder)

            new_suffix = norm_data_name + 'face'
            norm_face_folder = folder_path[::-1]. \
                replace(data_folder[::-1], new_suffix[::-1], 1)[::-1]

            mkdir_p(norm_face_folder)

            new_suffix = norm_data_name + 'leye'
            norm_leye_folder = folder_path[::-1]. \
                replace(data_folder[::-1], new_suffix[::-1], 1)[::-1]

            mkdir_p(norm_leye_folder)

            new_suffix = norm_data_name + 'reye'
            norm_reye_folder = folder_path[::-1]. \
                replace(data_folder[::-1], new_suffix[::-1], 1)[::-1]

            mkdir_p(norm_reye_folder)

            # Write-out normalized/warped images
            new_image_path = norm_frame_folder + frame_name
            cv2.imwrite(new_image_path, norm_frame_warped)

            new_image_path = norm_face_folder + frame_name
            cv2.imwrite(new_image_path, norm_face_warped)

            new_image_path = norm_leye_folder + frame_name
            cv2.imwrite(new_image_path, norm_leye_warped)

            new_image_path = norm_reye_folder + frame_name
            cv2.imwrite(new_image_path, norm_reye_warped)

        return self.AngleStruct(
            leye_cam_mm, reye_cam_mm,
            rot_mat, tvec,
            leye_gaze_vec, reye_gaze_vec,
            theta_ovr, phi_ovr,
            leye_theta, leye_phi,
            reye_theta, reye_phi,
            euler_angles, err_reproj_x2,
            head_pose_theta, head_pose_phi,
            mid_eyes_cam_mm, mid_gaze_vec,
            theta_mid, phi_mid,
            norm_face_gaze_theta, norm_face_gaze_phi,
            norm_face_hp_theta, norm_face_hp_phi,
            norm_leye_gaze_theta, norm_leye_gaze_phi,
            norm_leye_hp_theta, norm_leye_hp_phi,
            norm_reye_gaze_theta, norm_reye_gaze_phi,
            norm_reye_hp_theta, norm_reye_hp_phi,
            norm_face_bb, norm_leye_bb, norm_reye_bb,
            norm_landmarks, norm_per_oof,
            norm_frame_path, landmarks_3D,
            norm_face_cnv_mat, norm_leye_cnv_mat,
            norm_reye_cnv_mat, face_cam_mm)


class CustomNormalizeData(ThetaPhiCalcStrategy):
    """Class for generating gaze angles without pupil landmarks."""

    def __init__(self, logger, camera_matrix, distortion_coeffs, R, T, landmarks_2D,
                 frame_path, norm_folder_name, save_images, data_root_path):
        """Initialize camera parameters and landmarks."""
        super(CustomNormalizeData, self).__init__(
              logger=logger,
              gt_cam_mm=np.zeros((3, 1), dtype=np.longdouble),
              camera_matrix=camera_matrix,
              distortion_coeffs=distortion_coeffs,
              R=R,
              T=T,
              landmarks_2D=landmarks_2D)
        self._orig_frame_path = frame_path
        self._anthro_pts = AnthropometicPtsGenerator()
        self._pnp_pts = PnPPtsGenerator(camera_matrix, distortion_coeffs)
        self._norm_folder_name = norm_folder_name
        self._save_images = save_images
        self._data_root_path = data_root_path

    def get_normalized_data(self):
        """Get normalized data for the given sample."""
        info = self._compute_gaze_angles_theta_phi()
        return info

    def _extract_error(self, info):
        PoR_x, PoR_y, PoR_z = compute_PoR_from_theta_phi(
            info.theta_mid,
            info.phi_mid,
            info.mid_eyes_cam_mm,
            self._R,
            self._T)
        por_arr = np.asarray([PoR_x, PoR_y, PoR_z])
        self._max_theta_phi_val_err = np.sqrt(np.sum(np.power(por_arr - self._gt_cam_mm, 2)))

        PoR_x, PoR_y, PoR_z = compute_PoR_from_gaze_vector(
            info.mid_gaze_vec,
            info.mid_eyes_cam_mm,
            self._R,
            self._T)
        por_arr = np.asarray([PoR_x, PoR_y, PoR_z])
        self._max_gaze_vec_val_err = np.sqrt(np.sum(np.power(por_arr - self._gt_cam_mm, 2)))

    def _compute_gaze_angles_theta_phi(self):
        landmarks_2D_final, landmarks_3D_final, perc_occ_landmarks = \
            self._anthro_pts.get_selected_landmarks(
                self._landmarks_2D,
                self._anthro_pts.landmarks_2D_indices_selected,
                self._anthro_pts.get_pnp_landmarks())

        # Compute PnP between the generic 3D face model landmarks (WCS) and 2D landmarks (ICS)
        if perc_occ_landmarks < self.min_landmark_percentile_pnp:
            # Rotation and translation vectors for 3D-to-2D transformation
            _, rvec, tvec = self._pnp_pts.compute_EPnP(
                landmarks_3D_final, landmarks_2D_final)
        else:
            self._logger.add_warning(
                'Too many occluded landmarks (%' + str(perc_occ_landmarks) +
                '). SolvePnP might not be reliable. Discard sample.')
            return self.empty_angle_struct

        landmarks_2D_final, landmarks_3D_final, perc_occ_landmarks = \
            self._anthro_pts.get_selected_landmarks(
                self._landmarks_2D,
                self._anthro_pts.landmarks_2D_indices_expr_inv,
                self._anthro_pts.get_robust_expr_var_landmarks())

        if perc_occ_landmarks < self.min_landmark_percentile_pnp:
            _, rvec, tvec = self._pnp_pts.compute_PnP_Iterative(
                landmarks_3D_final, landmarks_2D_final, rvec, tvec)
        else:
            self._logger.add_warning(
                'Too many occluded landmarks (%' + str(perc_occ_landmarks) +
                '). SolvePnP might not be reliable. Discard sample.')
            return self.empty_angle_struct

        # Convert rotation vector into a rotation matrix
        rot_mat = cv2.Rodrigues(rvec)[0]

        # Concatenate translation vector with rotation matrix
        proj_mat = np.hstack((rot_mat, tvec))

        # use cv with distortion coeffs
        landmarks_2D_reproj_2, _ = cv2.projectPoints(
            PnPPtsGenerator.get_cv_array(landmarks_3D_final),
            rvec,
            tvec,
            self._camera_matrix,
            self._distortion_coeffs)

        err_reproj_x2 = calculate_reprojection_error(
            np.reshape(landmarks_2D_final, (len(landmarks_2D_final), 2)),
            np.reshape(landmarks_2D_reproj_2, (len(landmarks_2D_reproj_2), 2)))

        # Head pose-based direction vector
        Zv = rot_mat[:, 2]
        head_pose_theta, head_pose_phi = \
            compute_theta_phi_from_gaze_vector(-Zv[0], -Zv[1], -Zv[2])

        # Head pose euler angles in degrees
        euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]
        if euler_angles[0] < -90:
            euler_angles[0] = -(euler_angles[0] + 180)  # correct pitch
        elif euler_angles[0] > 90:
            euler_angles[0] = -(euler_angles[0] - 180)  # correct pitch

        euler_angles[1] = -euler_angles[1]  # correct the sign of yaw
        if euler_angles[2] < -90:
            euler_angles[2] = -(euler_angles[2] + 180)  # correct roll
        elif euler_angles[2] > 90:
            euler_angles[2] = -(euler_angles[2] - 180)  # correct roll

        face_center_obj = self._anthro_pts.get_face_center()
        face_cam_mm = projectObject2Camera(face_center_obj, rot_mat, tvec)
        face_gaze_vec = self._gt_cam_mm - face_cam_mm
        face_gv_mag = np.sqrt(face_gaze_vec[0] ** 2 + face_gaze_vec[1] ** 2 + face_gaze_vec[2] ** 2)
        face_gaze_vec = face_gaze_vec / face_gv_mag

        le_center, re_center = self._anthro_pts.get_eye_centers()

        mid_eyes_obj = (re_center + le_center) / 2.0
        mid_eyes_cam_mm = projectObject2Camera(mid_eyes_obj, rot_mat, tvec)
        if mid_eyes_cam_mm[2] > self.max_user_depth or mid_eyes_cam_mm[2] < self.min_user_depth:
            self._logger.add_warning('Mid eye cam coords incorrect: {}. Discard sample'.format(
                mid_eyes_cam_mm))
        elif np.any(euler_angles > 90) or np.any(euler_angles < -90):
            self._logger.add_warning('''Head pose angle range incorrect. Discard sample.
                Euler angles - Pitch: {}, Yaw: {}, Roll: {}'''.format(
                euler_angles[0], euler_angles[1], euler_angles[2]))
            return self.empty_angle_struct

        mid_gaze_vec = self._gt_cam_mm - mid_eyes_cam_mm
        mid_gaze_vec = mid_gaze_vec / np.sqrt(np.sum(np.power(mid_gaze_vec, 2)))

        theta_mid, phi_mid = compute_theta_phi_from_gaze_vector(mid_gaze_vec[0],
                                                                mid_gaze_vec[1],
                                                                mid_gaze_vec[2])

        leye_cam_mm = projectObject2Camera(le_center, rot_mat, tvec)
        leye_gaze_vec = self._gt_cam_mm - leye_cam_mm
        leye_gaze_vec /= np.sqrt(np.sum(np.power(leye_gaze_vec, 2)))
        leye_theta, leye_phi = compute_theta_phi_from_gaze_vector(leye_gaze_vec[0],
                                                                  leye_gaze_vec[1],
                                                                  leye_gaze_vec[2])

        reye_cam_mm = projectObject2Camera(re_center, rot_mat, tvec)
        reye_gaze_vec = self._gt_cam_mm - reye_cam_mm
        reye_gaze_vec /= np.sqrt(np.sum(np.power(reye_gaze_vec, 2)))
        reye_theta, reye_phi = compute_theta_phi_from_gaze_vector(reye_gaze_vec[0],
                                                                  reye_gaze_vec[1],
                                                                  reye_gaze_vec[2])

        theta_ovr = (leye_theta + reye_theta)/2.0
        phi_ovr = (leye_phi + reye_phi)/2.0

        # Get 3D landmarks according to current 58-pt landmark model
        # 38 of 58 3D landmarks matches with 2D landmarks, therefore, considered
        no_of_3D_landmarks = 38
        landmarks_3D = -1.0 * np.ones((no_of_3D_landmarks, 3), dtype=np.float32)
        landmarks_obj_3D = self._anthro_pts.get_landmarks_export_3D()
        for ind in range(no_of_3D_landmarks):
            obj_mm = np.reshape(landmarks_obj_3D[ind, :], (1, 3))
            cam_mm = projectObject2Camera(obj_mm, rot_mat, tvec)
            landmarks_3D[ind] = cam_mm.transpose()
        landmarks_3D = landmarks_3D.reshape(-1)

        # Full Frame Normalization
        frame_gray = cv2.imread(self._orig_frame_path, 0)

        lec_px = projectCamera2Image(leye_cam_mm, self._camera_matrix)
        rec_px = projectCamera2Image(reye_cam_mm, self._camera_matrix)
        ec_pxs = np.zeros((2, 2), dtype=np.float32)
        ec_pxs[0][0] = lec_px[0]
        ec_pxs[0][1] = lec_px[1]
        ec_pxs[1][0] = rec_px[0]
        ec_pxs[1][1] = rec_px[1]

        method = 'modified'

        # For 448 x 448, scale_factor is 2
        scale_factor = 2.0
        norm_frame_warped, norm_face_gaze_theta, norm_face_gaze_phi, norm_face_hp_theta, \
            norm_face_hp_phi, norm_face_bb, norm_leye_bb, norm_reye_bb, normlandmarks, \
            norm_face_cnv_mat = normalizeFullFrame(frame_gray, face_cam_mm, ec_pxs, rot_mat,
                                                   face_gaze_vec, self._landmarks_2D,
                                                   self._camera_matrix, self._distortion_coeffs,
                                                   method, scale_factor)

        if norm_frame_warped is None:
            print('Bad normalization, warped image is flipped. Discard sample:',
                  self._orig_frame_path)
            return self.AngleStruct(
                leye_cam_mm, reye_cam_mm,
                rot_mat, tvec,
                leye_gaze_vec, reye_gaze_vec,
                theta_ovr, phi_ovr,
                leye_theta, leye_phi,
                reye_theta, reye_phi,
                euler_angles, err_reproj_x2,
                head_pose_theta, head_pose_phi,
                mid_eyes_cam_mm, mid_gaze_vec,
                theta_mid, phi_mid,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None, None,
                None, None,
                None, None,
                None, None,
                None, None)

        invalid_landmark_ind = np.argwhere(self._landmarks_2D.reshape(-1) == -1)
        no_of_2D_landmarks = 104
        norm_landmarks = -1.0 * np.ones((no_of_2D_landmarks, 2), dtype=np.float32)
        norm_landmarks[:normlandmarks.shape[0]] = normlandmarks
        norm_landmarks = norm_landmarks.reshape(-1)
        norm_landmarks[invalid_landmark_ind] = -1

        # Face/Eyes Normalization
        imageWidth = 224
        imageHeight = 224
        norm_face_warped, _, _, _, _, norm_per_oof, _ = normalizeFace(frame_gray, face_cam_mm,
                                                                      rot_mat, face_gaze_vec,
                                                                      self._camera_matrix,
                                                                      self._distortion_coeffs,
                                                                      method, imageWidth,
                                                                      imageHeight)
        norm_face_cnv_mat = norm_face_cnv_mat.reshape(-1)

        imageWidth = 120
        # For rectangular eye, use imageHeight = int(imageWidth * 36.0 / 60.0)
        imageHeight = imageWidth
        norm_leye_warped, norm_leye_gaze_theta, norm_leye_gaze_phi, norm_leye_hp_theta, \
            norm_leye_hp_phi, norm_leye_cnv_mat = normalizeEye(frame_gray, leye_cam_mm, rot_mat,
                                                               leye_gaze_vec, self._camera_matrix,
                                                               self._distortion_coeffs, method,
                                                               imageWidth, imageHeight)
        norm_leye_cnv_mat = norm_leye_cnv_mat.reshape(-1)

        norm_reye_warped, norm_reye_gaze_theta, norm_reye_gaze_phi, norm_reye_hp_theta, \
            norm_reye_hp_phi, norm_reye_cnv_mat = normalizeEye(frame_gray, reye_cam_mm, rot_mat,
                                                               reye_gaze_vec, self._camera_matrix,
                                                               self._distortion_coeffs, method,
                                                               imageWidth, imageHeight)
        norm_reye_cnv_mat = norm_reye_cnv_mat.reshape(-1)

        folder_path = ''
        for tmp in self._orig_frame_path.split('/')[:-1]:
            folder_path += (tmp + '/')

        frame_name = self._orig_frame_path.split('/')[-1]

        # Replace only last occurrence (reverse, change first occurence, reverse back)
        norm_frame_path = os.path.join(self._data_root_path,
                                       self._norm_folder_name,
                                       'frame',
                                       frame_name)

        if self._save_images:
            norm_frame_folder = os.path.join(self._data_root_path,
                                             self._norm_folder_name,
                                             'frame')

            mkdir_p(norm_frame_folder)

            # new_suffix = norm_data_name + 'face'
            norm_face_folder = os.path.join(self._data_root_path,
                                            self._norm_folder_name,
                                            'face')

            mkdir_p(norm_face_folder)

            # new_suffix = norm_data_name + 'leye'
            norm_leye_folder = os.path.join(self._data_root_path,
                                            self._norm_folder_name,
                                            'leye')

            mkdir_p(norm_leye_folder)

            # new_suffix = norm_data_name + 'reye'
            norm_reye_folder = os.path.join(self._data_root_path,
                                            self._norm_folder_name,
                                            'reye')

            mkdir_p(norm_reye_folder)

            # Write-out normalized/warped images
            new_image_path = os.path.join(norm_frame_folder, frame_name)
            cv2.imwrite(new_image_path, norm_frame_warped)

            new_image_path = os.path.join(norm_face_folder, frame_name)
            cv2.imwrite(new_image_path, norm_face_warped)

            new_image_path = os.path.join(norm_leye_folder, frame_name)
            cv2.imwrite(new_image_path, norm_leye_warped)

            new_image_path = os.path.join(norm_reye_folder, frame_name)
            cv2.imwrite(new_image_path, norm_reye_warped)

        return self.AngleStruct(
            leye_cam_mm, reye_cam_mm,
            rot_mat, tvec,
            leye_gaze_vec, reye_gaze_vec,
            theta_ovr, phi_ovr,
            leye_theta, leye_phi,
            reye_theta, reye_phi,
            euler_angles, err_reproj_x2,
            head_pose_theta, head_pose_phi,
            mid_eyes_cam_mm, mid_gaze_vec,
            theta_mid, phi_mid,
            norm_face_gaze_theta, norm_face_gaze_phi,
            norm_face_hp_theta, norm_face_hp_phi,
            norm_leye_gaze_theta, norm_leye_gaze_phi,
            norm_leye_hp_theta, norm_leye_hp_phi,
            norm_reye_gaze_theta, norm_reye_gaze_phi,
            norm_reye_hp_theta, norm_reye_hp_phi,
            norm_face_bb, norm_leye_bb, norm_reye_bb,
            norm_landmarks, norm_per_oof,
            norm_frame_path, landmarks_3D,
            norm_face_cnv_mat, norm_leye_cnv_mat,
            norm_reye_cnv_mat, face_cam_mm)
