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

"""Test common functions."""

import unittest
import numpy as np
from nvidia_tao_tf1.cv.common.dataio.theta_phi_angle_utils import (
    normalizeEye,
    normalizeFace,
    normalizeFullFrame
    )
from nvidia_tao_tf1.cv.common.dataio.theta_phi_lm_utils import (
    AnthropometicPtsGenerator,
    projectObject2Camera)


class ThetaPhiAngleUtilsTest(unittest.TestCase):
    """Test PipelineReporter."""

    def setUp(self):

        # Create black blank image
        self.frame = np.zeros((800, 1280, 1), np.uint8)

        # Params from data collected from germany-1-gaze-1 config
        self.distortion_coeffs = np.asarray([
            -0.38252,
            0.195521,
            0.000719038,
            0.00196389,
            -0.0346336])

        self.camera_matrix = np.asarray([
            [1329.98, 0, 646.475],
            [0, 1329.48, 390.789],
            [0, 0, 1]])

        self.face_cam_mm = np.asarray([-98.20092574611859924,
                                       13.586749303863609923,
                                       591.0001357050918288])

        self.leye_cam_mm = np.asarray([-71.130428199788372425,
                                       -12.089548827437531463,
                                       575.3897685617227298])

        self.gt_cam_mm = np.asarray([-1099.5803948625,
                                     31.06042467049997,
                                     - 44.927703281250004])

        self.rot_mat = np.asarray([
            [0.75789226, -0.01591712,  0.65218553],
            [-0.09241607,  0.98700119,  0.13148351],
            [-0.64580074, -0.15992275,  0.74656957]])

        self.tvec = np.asarray([
            [-71.4840822],
            [45.93883556],
            [617.72451667]])

        self.ec_pxs = np.asarray([
            [482.06115723, 362.85522461],
            [385.83392334, 377.89053345]])

        self.landmarks = np.ones((68, 1, 2), dtype=np.float32)

        self.norm_face_theta = 0.08371164488099574275
        self.norm_face_phi = 1.16868653667981276

        self.norm_leye_theta = 0.11742628651787750257
        self.norm_leye_phi = 1.1475201861509552997

        self._anthro_pts = AnthropometicPtsGenerator()

    def test_normalize_fullframe(self):

        face_gaze_vec = self.gt_cam_mm - self.face_cam_mm
        face_gv_mag = np.sqrt(face_gaze_vec[0] ** 2 + face_gaze_vec[1] ** 2 + face_gaze_vec[2] ** 2)
        face_gaze_vec = face_gaze_vec / face_gv_mag

        _, norm_face_gaze_theta, norm_face_gaze_phi, _, _, _, _, _, _, _ = \
            normalizeFullFrame(self.frame, self.face_cam_mm, self.ec_pxs, self.rot_mat,
                               face_gaze_vec, self.landmarks, self.camera_matrix,
                               self.distortion_coeffs, 'modified', 2.0)

        self.assertAlmostEqual(self.norm_face_theta, norm_face_gaze_theta[0], 6)
        self.assertAlmostEqual(self.norm_face_phi, norm_face_gaze_phi[0], 6)

    def test_normalize_face(self):

        face_gaze_vec = self.gt_cam_mm - self.face_cam_mm
        face_gv_mag = np.sqrt(face_gaze_vec[0] ** 2 + face_gaze_vec[1] ** 2 + face_gaze_vec[2] ** 2)
        face_gaze_vec = face_gaze_vec / face_gv_mag

        imageWidth = 224
        imageHeight = 224
        _, norm_face_gaze_theta, norm_face_gaze_phi, _, _, _, _ = \
            normalizeFace(self.frame, self.face_cam_mm, self.rot_mat, face_gaze_vec,
                          self.camera_matrix, self.distortion_coeffs, 'modified',
                          imageWidth, imageHeight)

        self.assertAlmostEqual(self.norm_face_theta, norm_face_gaze_theta[0], 6)
        self.assertAlmostEqual(self.norm_face_phi, norm_face_gaze_phi[0], 6)

    def test_normalize_eye(self):
        leye_gaze_vec = self.gt_cam_mm - self.leye_cam_mm
        leye_gv_mag = np.sqrt(leye_gaze_vec[0] ** 2 + leye_gaze_vec[1] ** 2 + leye_gaze_vec[2] ** 2)
        leye_gaze_vec = leye_gaze_vec / leye_gv_mag

        imageWidth = 120
        imageHeight = imageWidth
        _, norm_leye_gaze_theta, norm_leye_gaze_phi, _, _, _ = \
            normalizeEye(self.frame, self.leye_cam_mm, self.rot_mat, leye_gaze_vec,
                         self.camera_matrix, self.distortion_coeffs, 'modified',
                         imageWidth, imageHeight)

        self.assertAlmostEqual(self.norm_leye_theta, norm_leye_gaze_theta[0], 6)
        self.assertAlmostEqual(self.norm_leye_phi, norm_leye_gaze_phi[0], 6)

    def test_3D_landmarks(self):
        no_of_3D_landmarks = 38
        landmarks_3D = -1.0 * np.ones((no_of_3D_landmarks, 3), dtype=np.float32)
        landmarks_obj_3D = self._anthro_pts.get_landmarks_export_3D()
        for ind in range(no_of_3D_landmarks):
            obj_mm = np.reshape(landmarks_obj_3D[ind, :], (1, 3))
            cam_mm = projectObject2Camera(obj_mm, self.rot_mat, self.tvec)
            landmarks_3D[ind] = cam_mm.transpose()
        landmarks_3D = landmarks_3D.reshape(-1)

        self.assertEqual(no_of_3D_landmarks*3, landmarks_3D.shape[0])
