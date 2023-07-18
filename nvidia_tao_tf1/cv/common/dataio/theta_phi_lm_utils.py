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

"""Generate landmarks for use in theta phi calculations."""

import cv2
import numpy as np

'''
Terminologies:
WCS: World (object) coordinate system
CCS: Camera coordinate system
ICS: Image coordinate system
'''

# FACE_MODEL can be 'OLD-58', 'IX-68' or 'EOS-50'
FACE_MODEL = 'OLD-58'


class AnthropometicPtsGenerator(object):
    """Class to provide anthropometic points of face."""

    if FACE_MODEL == 'OLD-58':
        # Generic 3D face model http://aifi.isr.uc.pt/Downloads.html
        anthropometic_3D_landmarks = np.asarray([
            [-7.308957, 0.913869, 0.000000],
            [-6.775290, -0.730814, -0.012799],
            [-5.665918, -3.286078, 1.022951],
            [-5.011779, -4.876396, 1.047961],
            [-4.056931, -5.947019, 1.636229],
            [-1.833492, -7.056977, 4.061275],
            [0.000000, -7.415691, 4.070434],
            [1.833492, -7.056977, 4.061275],
            [4.056931, -5.947019, 1.636229],
            [5.011779, -4.876396, 1.047961],
            [5.665918, -3.286078, 1.022951],
            [6.775290, -0.730814, -0.012799],
            [7.308957, 0.913869, 0.000000],
            [5.311432, 5.485328, 3.987654],
            [4.461908, 6.189018, 5.594410],
            [3.550622, 6.185143, 5.712299],
            [2.542231, 5.862829, 4.687939],
            [1.789930, 5.393625, 4.413414],
            [2.693583, 5.018237, 5.072837],
            [3.530191, 4.981603, 4.937805],
            [4.490323, 5.186498, 4.694397],
            [-5.311432, 5.485328, 3.987654],
            [-4.461908, 6.189018, 5.594410],
            [-3.550622, 6.185143, 5.712299],
            [-2.542231, 5.862829, 4.687939],
            [-1.789930, 5.393625, 4.413414],
            [-2.693583, 5.018237, 5.072837],
            [-3.530191, 4.981603, 4.937805],
            [-4.490323, 5.186498, 4.694397],
            [1.330353, 7.122144, 6.903745],
            [2.533424, 7.878085, 7.451034],
            [4.861131, 7.878672, 6.601275],
            [6.137002, 7.271266, 5.200823],
            [6.825897, 6.760612, 4.402142],
            [-1.330353, 7.122144, 6.903745],
            [-2.533424, 7.878085, 7.451034],
            [-4.861131, 7.878672, 6.601275],
            [-6.137002, 7.271266, 5.200823],
            [-6.825897, 6.760612, 4.402142],
            [-2.774015, -2.080775, 5.048531],
            [-0.509714, -1.571179, 6.566167],
            [0.000000, -1.646444, 6.704956],
            [0.509714, -1.571179, 6.566167],
            [2.774015, -2.080775, 5.048531],
            [0.589441, -2.958597, 6.109526],
            [0.000000, -3.116408, 6.097667],
            [-0.589441, -2.958597, 6.109526],
            [-0.981972, 4.554081, 6.301271],
            [-0.973987, 1.916389, 7.654050],
            [-2.005628, 1.409845, 6.165652],
            [-1.930245, 0.424351, 5.914376],
            [-0.746313, 0.348381, 6.263227],
            [0.000000, 0.000000, 6.763430],
            [0.746313, 0.348381, 6.263227],
            [1.930245, 0.424351, 5.914376],
            [2.005628, 1.409845, 6.165652],
            [0.973987, 1.916389, 7.654050],
            [0.981972, 4.554081, 6.301271]
        ], dtype=np.longdouble)
    elif FACE_MODEL == 'IX-68':
        # New generic 3D face model (mean of our 10 face scans)
        anthropometic_3D_landmarks = np.asarray([
            [0.463302314, 0.499617226, 2.824620485],
            [0.433904979, 0.505937393, 2.644347876],
            [0.39794359, 0.54824712, 2.468309015],
            [0.347156364, 0.608686736, 2.301015556],
            [0.261349984, 0.708693571, 2.164755151],
            [0.149679065, 0.846413877, 2.038914531],
            [0.020857666, 1.000756979, 1.96136412],
            [-0.124583332, 1.132211104, 1.890638679],
            [-0.332052324, 1.199630469, 1.870016173],
            [-0.521015424, 1.142547488, 1.891351938],
            [-0.659920681, 1.03973259, 1.961001828],
            [-0.797577906, 0.86913183, 2.057442099],
            [-0.912351593, 0.80159378, 2.188996568],
            [-0.994247332, 0.707895722, 2.321974874],
            [-1.045988813, 0.646902599, 2.488486946],
            [-1.07838694, 0.581114346, 2.655235291],
            [-1.096934579, 0.572226902, 2.832175642],
            [0.306385975, 0.988582142, 3.263724804],
            [0.238308419, 1.087680236, 3.319453031],
            [0.126437213, 1.167731345, 3.357794225],
            [-0.003806859, 1.229740798, 3.335844517],
            [-0.126166103, 1.249807343, 3.300820023],
            [-0.483642399, 1.261558414, 3.320731789],
            [-0.594755229, 1.244249567, 3.356189996],
            [-0.709202692, 1.193373024, 3.370144337],
            [-0.830934606, 1.118067637, 3.342299908],
            [-0.911886856, 1.022390895, 3.286355436],
            [-0.31427322, 1.28056182, 3.116396815],
            [-0.312322683, 1.355140246, 3.0163863],
            [-0.310799612, 1.452512272, 2.899074256],
            [-0.315011633, 1.537534878, 2.777368128],
            [-0.125134574, 1.256734014, 2.648497283],
            [-0.216964348, 1.329175174, 2.637426972],
            [-0.310138743, 1.389713913, 2.611324817],
            [-0.414820289, 1.334226191, 2.642694384],
            [-0.513519868, 1.265409455, 2.656487644],
            [0.196186462, 1.035192601, 3.090169013],
            [0.126957612, 1.119997166, 3.156619817],
            [-0.027273278, 1.136058375, 3.157634437],
            [-0.100839235, 1.102722079, 3.088872135],
            [-0.021972392, 1.132983871, 3.03742063],
            [0.127623449, 1.10177733, 3.034567326],
            [-0.520080116, 1.100469962, 3.095452815],
            [-0.586792942, 1.133374192, 3.17071414],
            [-0.745613977, 1.125613876, 3.170327187],
            [-0.819571108, 1.0455795, 3.105413705],
            [-0.744035766, 1.112881519, 3.048785478],
            [-0.589515403, 1.131952509, 3.048771381],
            [0.02306129, 1.158300541, 2.368660092],
            [-0.080868714, 1.272260003, 2.42186287],
            [-0.181587959, 1.345463172, 2.448015809],
            [-0.312187512, 1.385880813, 2.454812676],
            [-0.452711696, 1.3551175, 2.454890877],
            [-0.558453415, 1.285798028, 2.426469952],
            [-0.664228875, 1.164380819, 2.386185408],
            [-0.571288593, 1.24077671, 2.312964618],
            [-0.466966776, 1.311935268, 2.253052473],
            [-0.318221454, 1.336186148, 2.228322476],
            [-0.170658994, 1.297508962, 2.247573286],
            [-0.071347391, 1.225932129, 2.294786155],
            [-0.053068627, 1.196410157, 2.366681814],
            [-0.16933859, 1.303018831, 2.379662782],
            [-0.31283252, 1.344644643, 2.37743327],
            [-0.453441203, 1.322521709, 2.388329715],
            [-0.585467342, 1.213929333, 2.378763407],
            [-0.473488985, 1.302666686, 2.342419624],
            [-0.311414156, 1.341170423, 2.319458067],
            [-0.16669072, 1.297568522, 2.336282581]
        ], dtype=np.longdouble)
    elif FACE_MODEL == 'EOS-50':
        # https://github.com/patrikhuber/eos/blob/master/share/ibug_to_sfm.txt
        sfm_points_50 = np.load('nvidia_tao_tf1/cv/common/dataio/eos_mean_68.npy')
        list_points = []
        list_points.append(9)
        list_points += range(18, 61)
        list_points += range(62, 65)
        list_points += range(66, 69)

        anthropometic_3D_landmarks = np.zeros((68, 3), dtype=np.float32)
        i = 0
        for ind in list_points:
            anthropometic_3D_landmarks[ind-1, :] = sfm_points_50[i, :]
            i += 1

        anthropometic_3D_landmarks = np.asarray(anthropometic_3D_landmarks, dtype=np.longdouble)

    if FACE_MODEL == 'OLD-58':
        le_outer = anthropometic_3D_landmarks[13]
        le_inner = anthropometic_3D_landmarks[17]
        re_outer = anthropometic_3D_landmarks[21]
        re_inner = anthropometic_3D_landmarks[25]
    else:
        le_outer = anthropometic_3D_landmarks[45]
        le_inner = anthropometic_3D_landmarks[42]
        re_outer = anthropometic_3D_landmarks[36]
        re_inner = anthropometic_3D_landmarks[39]

    le_center = (le_inner + le_outer) / 2.0
    le_center = np.reshape(le_center, (1, 3))
    re_center = (re_inner + re_outer) / 2.0
    re_center = np.reshape(re_center, (1, 3))

    # Use conventional cord system [+x right to left eye] [+y nose to mouth] [nose to back of head].
    if FACE_MODEL == 'OLD-58':
        anthropometic_3D_landmarks[:, 1] *= -1  # Inverse Y axis.
        anthropometic_3D_landmarks[:, 2] *= -1  # Inverse Z axis.
    elif FACE_MODEL == 'IX-68':
        anthropometic_3D_landmarks[:, [1, 2]] = anthropometic_3D_landmarks[:, [2, 1]]
        anthropometic_3D_landmarks *= -1  # Inverse X, Y, Z axis.
    elif FACE_MODEL == 'EOS-50':
        anthropometic_3D_landmarks[:, 1] *= -1  # Inverse Y axis.
        anthropometic_3D_landmarks[:, 2] *= -1  # Inverse Z axis.

    if FACE_MODEL == 'OLD-58':
        '''
        OLD-58 face model has:
        - interpupillary distance of 70 mm
        - intercanthal distance of 35.8 mm
        => larger than average human values of 64-65 mm and 30-31 mm respectively
        => need a normalization (e.g. downscaling to 6.5/7 ratio)
        '''
        face_model_scaling = 65.0 / 7.0
    else:
        face_model_scaling = 65.0 / np.linalg.norm(le_center - re_center)

    anthropometic_3D_landmarks *= face_model_scaling

    if FACE_MODEL == 'OLD-58':
        '''
        Landmark correspondences (index-1)
        3D-58, 2D-68
        34,   27 - left eye, eyebrow, outer point
        30,   23 - left eye, eyebrow, inner point
        35,   22 - right eye, eyebrow, inner point
        39,   18 - right eye, eyebrow, outer point
        14,   46 - left eye, eye corner, outer point
        18,   43 - left eye, eye corner, inner point
        26,   40 - right eye, eye corner, inner point
        22,   37 - right eye, eye corner, outer point
        56,   36 - nose, right side, middle point
        50,   32 - nose, left side, middle point
        44,   55 - mouth, left corner point
        40,   49 - mouth, right corner point
        46,   58 - mouth, bottom center point
        7,   9 - chin, center point
        '''
        pnp_indices = [33, 29, 34, 38, 13, 17, 25, 21, 55, 49, 43, 39, 45, 6]

        '''
        Landmarks to be used for eye center estimation (index-1)
        left eye outer corner (3D:#14, Dlib:#46)
        left eye inner corner (3D:#18, Dlib:#43)
        right eye outer corner (3D:#22, Dlib:#37)
        right eye inner corner (3D:#26, Dlib:#40)
        chin center (3D:#7, Dlib:#9)
        nose leftmost (3D:#55, GT/Dlib:#36)
        nose left (3D:#54, GT/Dlib:#35)
        nose rightmost (3D:#51, GT/Dlib:#32)
        nose right (3D:#52, GT/Dlib:#33)
        '''
        robust_expr_var_indices = [13, 17, 21, 25, 6, 54, 53, 50, 51]

        landmarks_2D_indices_selected = [26, 22, 21, 17, 45, 42, 39, 36, 35, 31, 54, 48, 57, 8]
        landmarks_2D_indices_expr_inv = [45, 42, 36, 39, 8, 35, 34, 31, 32]
    else:  # Landmark correspondence is one to one for new model.
        landmarks_2D_indices_selected = [36, 39, 42, 45, 48, 54, 27, 28, 29, 30, 31, 32, 33, 34,
                                         35, 8, 17, 21, 22, 26]
        if FACE_MODEL == 'IX-68':
            landmarks_2D_indices_expr_inv = [45, 42, 36, 39, 8, 33, 30]
        elif FACE_MODEL == 'EOS-50':
            landmarks_2D_indices_expr_inv = [45, 42, 36, 39, 8, 35, 34, 32, 31]
            # Shalini v1
            # landmarks_2D_set_expr_inv = [45, 42, 36, 39, 33, 30, 27]
            # Shalini v2
            # landmarks_2D_set_expr_inv = [45, 42, 36, 39, 31, 32, 33, 34, 35, 30, 29, 28, 27]

        pnp_indices = landmarks_2D_indices_selected
        robust_expr_var_indices = landmarks_2D_indices_expr_inv

    def __init__(self):
        """Initialize landmarks to be None so they can be lazily set."""
        self._pnp_lm = None
        self._robust_expr_var_lm = None

    def get_all_landmarks(self):
        """Get all landmarks."""
        return self.anthropometic_3D_landmarks

    def get_landmarks_export_3D(self):
        """Get selected 3D landmarks to write out into tfrecords."""
        landmarks_3D = np.zeros((38, 3), dtype=np.float32)
        list_points = []
        if FACE_MODEL == 'OLD-58':
            list_points.append(6)
            list_points += range(13, 15)
            list_points += range(16, 19)
            list_points += range(20, 23)
            list_points += range(24, 27)
            list_points += range(29, 47)
            list_points += range(50, 55)
            i = 0
            for ind in list_points:
                landmarks_3D[i, :] = self.anthropometic_3D_landmarks[ind, :]
                i += 1
            landmarks_3D[36, :] = (self.anthropometic_3D_landmarks[48, :] +
                                   self.anthropometic_3D_landmarks[56, :]) / 2.0
            landmarks_3D[37, :] = (self.anthropometic_3D_landmarks[49, :] +
                                   self.anthropometic_3D_landmarks[55, :]) / 2.0
        elif FACE_MODEL == 'IX-68':
            list_points.append(8)
            list_points += range(17, 27)
            list_points += range(29, 48)
            list_points.append(48)
            list_points += range(50, 53)
            list_points.append(54)
            list_points += range(56, 59)
            i = 0
            for ind in list_points:
                landmarks_3D[i, :] = self.anthropometic_3D_landmarks[ind, :]
                i += 1

        return landmarks_3D

    def get_pnp_landmarks(self):
        """Get landmarks to be used in perspective-n-point algorithm."""
        if self._pnp_lm is None:
            self._pnp_lm = self.anthropometic_3D_landmarks[self.pnp_indices]

        return self._pnp_lm

    def get_robust_expr_var_landmarks(self):
        """Get landmarks that are robust against variations in expressions."""
        if self._robust_expr_var_lm is None:
            self._robust_expr_var_lm = self.anthropometic_3D_landmarks[self.robust_expr_var_indices]

        return self._robust_expr_var_lm

    def get_selected_landmarks(self, landmarks_2D, indices_2D, lm_3D):
        """Get selected landmarks from 2D and 3D landmarks and invalid landmarks percentage."""
        landmarks_selected_2D = []
        landmarks_selected_3D = []
        cnt_invalid = 0

        n_selected_landmarks = len(indices_2D)
        for i in range(n_selected_landmarks):
            coord = landmarks_2D[indices_2D[i]]
            if coord[0] > 0 and coord[1] > 0:
                landmarks_selected_2D.append(coord)
                landmarks_selected_3D.append(lm_3D[i])
            else:
                cnt_invalid += 1

        return np.asarray(landmarks_selected_2D),\
            np.asarray(landmarks_selected_3D),\
            cnt_invalid * 100.0 / n_selected_landmarks

    def get_eye_centers(self):
        """Get centers of eyes based off eye pts in landmarks."""
        eye_lm = self.get_robust_expr_var_landmarks()
        left_eye_pts = eye_lm[:2]
        right_eye_pts = eye_lm[2:4]

        return np.reshape(np.mean(left_eye_pts, axis=0), (1, 3)),\
            np.reshape(np.mean(right_eye_pts, axis=0), (1, 3))

    def get_face_center(self):
        """Get face center as mean of 6 landmarks (4 eye points + 2 mouth points)."""
        lm = self.get_pnp_landmarks()
        if FACE_MODEL == 'OLD-58':
            face_pts = lm[[4, 5, 6, 7, 10, 11]]
        elif FACE_MODEL == 'IX-68':
            face_pts = lm[[4, 5, 6, 7, 12, 13]]

        return np.reshape(np.mean(face_pts, axis=0), (1, 3))


class PnPPtsGenerator(object):
    """
    Class to provide Perspective-n-Point(PnP) rvec and tvec.

    PnP rvec, tvec describe how to project from pts coordinate system to image plane.
    """

    def __init__(self, camera_mat, distortion_coeffs):
        """Initialize camera matrix and distortion coefficients."""
        self._camera_mat = camera_mat
        self._distortion_coeffs = distortion_coeffs

    @staticmethod
    def get_cv_array(precise_arr):
        """Cast more precise array to float for cv."""
        return np.asarray(precise_arr, dtype=np.float)

    def compute_EPnP(self, points_3D, points_2D):
        """"Find object pose efficiently."""
        points_2D = np.expand_dims(points_2D, axis=1)
        points_3D = np.expand_dims(points_3D, axis=1)
        retval, rvec, tvec = cv2.solvePnP(
            self.get_cv_array(points_3D),
            self.get_cv_array(points_2D),
            self._camera_mat,
            self._distortion_coeffs,
            None,
            None,
            False,
            cv2.SOLVEPNP_EPNP)
        return retval, rvec, tvec

    def compute_PnP_Iterative(self, points_3D, points_2D, rvec, tvec):
        """"Find object pose which minimizes reporjection error."""
        retval, rvec, tvec = cv2.solvePnP(
            self.get_cv_array(points_3D),
            self.get_cv_array(points_2D),
            self._camera_mat,
            self._distortion_coeffs,
            rvec,
            tvec,
            True,
            cv2.SOLVEPNP_ITERATIVE)
        return retval, rvec, tvec


def projectObject2Camera(object_coords, rot_mat, tvec):
    """Project object coordinates (WCS) to camera coordinates (CCS).

    Args:
        object_coords (np.ndarray): (x, y, z) coordinates in WCS.
        rot_mat (np.ndarray): 3D WCS to 2D ICS rotational transformation matrix.
        tvec (np.ndarray): 3D WCS to 2D ICS translation transformation matrix.

    Returns:
        cam_coords (np.ndarray): output cam coordinates.
    """
    RPw = rot_mat.dot(object_coords.transpose())
    cam_coords = RPw + tvec
    return cam_coords


def projectCamera2Image(cam_coords, cam_mat):
    """Project camera coordinates (CCS) to image coordinates (ICS).

    P_i = cam_mat*cam_coords

    Args:
        cam_coords (np.ndarray): (x, y, z) coordinates in CCS.
        cam_mat (np.ndarray): camera calibration matrix.

    Returns:
        image_coords (np.ndarray): output image coordinates.
    """
    image_coords = np.matmul(cam_mat, cam_coords)
    image_coords /= image_coords[2]
    return image_coords
