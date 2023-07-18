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

"""Helper for retrieving gaze vector and theta phi from existing results."""

import cv2
import numpy as np

'''
Terminologies:
WCS: World (object) coordinate system
CCS: Camera coordinate system
ICS: Image coordinate system
'''


def populate_theta_phi(info, is_valid_theta_phi):
    """Populate 1D array with scalar."""
    if not is_valid_theta_phi or info is None:
        return -1

    assert info.shape == (1,)
    return np.asscalar(info)


def populate_gaze_info(np_2D_arr, is_valid_theta_phi):
    """Populate array with scalar np values."""
    if not is_valid_theta_phi or np_2D_arr is None:
        return [-1] * 3

    assert isinstance(np_2D_arr, np.ndarray)

    json_list = []
    for arr in np_2D_arr:
        assert isinstance(arr, np.ndarray) and arr.shape[0] == 1
        json_list.append(np.asscalar(arr))
    return json_list


def populate_head_norm_bbinfo(info, ind, is_valid_theta_phi):
    """Populate with scalar value."""
    if not is_valid_theta_phi or info is None:
        return -1

    return info[ind]


def populate_head_norm_listinfo(info, typ, is_valid_theta_phi):
    """Populate array with scalar np values."""
    if not is_valid_theta_phi or info is None:
        if typ == '3D':
            return -1.0 * np.ones((114,), dtype=np.float32)
        if typ == '2D':
            return -1.0 * np.ones((208,), dtype=np.float32)
        if typ == 'cnv_mat':
            return -1.0 * np.ones((9,), dtype=np.float32)

    return np.float32(info)


def populate_head_norm_path(info, is_valid_theta_phi):
    """Populate string."""
    if not is_valid_theta_phi or info is None:
        return ''

    return info


def populate_head_norm_float(info, is_valid_theta_phi):
    """Populate np float."""
    if not is_valid_theta_phi or info is None:
        return -1.0

    return np.float32(info)


def calculate_reprojection_error(ref, reproj):
    """Get reprojection error.

    Args:
        ref (np.ndarray): Original 3D points.
        reproj (np.ndarray): Projected 3D points onto image plane.

    Returns:
        reprojection error (float):
            The mean squared difference between projected and original points.
    """
    if isinstance(ref, list):
        ref = np.asarray(ref)
    if isinstance(reproj, list):
        reproj = np.asarray(reproj)

    diff = ref[:2, :] - reproj[:2, :]
    return np.mean(np.sqrt(diff[0, :] ** 2 + diff[1, :] ** 2))


def compute_gaze_vector_from_theta_phi(theta, phi):
    """Get forward facing gaze vector [x, y, z] in CCS from theta/beta and alpha/phi angles.

    Args:
        theta (float): Gaze pitch in radians.
        phi (float): Gaze yaw in radians.

    Returns:
        gaze_vec (np.ndarray): Forward facing gaze vector [x, y, z] in CCS.
    """
    gaze_vec = np.zeros(3)
    gaze_vec[0] = -np.cos(theta) * np.sin(phi)
    gaze_vec[1] = -np.sin(theta)
    gaze_vec[2] = -np.cos(theta) * np.cos(phi)

    return gaze_vec


def compute_theta_phi_from_gaze_vector(x, y, z):
    """Get theta phi angles from forward facing gaze vector in CCS.

    Using appearance-based gaze estimation method in the Wild (MPIIGaze).

    Args:
        x (float): CCS x-coord of forward facing gaze vector.
        y (float): CCS y-coord of forward facing gaze vector.
        z (float): CCS z-coord of forward facing gaze vector.

    Returns:
        theta (float): Gaze pitch in radians.
        phi (float): Gaze yaw in radians.
    """
    theta = np.arcsin(-y)  # Range [-pi/2, pi/2]
    phi = np.arctan2(-x, -z)  # Range [-pi, pi], not the range [-pi/2, pi/2] of arctan

    return np.reshape(theta, (1,)), np.reshape(phi, (1,))


def compute_PoR_from_theta_phi(theta, phi, pc_cam_mm, R, T):
    """Get the intersection of gaze vector (generated using theta & phi) with the monitor plane.

    TODO: consider normalized camera space (scaling, divide by Z).

    Args:
        theta (float): Gaze pitch in radians.
        phi (float): Gaze yaw in radians.
        pc_cam_mm (np.ndarray): 3D pupil center coordinates in camera space in mm.
        R (np.ndarray): Rotation matrix computed from mirror calibration.
        T (np.ndarray): Translation vector computed from mirror calibration.

    Returns:
        PoR_x (np.ndarray): x-coord of intersection point in camera space in mm.
        PoR_y (np.ndarray): y-coord of intersection point in camera space in mm.
        PoR_z (np.ndarray): z-coord of intersection point in camera space in mm.
    """
    screenNormal = R

    screenNormal = screenNormal[:, 2]

    screenOrigin = T

    gaze_vec = compute_gaze_vector_from_theta_phi(theta, phi)

    dNormalGazeVec = np.dot(screenNormal, gaze_vec)

    d = np.dot(screenNormal, screenOrigin)

    dNormalEyeCenter = np.dot(screenNormal, pc_cam_mm)

    t = (d - dNormalEyeCenter) / dNormalGazeVec

    gaze_vec = np.expand_dims(gaze_vec, axis=1)
    intersectPoint = pc_cam_mm + t * gaze_vec
    PoR_x = intersectPoint[0]
    PoR_y = intersectPoint[1]
    PoR_z = intersectPoint[2]

    return PoR_x, PoR_y, PoR_z


def compute_PoR_from_gaze_vector(gaze_vec, pc_cam_mm, R, T):
    """Get the intersection of gaze vector (generated using theta & phi) with the monitor plane.

    TODO: consider normalized camera space (scaling, divide by Z).

    Args:
        gaze vec (np.ndarray): 3D gaze vector.
        pc_cam_mm (np.ndarray): 3D pupil center coordinates in camera space in mm.
        R (np.ndarray): Rotation matrix computed from mirror calibration.
        T (np.ndarray): Translation vector computed from mirror calibration.

    Returns:
        PoR_x (np.ndarray): x-coord of intersection point in camera space in mm.
        PoR_y (np.ndarray): y-coord of intersection point in camera space in mm.
        PoR_z (np.ndarray): z-coord of intersection point in camera space in mm.
    """
    screenNormal = R

    screenNormal = screenNormal[:, 2]

    screenOrigin = T

    dNormalGazeVec = np.dot(screenNormal, gaze_vec)

    d = np.dot(screenNormal, screenOrigin)

    dNormalEyeCenter = np.dot(screenNormal, pc_cam_mm)

    t = (d - dNormalEyeCenter) / dNormalGazeVec

    intersectPoint = pc_cam_mm + t * gaze_vec
    PoR_x = intersectPoint[0]
    PoR_y = intersectPoint[1]
    PoR_z = intersectPoint[2]

    return PoR_x, PoR_y, PoR_z


def normalizeFullFrame(img, face_cam, ec_pxs, rot_mat, face_gaze_vec, landmarks, cam, dist,
                       method='modified', scale_factor=2.0):
    """Perform face normalization with full frame warping.

    Args:
        img (np.ndarray): input frame.
        face_cam (np.ndarray): face center in camera space in mm.
        ec_pxs (np.ndarray): left/right eye centers in pixel coordinates.
        rot_mat (np.ndarray): head pose rotation matrix obtained from PnP.
        face_gaze_vec (np.ndarray): 3D gaze vector for face.
        landmarks (np.ndarray): input 2D landmarks.
        cam (np.ndarray): camera calibration matrix.
        dist (np.ndarray): lens distortion coefficients.
        method (str): normalization method.
        scale_factor (float): face size scaling factor.

    Returns:
        img_warped (np.ndarray): output warped frame.
        gaze_theta_new (np.float): face normalized gaze pitch.
        gaze_phi_new (np.float): face normalized gaze yaw.
        hp_theta_new (np.float): face normalized hp pitch.
        hp_phi_new (np.float): face normalized hp yaw.
        face_bb (np.array): face rectangle on normalized frame.
        leye_bb (np.array): left eye rectangle on normalized frame.
        reye_bb (np.array): right eye rectangle on normalized frame.
        landmarks_norm (np.ndarray): normalized 2D landmarks.
        cnvMat (np.ndarray): warping conversion matrix.
    """
    focal_new = 960 * scale_factor
    distance_new = 600.0
    imageWidth = img.shape[1]
    imageHeight = img.shape[0]
    roiSize = (imageWidth, imageHeight)

    img_u = cv2.undistort(img, cam, dist)

    distance = np.linalg.norm(face_cam)
    z_scale = distance_new / distance
    cam_new = np.array([
        [focal_new, 0, roiSize[0] / 2],
        [0, focal_new, roiSize[1] / 2],
        [0, 0, 1.0], ])

    scaleMat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale], ])

    hRx = rot_mat[:, 0]
    forward = (face_cam / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)

    rotMat = np.c_[right, down, forward].T

    # Conversion matrix M.
    cnvMat = np.dot(scaleMat, rotMat)
    cnvMat = np.float64(cnvMat)

    # Warping matrix, W.
    warpMat = np.dot(np.dot(cam_new, cnvMat), np.linalg.inv(cam))
    img_warped = cv2.warpPerspective(img_u, warpMat, roiSize)

    # Face bounding box (upper left corner (x,y), width, height).
    face_bb = [int(imageWidth/2 - 112*scale_factor), int(imageHeight/2 - 112*scale_factor),
               int(224*scale_factor), int(224*scale_factor)]

    # Warp 2D landmarks.
    landmarks = landmarks.astype(float)
    landmarks = landmarks.reshape(landmarks.shape[0], 1, 2)
    undist_landmarks = cv2.undistortPoints(landmarks, cam, dist, P=cam)
    landmarks_norm = cv2.perspectiveTransform(undist_landmarks, warpMat)
    landmarks_norm = np.reshape(landmarks_norm, (landmarks_norm.shape[0], landmarks_norm.shape[2]))

    # Warp 2D eye centers.
    ec_pxs = ec_pxs.astype(float)
    ec_pxs = ec_pxs.reshape(ec_pxs.shape[0], 1, 2)
    undist_ec_pxs = cv2.undistortPoints(ec_pxs, cam, dist, P=cam)
    ec_pxs_norm = cv2.perspectiveTransform(undist_ec_pxs, warpMat)
    ec_pxs_norm = np.reshape(ec_pxs_norm, (ec_pxs_norm.shape[0], ec_pxs_norm.shape[2]))
    leye_center = ec_pxs_norm[0]
    reye_center = ec_pxs_norm[1]

    leye_bb = [int(leye_center[0] - 40*scale_factor), int(leye_center[1] - 40*scale_factor),
               int(80*scale_factor), int(80*scale_factor)]
    reye_bb = [int(reye_center[0] - 40*scale_factor), int(reye_center[1] - 40*scale_factor),
               int(80*scale_factor), int(80*scale_factor)]

    if leye_bb[0] < 0 or leye_bb[1] < 0 or scale_factor <= 0 or \
            (leye_bb[0] + leye_bb[2]) > imageWidth or \
            (leye_bb[1] + leye_bb[3]) > imageHeight:
        leye_bb = None

    if reye_bb[0] < 0 or reye_bb[1] < 0 or scale_factor <= 0 or \
            (reye_bb[0] + reye_bb[2]) > imageWidth or \
            (reye_bb[1] + reye_bb[3]) > imageHeight:
        reye_bb = None

    # Validate norm image orientation (ie, chin is lower than eye center).
    mideyes_center = (leye_center + reye_center) / 2.0
    chin = landmarks_norm[8]
    if chin[1] <= (mideyes_center[1] + 100 * scale_factor):
        img_warped = None

    # Calculate head orientation.
    dY = chin[1] - mideyes_center[1]
    dX = abs(chin[0] - mideyes_center[0])
    angle = np.degrees(np.arctan2(dX, dY))
    THR = 30

    # Validate head orientation.
    # Normal < THR, horizontal flip < (90-THR), vertical flip < (180-THR).
    if angle > (90-THR):
        print('Invalid head orientation: ', angle)
        if angle > (180 - THR):
            print('Invalid head orientation: ', angle, '. Flipped vertically.')
        else:
            print('Invalid head orientation: ', angle, '. Flipped horizontally.')
        img_warped = None

    if method == 'original':
        cnvMat = np.dot(scaleMat, rotMat)
    else:
        cnvMat = rotMat

    # Calculate new HP vector.
    hR_new = np.dot(cnvMat, rot_mat)
    Zv = hR_new[:, 2]

    # Reverse direction as we want head pose z pointing positive Z direction.
    # (towards the back of the head)! Also, convention in all research papers.
    hp_theta_new, hp_phi_new = compute_theta_phi_from_gaze_vector(-Zv[0], -Zv[1], -Zv[2])

    # Calculate new gaze GT vector: [g_n = M * g_r] where M is only rotation.
    face_gaze_vec_new = np.dot(cnvMat, face_gaze_vec)
    gaze_theta_new, gaze_phi_new = compute_theta_phi_from_gaze_vector(face_gaze_vec_new[0],
                                                                      face_gaze_vec_new[1],
                                                                      face_gaze_vec_new[2])

    return img_warped, gaze_theta_new, gaze_phi_new, hp_theta_new, hp_phi_new, face_bb, leye_bb, \
        reye_bb, landmarks_norm, cnvMat


def normalizeFace(img, face_cam, rot_mat, face_gaze_vec, cam, dist,
                  method='modified', imageWidth=224, imageHeight=224):
    """Perform face normalization.

    Args:
        img (np.ndarray): input frame.
        face_cam (np.ndarray): face center in camera space in mm.
        rot_mat (np.ndarray): head pose rotation matrix obtained from PnP.
        face_gaze_vec (np.ndarray): 3D gaze vector for face.
        cam (np.ndarray): camera calibration matrix.
        dist (np.ndarray): lens distortion coefficients.
        method (str): normalization method.
        imageWidth (int): output image width.
        imageHeight (int): output image height.

    Returns:
        img_warped (np.ndarray): output warped frame.
        gaze_theta_new (np.float): face normalized gaze pitch.
        gaze_phi_new (np.float): face normalized gaze yaw.
        hp_theta_new (np.float): face normalized hp pitch.
        hp_phi_new (np.float): face normalized hp yaw.
        per_oof (np.array): percentage of out of frame after normalization.
        cnvMat (np.ndarray): warping conversion matrix.
    """
    focal_new = 960 * imageWidth / 224
    distance_new = 600.0
    roiSize = (imageWidth, imageHeight)

    img_u = cv2.undistort(img, cam, dist)

    distance = np.linalg.norm(face_cam)
    z_scale = distance_new / distance
    cam_new = np.array([
        [focal_new, 0, roiSize[0] / 2],
        [0, focal_new, roiSize[1] / 2],
        [0, 0, 1.0], ])

    scaleMat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale], ])

    hRx = rot_mat[:, 0]
    forward = (face_cam / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)

    rotMat = np.c_[right, down, forward].T

    # Conversion matrix M.
    cnvMat = np.dot(scaleMat, rotMat)
    cnvMat = np.float64(cnvMat)

    # Warping matrix W.
    warpMat = np.dot(np.dot(cam_new, cnvMat), np.linalg.inv(cam))
    img_warped = cv2.warpPerspective(img_u, warpMat, roiSize)

    # Calculate percentage of face which is out of frame.
    per_oof = round(np.count_nonzero(img_warped == 0) / (1.0 * roiSize[0] * roiSize[1]), 2)

    if method == 'original':
        cnvMat = np.dot(scaleMat, rotMat)
    else:
        cnvMat = rotMat

    # Calculate new HP vector.
    hR_new = np.dot(cnvMat, rot_mat)
    Zv = hR_new[:, 2]

    # Reverse direction as we want head pose z pointing positive Z direction.
    # (towards the back of the head)! Also, convention in all research papers.
    hp_theta_new, hp_phi_new = compute_theta_phi_from_gaze_vector(-Zv[0], -Zv[1], -Zv[2])

    # Calculate new gaze GT vector: [g_n = M * g_r] where M is only rotation.
    face_gaze_vec_new = np.dot(cnvMat, face_gaze_vec)
    gaze_theta_new, gaze_phi_new = compute_theta_phi_from_gaze_vector(face_gaze_vec_new[0],
                                                                      face_gaze_vec_new[1],
                                                                      face_gaze_vec_new[2])

    return img_warped, gaze_theta_new, gaze_phi_new, hp_theta_new, hp_phi_new, per_oof, cnvMat


def normalizeEye(img, eye_cam, rot_mat, gaze_vec, cam, dist,
                 method='modified', imageWidth=60, imageHeight=36):
    """Perform eye normalization.

    Args:
        img (np.ndarray): input frame.
        eye_cam (np.ndarray): eye center in camera space in mm.
        rot_mat (np.ndarray): head pose rotation matrix obtained from PnP.
        gaze_vec (np.ndarray): 3D gaze vector for face.
        cam (np.ndarray): camera calibration matrix.
        dist (np.ndarray): lens distortion coefficients.
        method (str): normalization method.
        imageWidth (int): output image width.
        imageHeight (int): output image height.

    Returns:
        img_warped (np.ndarray): output warped frame.
        gaze_theta_new (np.float): face normalized gaze pitch.
        gaze_phi_new (np.float): face normalized gaze yaw.
        hp_theta_new (np.float): face normalized hp pitch.
        hp_phi_new (np.float): face normalized hp yaw.
        per_oof (np.array): percentage of out of frame after normalization.
        cnvMat (np.ndarray): warping conversion matrix.
    """
    focal_new = 960 * imageWidth / 60
    distance_new = 600.0
    roiSize = (imageWidth, imageHeight)

    img_u = cv2.undistort(img, cam, dist)

    distance = np.linalg.norm(eye_cam)
    z_scale = distance_new / distance
    cam_new = np.array([
        [focal_new, 0, roiSize[0] / 2],
        [0, focal_new, roiSize[1] / 2],
        [0, 0, 1.0], ])

    scaleMat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale], ])

    hRx = rot_mat[:, 0]
    forward = (eye_cam / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)

    rotMat = np.c_[right, down, forward].T

    # Conversion matrix M.
    cnvMat = np.dot(scaleMat, rotMat)
    cnvMat = np.float64(cnvMat)

    # Warping matrix W.
    warpMat = np.dot(np.dot(cam_new, cnvMat), np.linalg.inv(cam))
    img_warped = cv2.warpPerspective(img_u, warpMat, roiSize)

    if method == 'original':
        cnvMat = np.dot(scaleMat, rotMat)
    else:
        cnvMat = rotMat

    # Calculate new HP vector.
    hR_new = np.dot(cnvMat, rot_mat)
    Zv = hR_new[:, 2]

    # Reverse direction as we want head pose z pointing positive Z direction.
    # (towards the back of the head)! Also, convention in all research papers.
    hp_theta_new, hp_phi_new = compute_theta_phi_from_gaze_vector(-Zv[0], -Zv[1], -Zv[2])

    # Calculate new gaze GT vector: [g_n = M * g_r] where M is only rotation.
    gaze_vec_new = np.dot(cnvMat, gaze_vec)
    gaze_theta_new, gaze_phi_new = compute_theta_phi_from_gaze_vector(gaze_vec_new[0],
                                                                      gaze_vec_new[1],
                                                                      gaze_vec_new[2])

    return img_warped, gaze_theta_new, gaze_phi_new, hp_theta_new, hp_phi_new, cnvMat
