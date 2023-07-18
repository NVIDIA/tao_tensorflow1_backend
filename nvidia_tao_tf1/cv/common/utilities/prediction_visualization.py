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

"""Prediction visualizer that bucketizes users predictions."""

import math
import os
import shutil
import subprocess
import cv2
import matplotlib
matplotlib.use('agg')  # Causes 'E402:module level import not at top of file'.
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p  # noqa: E402

pd.set_option('display.max_columns', None)


class kpi_prediction_linker():
    """Links together the predictions from each frame with KPI set."""

    def __init__(self, model_type, kpi_sets, path_info, gaze_origin,
                 theta_phi_degrees, write_time_instance_info=False):
        """
        Initialize kpi_prediction_linker.

        model type (str): type of model to visualize, can be one of [xyz_model, joint_model
        theta_phi_model]
        kpi sets (list(str)) : kpi sets to visualize
        gaze_origin (str): origin point of theta-phi, can be one of:
            normal: normal theta-phi used as GT theta-phi from previous pipelines
            mid_eyes: mid point of two eyes
            left_eye: left eye
            right_eye: right eye
            delta: when learning delta theta-phi between normal theta-phi and head pose theta-phi
                    it uses mid_eyes as gaze origin
            headnorm_face: when learning face normalized model
                    it uses mid point of face (mean of leye, reye and mouth corners)
            headnorm_leye: when learning left eye normalized model
                    it uses left eye as gaze origin
            headnorm_reye: when learning right eye normalized model
                    it uses right eye as gaze origin
        theta_phi_degrees (bool): Specify this field if theta-phi model was learned in
                                  degrees or radian.
        write_time_instance_info (bool): Specify field if writing time instance info is needed.
        """

        self._root_path = path_info['root_path']
        self.gcosmos = path_info['set_directory_path']

        self.model_type = model_type
        if self.model_type not in ['xyz', 'joint', 'theta_phi', 'joint_xyz', 'joint_tp']:
            raise NotImplementedError(
                'A model of type {} is not implemented, please choose '
                'from one of the following: xyz, joint, theta_phi'.format(self.model_type)
            )
        self.theta_phi_degrees = theta_phi_degrees

        self.kpi_GT_dir = path_info['ground_truth_folder_name']
        self.kpi_GT_file_folder_name = path_info['ground_truth_file_folder_name']

        if write_time_instance_info:
            self.kpi_GT_file = ['test', 'train', 'validate']
        else:
            self.kpi_GT_file = ['test']

        self.gaze_origin = gaze_origin
        self.kpisets = []
        for sets_list in kpi_sets:
            self.kpisets.append(sets_list.split(' '))

        if len(self.gcosmos) != len(self.kpi_GT_dir) != len(self.kpi_GT_file_folder_name)\
                != len(self.kpisets):
            raise Exception('Expected lists of directory paths and lists of GT path and'
                            'lists of folders and lists of kpi sets'
                            'to be the same, received'
                            ' {}, {}, {}, {}'.format(len(self.gcosmos), len(self.kpi_GT_dir),
                                                     len(self.kpi_GT_file_folder_name),
                                                     len(self.kpisets)))

        self.resultCols = ["image_path", "image_name",
                           "set_id", "user_id", "region_name", "region_label",
                           # Ground truth.
                           "screen_pix_x", "screen_pix_y",
                           "GT_cam_x", "GT_cam_y", "GT_cam_z", "GT_cam_norm",
                           "GT_theta", "GT_phi",
                           "center_of_eyes_x", "center_of_eyes_y", "center_of_eyes_z",
                           "GT_hp_theta", "GT_hp_phi",
                           # Prediction.
                           "gaze_cam_mm_x", "gaze_cam_mm_y", "gaze_cam_mm_z", "gaze_cam_norm",
                           "gaze_theta", "gaze_phi",
                           # Error.
                           "cam_err_x", "cam_err_y", "cam_err_z", "cam_err_xy",
                           "cam_error", "screen_phy_error", "user_dist", "degree_error",
                           "direct_degree_error", "backwardIntersection", "cone_degree"]

        self.dfgaze = None
        self.dfunqgaze = None

    @staticmethod
    def _get_region_names(org_path, set_id):
        """Read the region names from the folders in config."""
        config_path = os.path.join(org_path, set_id, 'Config')
        configs = os.listdir(config_path)
        regions = [
            config
            for config in configs
            if os.path.isdir(os.path.join(config_path, config))
            and config.startswith('region_')
        ]
        if not regions:
            # On bench collection has no regions.
            if 'incar' in set_id:
                raise IOError('In car data set should have region_ folders in {}'
                              .format(config_path))
            return ['']
        if 'incar' not in set_id:
            raise IOError('On bench data set should not have region_ folders in {}'
                          .format(config_path))
        return regions

    @staticmethod
    def load_screen_parameters(org_path, set_id):
        """Loads R and T parameters."""
        config_path = os.path.join(org_path, set_id, 'Config')
        screens = {}
        for region_name in kpi_prediction_linker._get_region_names(org_path, set_id):
            scrpW, scrpH = 1920.0, 1080.0  # Default vals.
            # Try to find a config file for the resolution.
            resolution = os.path.join(config_path, region_name, 'resolution.txt')
            if os.path.isfile(resolution):
                with open(resolution) as f:
                    scrpW = float(f.readline())
                    scrpH = float(f.readline())

            # Check which of board_size or TV_size is available.
            board_size = os.path.join(config_path, region_name, 'board_size.txt')
            tv_size = os.path.join(config_path, region_name, 'TV_size')
            if os.path.isfile(board_size):
                if os.path.isfile(tv_size):
                    raise IOError("Both board_size.txt and TV_size exist in {}"
                                  .format(os.path.join(config_path, region_name)))
                size_file = board_size
            elif os.path.isfile(tv_size):
                size_file = tv_size
            else:
                raise IOError("Neither board_size.txt nor TV_size exists in {}"
                              .format(os.path.join(config_path, region_name)))
            with open(size_file) as f:
                scrmW = float(f.readline())
                scrmH = float(f.readline())

            screens[region_name] = (scrmW, scrmH, scrpW, scrpH)
        return screens

    @staticmethod
    def load_cam_extrinsics(org_path, set_id):
        """load calibration matrices: R and T."""
        config_path = os.path.join(org_path, set_id, 'Config')
        extrinsics = {}
        for region_name in kpi_prediction_linker._get_region_names(org_path, set_id):
            R_file_path = os.path.join(config_path, region_name, 'R.txt')
            T_file_path = os.path.join(config_path, region_name, 'T.txt')

            R = np.loadtxt(R_file_path, delimiter=',')
            T = np.loadtxt(T_file_path)

            extrinsics[region_name] = (R, T)
        return extrinsics

    @staticmethod
    def load_region_labels(org_path, set_id):
        """Load a dict (region_name->region_label) for the set."""
        config_path = os.path.join(org_path, set_id, 'Config')
        labels = {}
        for region_name in kpi_prediction_linker._get_region_names(org_path, set_id):
            label_path = os.path.join(config_path, region_name, 'region_label.txt')
            if os.path.isfile(label_path):
                with open(label_path, 'r') as label_file:
                    label = label_file.read().strip()
            else:
                label = 'unknown'
            labels[region_name] = label
        return labels

    @staticmethod
    def screenPix2Phy(gaze_pix, scrmW, scrmH, scrpH, scrpW):
        """
        Convert gaze point pixel coordinates (x, y) to physical coordinates in mm (x, y).

        gaze_pix: pixel positions of gaze
        scrmW, scrmH: screen size in mm
        scrpW, scrpH: screen size in pixel
        """

        gaze_phy = np.zeros(2, dtype=float)
        # compute gaze on screen
        gaze_phy[0] = gaze_pix[0] * scrmW / scrpW
        gaze_phy[1] = gaze_pix[1] * scrmH / scrpH

        return gaze_phy

    @staticmethod
    def fromCamera2ScreenProj_3inp(gaze, R, T):
        """
        Calculates projection from camera space to screen spaces.

        gaze_cam = R*gaze_scr + T
        gaze_scr = R.transpose()*(gaze_cam - T)
        R.transpose() and R.inverse() are equal because R is a square orthogonal matrix
        Convert camera coordinates in mm (x, y, z) to screen coordinates in mm (x, y)
        """

        # build array for output gaze
        gaze_out = np.zeros(3)
        gaze_out[0] = gaze[0]
        gaze_out[1] = gaze[1]
        gaze_out[2] = gaze[2]
        # apply calibration results: R and T
        Rt = R.transpose()
        gaze_out = Rt.dot(gaze_out - T.transpose())
        return gaze_out

    @staticmethod
    def compute_theta_phi_from_unit_vector(x, y, z):
        """
        Computes theta phi angles from forward facing gaze vector in the camera space.

        Args:
        x (float): gaze vector x
        y (float): gaze vector y
        z (float): gaze vector z

        Returns:
        theta (float): gaze pitch in radian
        phi (float): gaze yaw in radian
        """
        theta = np.arcsin(-y)
        phi = np.arctan2(-x, -z)

        return theta, phi

    @staticmethod
    def compute_gaze_vector_from_theta_phi(theta, phi):
        """
        Computes gaze vector through theta and phi.

        Compute forward facing gaze vector (look vector) in the camera space from theta & phi
        Args:
        theta(float): gaze pitch in radians
        phi(float): gaze yaw in radians

        Returns:
        gaze_vec(array of floats): forward facing gaze vector in the camera space
        """

        # forward facing gaze vector in the camera space
        gaze_vec = np.zeros(3)
        gaze_vec[0] = -np.cos(theta) * np.sin(phi)
        gaze_vec[1] = -np.sin(theta)
        gaze_vec[2] = -np.cos(theta) * np.cos(phi)

        return gaze_vec

    @staticmethod
    def compute_PoR_from_theta_phi(theta, phi, pc_cam_mm, R, T):
        """
        Compute the intersection of gaze vector (generated using theta&phi) with the monitor plane.

        Args:
        theta (float): gaze pitch in radians
        phi (float): gaze yaw in radians
        pc_cam_mm (float): 3D pupil center coordinates in camera space in mm
        R, T (float): camera extrinsics with respect to the monitor plane

        Returns:
        PoR_x, PoR_y, PoR_z (float): point of regard coordinates in camera space in mm,
            or None if there is no intersection between the directed gaze ray and the plane.
        """

        # Calculate point of regard given theta and phi angles
        screenNormal = R
        #  Rotation matrix computed from mirror calibration
        screenNormal = screenNormal[:, 2]
        screenOrigin = T
        #  Translation vector computed from mirror calibration
        gaze_vec = kpi_prediction_linker.compute_gaze_vector_from_theta_phi(theta, phi)

        dNormalGazeVec = np.dot(screenNormal, gaze_vec)

        d = np.dot(screenNormal, screenOrigin)

        dNormalEyeCenter = np.dot(screenNormal, pc_cam_mm)

        t = (d - dNormalEyeCenter) / dNormalGazeVec
        flagBackwardIntersection = 'No'
        if t < 0:  # Intersects at a point behind gaze origin.
            flagBackwardIntersection = 'Yes'
        gaze_vec = np.expand_dims(gaze_vec, axis=1)
        intersectPoint = pc_cam_mm + t * gaze_vec
        PoR_x = intersectPoint[0]
        PoR_y = intersectPoint[1]
        PoR_z = intersectPoint[2]

        return PoR_x, PoR_y, PoR_z, flagBackwardIntersection

    def getDataFrame(self, predictions, kpi_dataframe):
        """Return final data frame."""

        predictions = [pred.split(' ') for pred in predictions]
        dfpred = pd.DataFrame([], columns=self.resultCols)

        for root_dir, gt_folder, gt_file_folder, \
            kpi_sets in zip(self.gcosmos,
                            self.kpi_GT_dir,
                            self.kpi_GT_file_folder_name,
                            self.kpisets):

            org_dir = root_dir.replace('postData', 'orgData')

            if 'single_eye' == self.gaze_origin:
                df = self.createTableFromSingleEyePred(predictions, root_dir, org_dir,
                                                       gt_folder, gt_file_folder, kpi_sets)
            else:
                df = self.createTableFromPred(predictions, root_dir, org_dir,
                                              gt_folder, gt_file_folder, kpi_sets)
            dfpred = pd.concat([dfpred, df], ignore_index=True, sort=True)

        if kpi_dataframe is None or kpi_dataframe.empty:
            dfmerged = dfpred
        else:
            dfmerged = pd.merge(dfpred,
                                kpi_dataframe,
                                on=['set_id', 'user_id', 'region_name', 'image_name'],
                                how='left')

        return dfmerged

    @staticmethod
    def _get_root_set(set_id):
        return set_id.rsplit('-', 1)[0]

    @staticmethod
    def _get_camera_id(set_id):
        split = set_id.rsplit('-', 1)
        if len(split) > 1:
            return split[1]

        # s400_KPI has no cam in the name.
        return 'unknown'

    def ConvertTableToDF(self, table):
        """Convert input table(list) to pandas data frame."""

        df = pd.DataFrame(table, columns=self.resultCols)
        df['frame_num'] = df.image_name.map(lambda x: x.split('.')[0])
        df['frame_num'] = df.frame_num.map(lambda x: x.split('_')[-1])

        df['root_set_id'] = df.set_id.map(self._get_root_set)
        df['orig_frame_id'] = df['set_id'] + "_" + df["user_id"] + \
            "_" + df.region_name.map(lambda r: r + "_" if r != "" else "") \
            + df["screen_pix_x"] + "_" + df["screen_pix_y"]
        df['unique_cap_id'] = df['root_set_id'] + "_" + df["user_id"] + \
            "_" + df["screen_pix_x"] + "_" + df["screen_pix_y"]
        df.frame_num = df.frame_num.astype(float)
        df = df.sort_values(by=['orig_frame_id', 'frame_num'])
        df['cam_err_x'] = df['cam_err_x'] / 10.0
        df['cam_err_y'] = df['cam_err_y'] / 10.0
        df['cam_err_z'] = df['cam_err_z'] / 10.0
        df['cam_err_xy'] = df['cam_err_xy'] / 10.0
        df['cam_error'] = df['cam_error'] / 10.0  # Convert error to cm.
        df['screen_phy_error'] = df['screen_phy_error'] / 10.0  # Convert error to cm.

        return df

    def createTableFromPred(self, predictions, root_dir, org_dir, gt_folder, gt_file_folder,
                            kpi_sets):
        """Constructs table(list) through prediction and KPI data frame.

        Entries in each line of a joint2 ground truth file:
        l[0]: path
        l[1]: frame id
        l[2:5] face bounding box
        l[6:8]: ground truth in camera space
        l[9:10] ground truth in screen space (mm)
        l[11:13] PnP based head pose angles
        l[14:16] Gaze angles theta-phi
        l[17:19] Unit norm gaze vector
        l[20:22] Head location
        l[23:25] Left eye center location
        l[26:28] Right eye center location

        Entries in joint model:
        l[0]: path + frame id
        l[1:3]: gt_x, y, z
        l[4:5]: gt_theta, phi
        l[6:8]: pred_x, y, z
        l[9:10] pred_theta, phi

        Entries in theta-phi model:
        l[0] = path + frame id
        l[1:2] = gt_theta, phi
        l[3:4] = pred_theta, phi
        """

        final_table = []
        # Read the prediction file
        orig_data = pd.DataFrame(predictions)
        if self.model_type == 'xyz':
            orig_data.columns = ["frame_id", "GT_x", "GT_y", "GT_z", "cam_x", "cam_y", "cam_z"]
        elif self.model_type in ['joint_xyz', 'joint_tp', 'joint']:
            orig_data.columns = ["frame_id", "GT_x", "GT_y", "GT_z", "GT_theta", "GT_phi",
                                 "cam_x", "cam_y", "cam_z", "cam_theta", "cam_phi"]
        elif self.model_type == 'theta_phi':
            orig_data.columns = ["frame_id", "GT_theta", "GT_phi", "cam_theta", "cam_phi"]

        gt_pr = orig_data.values

        indent = 8

        # construct dictionary for predictions
        pr_dict = {}
        for prediction in gt_pr:
            chunks = prediction[0].split('/')
            for c in chunks:
                if c == '':
                    continue
                if (c[0].lower() == 's' or 'germany' in c)\
                        and ('kpi' in c or 'KPI' in c or 'gaze' in c):
                    set_id = c
            frame_id = chunks[-1]
            region_name = chunks[-2]
            user_index = -3
            if not region_name.startswith('region_'):
                # Legacy flat structure, no regions.
                region_name = ''
                user_index += 1
            if 'MITData_DataFactory' in prediction[0]:
                user_index -= 1
            user_id = chunks[user_index]

            if set_id not in pr_dict:
                pr_dict[set_id] = {}

            if user_id not in pr_dict[set_id]:
                pr_dict[set_id][user_id] = {}

            if region_name not in pr_dict[set_id][user_id]:
                pr_dict[set_id][user_id][region_name] = {}

            pr_dict[set_id][user_id][region_name][frame_id] = [float(x) for x in prediction[1:]]

        print('gt_pr.shape: {}'.format(gt_pr.shape))

        if self._root_path is None:
            self._root_path = ''
        base_directory = os.path.join(self._root_path, root_dir)
        org_path = os.path.join(self._root_path, org_dir)

        for kpi_dir in kpi_sets:
            base_dir = os.path.join(base_directory, kpi_dir, gt_folder, gt_file_folder)

            screens = self.load_screen_parameters(org_path, kpi_dir)
            extrinsics = self.load_cam_extrinsics(org_path, kpi_dir)
            region_labels = self.load_region_labels(org_path, kpi_dir)

            sample_all = 0
            sample = 0
            sample_NA = 0
            # read the GT file with theta-phi conversions
            for kpi_file in self.kpi_GT_file:
                data_file = kpi_file + '.txt'

                if os.path.exists(os.path.join(base_dir, data_file)):
                    data_file = os.path.join(base_dir, data_file)
                else:
                    data_file = os.path.join(root_dir, data_file)

                if not os.path.exists(data_file):
                    print('ATTENTION: BASE GT FILE DOES NOT EXIST! \t', data_file)
                    continue

                for line in open(data_file):
                    gt_info = line.split(' ')
                    gt_info = gt_info[:6] + gt_info[indent + 6:]
                    sample_all += 1

                    region_name = gt_info[0].split('/')[-1]
                    if not region_name.startswith('region_'):  # No regions
                        set_id = gt_info[0].split('/')[-3]
                        user_id = gt_info[0].split('/')[-1]
                        region_name = ''
                    else:
                        set_id = gt_info[0].split('/')[-4]
                        user_id = gt_info[0].split('/')[-2]
                    frame_id = gt_info[1]

                    if set_id in pr_dict:
                        if user_id in pr_dict[set_id]:
                            if region_name in pr_dict[set_id][user_id]:
                                if frame_id in pr_dict[set_id][user_id][region_name]:
                                    pr = pr_dict[set_id][user_id][region_name][frame_id]
                                else:
                                    print('Missing frame id:', frame_id, '\t in ', gt_info[0])
                                    continue
                            else:
                                print('Missing region name:', region_name, '\t in ', gt_info[0])
                                continue
                        else:
                            print('Missing user id:', user_id, '\t in ', gt_info[0])
                            continue
                    else:
                        print('Missing set id:', set_id, '\t in ', gt_info[0])
                        continue

                    GT_screen_px = np.zeros((2, 1), dtype=float)
                    GT_screen_px[0] = float(gt_info[9])
                    GT_screen_px[1] = float(gt_info[10])

                    # GT on screen plane in mm
                    GT_screen_mm = self.screenPix2Phy(GT_screen_px, *screens[region_name])
                    GT_camera_mm = np.zeros((3, 1), dtype=float)

                    # Read GT values
                    GT_camera_mm[0] = gt_info[6]
                    GT_camera_mm[1] = gt_info[7]
                    GT_camera_mm[2] = gt_info[8]

                    # Center between two eyes
                    gaze_origin_cam_mm = np.zeros((3, 1), dtype=float)
                    gaze_origin_cam_mm[2][0] = 800.0

                    # Read HP GT values
                    GT_hp_theta = float(gt_info[25])
                    GT_hp_phi = float(gt_info[26])

                    # GT_theta, GT_phi
                    if self.gaze_origin == 'normal':
                        GT_theta = float(gt_info[14])
                        GT_phi = float(gt_info[15])
                    elif self.gaze_origin == 'mid_eyes':
                        GT_theta = float(gt_info[27])
                        GT_phi = float(gt_info[28])
                    elif self.gaze_origin == 'left_eye':
                        GT_theta = float(gt_info[29])
                        GT_phi = float(gt_info[30])
                    elif self.gaze_origin == 'right_eye':
                        GT_theta = float(gt_info[31])
                        GT_phi = float(gt_info[32])
                    elif self.gaze_origin == 'delta':
                        GT_theta = float(gt_info[27])
                        GT_phi = float(gt_info[28])
                        if GT_theta != -1.0:
                            GT_theta -= GT_hp_theta
                        if GT_phi != -1.0:
                            GT_phi -= GT_hp_phi
                    elif self.gaze_origin == 'headnorm_face':
                        GT_theta = float(gt_info[37])
                        GT_phi = float(gt_info[38])
                        cnvMat = np.zeros((3, 3), dtype=np.float32)
                        cnvMat[0][0] = float(gt_info[62])
                        cnvMat[0][1] = float(gt_info[63])
                        cnvMat[0][2] = float(gt_info[64])
                        cnvMat[1][0] = float(gt_info[65])
                        cnvMat[1][1] = float(gt_info[66])
                        cnvMat[1][2] = float(gt_info[67])
                        cnvMat[2][0] = float(gt_info[68])
                        cnvMat[2][1] = float(gt_info[69])
                        cnvMat[2][2] = float(gt_info[70])
                    elif self.gaze_origin == 'headnorm_leye':
                        GT_theta = float(gt_info[41])
                        GT_phi = float(gt_info[42])
                        cnvMat = np.zeros((3, 3), dtype=np.float32)
                        cnvMat[0][0] = float(gt_info[71])
                        cnvMat[0][1] = float(gt_info[72])
                        cnvMat[0][2] = float(gt_info[73])
                        cnvMat[1][0] = float(gt_info[74])
                        cnvMat[1][1] = float(gt_info[75])
                        cnvMat[1][2] = float(gt_info[76])
                        cnvMat[2][0] = float(gt_info[77])
                        cnvMat[2][1] = float(gt_info[78])
                        cnvMat[2][2] = float(gt_info[79])
                    elif self.gaze_origin == 'headnorm_reye':
                        GT_theta = float(gt_info[45])
                        GT_phi = float(gt_info[46])
                        cnvMat = np.zeros((3, 3), dtype=np.float32)
                        cnvMat[0][0] = float(gt_info[80])
                        cnvMat[0][1] = float(gt_info[81])
                        cnvMat[0][2] = float(gt_info[82])
                        cnvMat[1][0] = float(gt_info[83])
                        cnvMat[1][1] = float(gt_info[84])
                        cnvMat[1][2] = float(gt_info[85])
                        cnvMat[2][0] = float(gt_info[86])
                        cnvMat[2][1] = float(gt_info[87])
                        cnvMat[2][2] = float(gt_info[88])
                    else:
                        raise NotImplementedError('Gaze origin {} is not implemented, '
                                                  'please choose from one of the following:'
                                                  'normal, left_eye, right_eye, mid_eyes, '
                                                  'delta, headnorm_face,'
                                                  'headnorm_leye, headnorm_reye'
                                                  .format(self.gaze_origin))

                    # Ground truth mid point of two eye centers
                    GT_mid_ec = np.zeros((3, 1), dtype=float)
                    GT_mid_ec[0] = float(gt_info[16])
                    GT_mid_ec[1] = float(gt_info[17])
                    GT_mid_ec[2] = float(gt_info[18])

                    # Ground truth left eye center
                    GT_le_ec = np.zeros((3, 1), dtype=float)
                    GT_le_ec[0] = float(gt_info[19])
                    GT_le_ec[1] = float(gt_info[20])
                    GT_le_ec[2] = float(gt_info[21])

                    # Ground truth right eye center
                    GT_re_ec = np.zeros((3, 1), dtype=float)
                    GT_re_ec[0] = float(gt_info[22])
                    GT_re_ec[1] = float(gt_info[23])
                    GT_re_ec[2] = float(gt_info[24])

                    valid_left_eye = False
                    if GT_le_ec[0] != -1.0 and GT_le_ec[1] != -1.0 and GT_le_ec[2] != -1.0:
                        valid_left_eye = True

                    valid_right_eye = False
                    if GT_re_ec[0] != -1.0 and GT_re_ec[1] != -1.0 and GT_re_ec[2] != -1.0:
                        valid_right_eye = True

                    if valid_left_eye and valid_right_eye:
                        gaze_origin_cam_mm = GT_mid_ec
                    elif valid_left_eye is True and valid_right_eye is False:
                        gaze_origin_cam_mm = GT_le_ec
                    elif valid_left_eye is False and valid_right_eye is True:
                        gaze_origin_cam_mm = GT_re_ec

                    if 'headnorm_face' == self.gaze_origin:
                        gaze_origin_cam_mm[0] = float(gt_info[89])
                        gaze_origin_cam_mm[1] = float(gt_info[90])
                        gaze_origin_cam_mm[2] = float(gt_info[91])
                    elif 'headnorm_leye' == self.gaze_origin:
                        gaze_origin_cam_mm = GT_le_ec
                    elif 'headnorm_reye' == self.gaze_origin:
                        gaze_origin_cam_mm = GT_re_ec

                    # Calculate gaze vector
                    GT_gaze_vec = GT_camera_mm - gaze_origin_cam_mm
                    GT_gaze_vec = np.squeeze(GT_gaze_vec, axis=1)
                    GT_gaze_vec_mag = np.sqrt(
                        GT_gaze_vec[0] ** 2 + GT_gaze_vec[1] ** 2 + GT_gaze_vec[2] ** 2)
                    GT_gaze_vec = GT_gaze_vec / GT_gaze_vec_mag
                    d_deg_err_direct = 0

                    # Read/calculate prediction gaze_cam_mm and gaze_theta, gaze_phi
                    if self.model_type == 'xyz' or 'joint' in self.model_type:
                        if int(GT_camera_mm[0]) != int(pr[0]) \
                                or int(GT_camera_mm[1]) != int(pr[1]) \
                                or int(GT_camera_mm[2]) != int(pr[2]):
                            print('GT - Data file mismatch! Sample: ',
                                  sample, GT_camera_mm[0],
                                  pr[0], GT_camera_mm[1], pr[1], GT_camera_mm[2], pr[2])
                            continue

                        if self.model_type == 'xyz':
                            gaze_cam_mm = np.zeros((3, 1), dtype=float)
                            gaze_cam_mm[0] = pr[3]
                            gaze_cam_mm[1] = pr[4]
                            gaze_cam_mm[2] = pr[5]
                            gaze_theta = -1.0
                            gaze_phi = -1.0

                        # 'joint_xyz', 'joint_tp', or 'joint'.
                        elif 'joint' in self.model_type:
                            gaze_cam_mm = np.zeros((3, 1), dtype=float)
                            gaze_cam_mm[0] = pr[5]
                            gaze_cam_mm[1] = pr[6]
                            gaze_cam_mm[2] = pr[7]
                            gaze_theta = pr[8]
                            gaze_phi = pr[9]
                            if self.theta_phi_degrees:
                                gaze_theta = (gaze_theta * math.pi) / 180
                                gaze_phi = (gaze_phi * math.pi) / 180

                            if self.gaze_origin == 'delta':
                                gaze_theta += GT_hp_theta
                                gaze_phi += GT_hp_phi

                    elif self.model_type == 'theta_phi':
                        if abs(GT_theta - float(pr[0])) > 1e-2 or \
                                abs(GT_phi - float(pr[1])) > 1e-2:
                            print('GT - Data file mismatch! Sample: ',
                                  sample, GT_theta, pr[0], GT_phi, pr[1])
                            continue

                        GT_ngv = self.compute_gaze_vector_from_theta_phi(GT_theta, GT_phi)
                        gaze_theta = pr[2]
                        gaze_phi = pr[3]
                        ngv = self.compute_gaze_vector_from_theta_phi(gaze_theta, gaze_phi)
                        d_deg_err_direct = np.arccos(np.dot(GT_ngv, ngv)) * 180 / np.pi

                        if self.theta_phi_degrees:
                            gaze_theta = (gaze_theta * math.pi) / 180
                            gaze_phi = (gaze_phi * math.pi) / 180

                        if self.gaze_origin == 'delta':
                            gaze_theta += GT_hp_theta
                            gaze_phi += GT_hp_phi

                        if 'headnorm' in self.gaze_origin:
                            ngv = self.compute_gaze_vector_from_theta_phi(gaze_theta, gaze_phi)
                            gv = np.dot(np.linalg.inv(cnvMat), ngv)
                            gaze_theta, gaze_phi = self.compute_theta_phi_from_unit_vector(
                                gv[0], gv[1], gv[2])

                    # Calculate gaze_cam_mm according to the model type.
                    flagBackwardIntersection = 'No'
                    if self.model_type ['theta_phi', 'joint_tp']:
                        if GT_theta == -1.0 and GT_phi == -1.0:
                            print('Invalid Theta-Phi sample: {} {} {} {} {}'
                                  .format(sample, set_id, user_id, region_name, frame_id))
                            continue
                        gaze_cam_mm = self.compute_PoR_from_theta_phi(
                            gaze_theta, gaze_phi, gaze_origin_cam_mm,
                            *extrinsics[region_name])
                        flagBackwardIntersection = gaze_cam_mm[3]
                        if flagBackwardIntersection == 'Yes':
                            print('No forward gaze vector screen plane intersection: '
                                    '{} {} {} {} {}'
                                    .format(sample, set_id, user_id, region_name, frame_id))
                            sample_NA += 1

                        gaze_cam_mm = np.array([gaze_cam_mm[0], gaze_cam_mm[1], gaze_cam_mm[2]])

                    elif self.model_type == 'joint':
                        # Combine gaze from TP and XYZ portions.
                        if GT_theta != -1.0 and GT_phi != -1.0:
                            tp_gaze_cam_mm = self.compute_PoR_from_theta_phi(
                                gaze_theta, gaze_phi, gaze_origin_cam_mm,
                                *extrinsics[region_name])
                            flagBackwardIntersection = tp_gaze_cam_mm[3]
                            if flagBackwardIntersection == 'No':
                                tp_gaze_cam_mm = np.array(
                                    [tp_gaze_cam_mm[0], tp_gaze_cam_mm[1], tp_gaze_cam_mm[2]])

                                gaze_cam_mm = (gaze_cam_mm + tp_gaze_cam_mm) / 2.0
                                gaze_cam_mm[2] = tp_gaze_cam_mm[2]
                            else:
                                print('No forward gaze vector screen plane intersection, '
                                      'but use XYZ portion sample: {} {} {} {} {}'
                                      .format(sample, set_id, user_id, region_name, frame_id))
                                sample_NA += 1
                        else:
                            print('Invalid Theta-Phi but use XYZ portion sample: {} {} {} {} {}'
                                  .format(sample, set_id, user_id, region_name, frame_id))
                    # else: #self.model_type == 'xyz' or self.model_type == 'joint_xyz'
                    # or self.model_type == 'joint':
                    # no need to have an additional step, but directly use gaze_cam_mm set from file

                    # Mind the indent here, this should not go under elif case above
                    # d-x, d-y, d-z, d-XY, d-XYZ, and degree error calculation.
                    x_cam_err = abs(GT_camera_mm[0] - gaze_cam_mm[0])[0]
                    y_cam_err = abs(GT_camera_mm[1] - gaze_cam_mm[1])[0]
                    z_cam_err = abs(GT_camera_mm[2] - gaze_cam_mm[2])[0]

                    GT_xy_vec = np.array([GT_camera_mm[0], GT_camera_mm[1]])
                    gaze_xy_vec = np.array([gaze_cam_mm[0], gaze_cam_mm[1]])
                    xy_cam_err = abs(GT_xy_vec - gaze_xy_vec)
                    d_xy_cam_err = np.linalg.norm(xy_cam_err)

                    xyz_cam_err = abs(GT_camera_mm - gaze_cam_mm)
                    d_xyz_cam_err = np.linalg.norm(xyz_cam_err)

                    gaze_scr_mm = self.fromCamera2ScreenProj_3inp(gaze_cam_mm,
                                                                  *extrinsics[region_name])
                    xyz_scr_err = abs(GT_screen_mm - gaze_scr_mm[0:2])
                    d_xyz_scr_err = np.linalg.norm(xyz_scr_err)
                    user_dist = np.linalg.norm(gaze_origin_cam_mm - GT_camera_mm)

                    gaze_vec = gaze_cam_mm - gaze_origin_cam_mm
                    gaze_vec = np.squeeze(gaze_vec, axis=1)
                    gaze_vec_mag = np.sqrt(
                        gaze_vec[0] ** 2 + gaze_vec[1] ** 2 + gaze_vec[2] ** 2)
                    gaze_vec = gaze_vec / gaze_vec_mag
                    d_xyz_err_deg = np.arccos(np.dot(GT_gaze_vec, gaze_vec)) * 180 / np.pi

                    if flagBackwardIntersection == 'Yes':
                        d_xyz_err_deg = d_deg_err_direct

                    if abs(d_xyz_err_deg-d_deg_err_direct) > 1.0:
                        print('Error calculation might be problematic -> direct:' +
                              str(d_deg_err_direct) + ', post:' + str(d_xyz_err_deg))

                    # Calculate ref gaze vector for 45-deg cone
                    ref_gaze_vec = np.zeros((3, 1), dtype=float)
                    ref_gaze_vec[2] = -1.0
                    # Calculate angle between GT_gaze_vec and ref_gaze_vec
                    # for 45-deg cone, this should not be larger than 45/2
                    d_cone_deg = np.arccos(np.dot(GT_gaze_vec, ref_gaze_vec)) * 180 / np.pi

                    # if valid_overall_gaze:
                    row = [gt_info[0], gt_info[1]]
                    row += [set_id, user_id, region_name, region_labels[region_name]]
                    row += [str(int(float(gt_info[9]))), str(int(float(gt_info[10])))]
                    row += GT_camera_mm.flatten().tolist() + [np.linalg.norm(GT_camera_mm)]
                    row += [GT_theta, GT_phi]
                    row += gaze_origin_cam_mm.flatten().tolist()
                    row += [GT_hp_theta, GT_hp_phi]
                    row += gaze_cam_mm.flatten().tolist() + [np.linalg.norm(gaze_cam_mm)]
                    row += [gaze_theta, gaze_phi]
                    row += [x_cam_err, y_cam_err, z_cam_err, d_xy_cam_err]
                    row += [d_xyz_cam_err, d_xyz_scr_err, user_dist, d_xyz_err_deg,
                            d_deg_err_direct, flagBackwardIntersection, d_cone_deg]

                    final_table.append(row)
                    sample += 1
            print('KPI: ' + kpi_dir + ' is completed!')
            print('No of existing samples:' + str(sample_all))
            print('No of total evaluated samples:' + str(sample))
            print('No of no-forward-intersection samples:' + str(sample_NA))

        return self.ConvertTableToDF(final_table)

    def createTableFromSingleEyePred(self, predictions, root_dir, org_dir, gt_folder,
                                     gt_file_folder, kpi_sets):
        """Constructs table(list) through prediction and KPI data frame.

        Entries in each line of a joint2 ground truth file:
        l[0]: path
        l[1]: frame id
        l[2:5] face bounding box
        l[6:8]: ground truth in camera space
        l[9:10] ground truth in screen space (mm)
        l[11:13] PnP based head pose angles
        l[14:16] Gaze angles theta-phi
        l[17:19] Unit norm gaze vector
        l[20:22] Head location
        l[23:25] Left eye center location
        l[26:28] Right eye center location
        """

        final_table = []
        # Read the prediction file
        orig_data = pd.DataFrame(predictions)
        orig_data.columns = ["frame_id", "GT_theta", "GT_phi", "cam_theta", "cam_phi"]
        gt_pr = orig_data.values

        indent = 8

        # construct dictionary for predictions
        print('gt_pr.shape: {}'.format(gt_pr.shape))

        sample = 0
        ind = 0
        pr_dict = {}
        while ind < gt_pr.shape[0]:
            chunks = predictions[ind][0].split('/')
            for c in chunks:
                if c == '':
                    continue
                if (c[0].lower() == 's' or 'germany' in c)\
                        and ('kpi' in c or 'KPI' in c or 'gaze' in c):
                    set_id = c
            frame_id = chunks[-1]
            region_name = chunks[-2]
            user_index = -3
            if not region_name.startswith('region_'):
                # Legacy flat structure, no regions.
                region_name = ''
                user_index += 1
            if 'MITData_DataFactory' in predictions[ind][0]:
                user_index -= 1
            user_id = chunks[user_index]

            if set_id not in pr_dict:
                pr_dict[set_id] = {}

            if user_id not in pr_dict[set_id]:
                pr_dict[set_id][user_id] = {}

            if region_name not in pr_dict[set_id][user_id]:
                pr_dict[set_id][user_id][region_name] = {}

            if frame_id not in pr_dict[set_id][user_id][region_name]:
                pr_dict[set_id][user_id][region_name][frame_id] = {}

            eye_id = 0
            pr_dict[set_id][user_id][region_name][frame_id][eye_id] = \
                [float(x) for x in predictions[ind][1:]]

            eye_id = 1
            pr_dict[set_id][user_id][region_name][frame_id][eye_id] = \
                [float(x) for x in predictions[ind+1][1:]]
            ind += 2
            sample += 1

        print('Parsed samples:', sample)
        if self._root_path is None:
            self._root_path = ''
        base_directory = os.path.join(self._root_path, root_dir)
        org_path = os.path.join(self._root_path, org_dir)

        for kpi_dir in kpi_sets:
            base_dir = os.path.join(base_directory, kpi_dir, gt_folder, gt_file_folder)
            print('kpi_dir:', kpi_dir)
            print('gt_folder:', gt_folder)
            print('gt_file_folder:', gt_file_folder)
            print('base_dir:', base_dir)

            screens = self.load_screen_parameters(org_path, kpi_dir)
            extrinsics = self.load_cam_extrinsics(org_path, kpi_dir)
            region_labels = self.load_region_labels(org_path, kpi_dir)

            sample_all = 0
            sample = 0
            sample_NA = 0
            # read the GT file with theta-phi conversions
            for kpi_file in self.kpi_GT_file:
                data_file = kpi_file + '.txt'

                if os.path.exists(os.path.join(base_dir, data_file)):
                    data_file = os.path.join(base_dir, data_file)
                else:
                    data_file = os.path.join(root_dir, data_file)

                if not os.path.exists(data_file):
                    print('ATTENTION: BASE GT FILE DOES NOT EXIST! \t', data_file)
                    continue

                for line in open(data_file):
                    gt_info = line.split(' ')
                    gt_info = gt_info[:6] + gt_info[indent + 6:]
                    sample_all += 1

                    region_name = gt_info[0].split('/')[-1]
                    if not region_name.startswith('region_'):  # No regions
                        set_id = gt_info[0].split('/')[-3]
                        user_id = gt_info[0].split('/')[-1]
                        region_name = ''
                    else:
                        set_id = gt_info[0].split('/')[-4]
                        user_id = gt_info[0].split('/')[-2]
                    frame_id = gt_info[1]

                    if set_id in pr_dict:
                        if user_id in pr_dict[set_id]:
                            if region_name in pr_dict[set_id][user_id]:
                                if frame_id in pr_dict[set_id][user_id][region_name]:
                                    pr_leye = pr_dict[set_id][user_id][region_name][frame_id][0]
                                    pr_reye = pr_dict[set_id][user_id][region_name][frame_id][1]
                                    # According to dataloader take neg of right eye gaze yaw
                                    pr_reye[1] *= -1.0
                                    pr_reye[3] *= -1.0
                                else:
                                    print('Missing frame id:', frame_id, '\t in ', gt_info[0])
                                    continue
                            else:
                                print('Missing region name:', region_name, '\t in ', gt_info[0])
                                continue
                        else:
                            print('Missing user id:', user_id, '\t in ', gt_info[0])
                            continue
                    else:
                        print('Missing set id:', set_id, '\t in ', gt_info[0])
                        continue

                    # Left eye CnvMat
                    cnvMat_leye = np.zeros((3, 3), dtype=np.float32)
                    cnvMat_leye[0][0] = float(gt_info[71])
                    cnvMat_leye[0][1] = float(gt_info[72])
                    cnvMat_leye[0][2] = float(gt_info[73])
                    cnvMat_leye[1][0] = float(gt_info[74])
                    cnvMat_leye[1][1] = float(gt_info[75])
                    cnvMat_leye[1][2] = float(gt_info[76])
                    cnvMat_leye[2][0] = float(gt_info[77])
                    cnvMat_leye[2][1] = float(gt_info[78])
                    cnvMat_leye[2][2] = float(gt_info[79])

                    # Denormalize left eye gaze
                    ngv_leye = self.compute_gaze_vector_from_theta_phi(pr_leye[2], pr_leye[3])
                    gv_leye = np.dot(np.linalg.inv(cnvMat_leye), ngv_leye)
                    gaze_theta_leye, gaze_phi_leye = self.compute_theta_phi_from_unit_vector(
                        gv_leye[0], gv_leye[1], gv_leye[2])

                    # Right eye CnvMat
                    cnvMat_reye = np.zeros((3, 3), dtype=np.float32)
                    cnvMat_reye[0][0] = float(gt_info[80])
                    cnvMat_reye[0][1] = float(gt_info[81])
                    cnvMat_reye[0][2] = float(gt_info[82])
                    cnvMat_reye[1][0] = float(gt_info[83])
                    cnvMat_reye[1][1] = float(gt_info[84])
                    cnvMat_reye[1][2] = float(gt_info[85])
                    cnvMat_reye[2][0] = float(gt_info[86])
                    cnvMat_reye[2][1] = float(gt_info[87])
                    cnvMat_reye[2][2] = float(gt_info[88])

                    # Denormalize right eye gaze
                    ngv_reye = self.compute_gaze_vector_from_theta_phi(pr_reye[2], pr_reye[3])
                    gv_reye = np.dot(np.linalg.inv(cnvMat_reye), ngv_reye)
                    gaze_theta_reye, gaze_phi_reye = self.compute_theta_phi_from_unit_vector(
                        gv_reye[0], gv_reye[1], gv_reye[2])

                    GT_screen_px = np.zeros((2, 1), dtype=float)
                    GT_screen_px[0] = float(gt_info[9])
                    GT_screen_px[1] = float(gt_info[10])

                    # GT on screen plane in mm
                    GT_screen_mm = self.screenPix2Phy(GT_screen_px, *screens[region_name])
                    GT_camera_mm = np.zeros((3, 1), dtype=float)

                    # Read GT values
                    GT_camera_mm[0] = gt_info[6]
                    GT_camera_mm[1] = gt_info[7]
                    GT_camera_mm[2] = gt_info[8]

                    # Center between two eyes
                    gaze_origin_cam_mm = np.zeros((3, 1), dtype=float)
                    gaze_origin_cam_mm[2][0] = 80.0

                    # Read HP GT values
                    GT_hp_theta = float(gt_info[25])
                    GT_hp_phi = float(gt_info[26])

                    # Ground truth mid point of two eye centers
                    GT_mid_ec = np.zeros((3, 1), dtype=float)
                    GT_mid_ec[0] = float(gt_info[16])
                    GT_mid_ec[1] = float(gt_info[17])
                    GT_mid_ec[2] = float(gt_info[18])

                    # Ground truth left eye center
                    GT_le_ec = np.zeros((3, 1), dtype=float)
                    GT_le_ec[0] = float(gt_info[19])
                    GT_le_ec[1] = float(gt_info[20])
                    GT_le_ec[2] = float(gt_info[21])

                    # Ground truth right eye center
                    GT_re_ec = np.zeros((3, 1), dtype=float)
                    GT_re_ec[0] = float(gt_info[22])
                    GT_re_ec[1] = float(gt_info[23])
                    GT_re_ec[2] = float(gt_info[24])

                    # Ground truth left eye norm gaze
                    GT_norm_gaze_theta_leye = float(gt_info[41])
                    GT_norm_gaze_phi_leye = float(gt_info[42])

                    # Ground truth right eye norm gaze
                    GT_norm_gaze_theta_reye = float(gt_info[45])
                    GT_norm_gaze_phi_reye = float(gt_info[46])

                    # Discard samples that do not match in GT file and prediction file
                    if abs(GT_norm_gaze_theta_leye - float(pr_leye[0])) > 1e-2 or \
                            abs(GT_norm_gaze_phi_leye - float(pr_leye[1])) > 1e-2 or \
                            abs(GT_norm_gaze_theta_reye - float(pr_reye[0])) > 1e-2 or \
                            abs(GT_norm_gaze_phi_reye - float(pr_reye[1])) > 1e-2:
                        print('GT - Data file mismatch! Sample: ',
                              sample, GT_norm_gaze_theta_leye, pr_leye[0], GT_norm_gaze_phi_leye,
                              pr_leye[1], GT_norm_gaze_theta_reye, pr_reye[0],
                              GT_norm_gaze_phi_reye, pr_reye[1])
                        continue

                    # Discard samples that are invalid
                    if (GT_norm_gaze_theta_leye == -1.0 and GT_norm_gaze_phi_leye == -1.0) or \
                            (GT_norm_gaze_theta_reye == -1.0 and GT_norm_gaze_phi_reye == -1.0):
                        print('Invalid Theta-Phi sample: {} {} {} {} {}'
                              .format(sample, set_id, user_id, region_name, frame_id))
                        continue

                    # Check validity of left and right eye centers in 3D
                    valid_left_eye = False
                    if GT_le_ec[0] != -1.0 and GT_le_ec[1] != -1.0 and GT_le_ec[2] != -1.0:
                        valid_left_eye = True

                    valid_right_eye = False
                    if GT_re_ec[0] != -1.0 and GT_re_ec[1] != -1.0 and GT_re_ec[2] != -1.0:
                        valid_right_eye = True

                    # TODO(marar) add more sophisticated fusion algos
                    # Simple head pose based fusion
                    if GT_hp_phi > 0.5:
                        valid_right_eye = False
                    elif GT_hp_phi < -0.5:
                        valid_left_eye = False

                    GT_gaze_theta_norm = (GT_norm_gaze_theta_leye +
                                          GT_norm_gaze_theta_reye) * 0.5
                    GT_gaze_phi_norm = (GT_norm_gaze_phi_leye + GT_norm_gaze_phi_reye) * 0.5
                    if valid_left_eye and valid_right_eye:
                        gaze_origin_cam_mm = GT_mid_ec
                        gaze_theta = (gaze_theta_leye + gaze_theta_reye) * 0.5
                        gaze_phi = (gaze_phi_leye + gaze_phi_reye) * 0.5
                        gaze_theta_norm = (pr_leye[2] + pr_reye[2]) * 0.5
                        gaze_phi_norm = (pr_leye[3] + pr_reye[3]) * 0.5
                    elif valid_left_eye is True and valid_right_eye is False:
                        gaze_origin_cam_mm = GT_le_ec
                        gaze_theta = gaze_theta_leye
                        gaze_phi = gaze_phi_leye
                        gaze_theta_norm = pr_leye[2]
                        gaze_phi_norm = pr_leye[3]
                        GT_gaze_theta_norm = GT_norm_gaze_theta_leye
                        GT_gaze_phi_norm = GT_norm_gaze_phi_leye
                    elif valid_left_eye is False and valid_right_eye is True:
                        gaze_origin_cam_mm = GT_re_ec
                        gaze_theta = gaze_theta_reye
                        gaze_phi = gaze_phi_reye
                        gaze_theta_norm = pr_reye[2]
                        gaze_phi_norm = pr_reye[3]
                        GT_gaze_theta_norm = GT_norm_gaze_theta_reye
                        GT_gaze_phi_norm = GT_norm_gaze_phi_reye

                    ngv = self.compute_gaze_vector_from_theta_phi(gaze_theta_norm, gaze_phi_norm)
                    GT_ngv = self.compute_gaze_vector_from_theta_phi(GT_gaze_theta_norm,
                                                                     GT_gaze_phi_norm)
                    d_deg_err_direct = np.arccos(np.dot(GT_ngv, ngv)) * 180 / np.pi

                    # Calculate GT gaze vector
                    GT_gaze_vec = GT_camera_mm - gaze_origin_cam_mm
                    GT_gaze_vec = np.squeeze(GT_gaze_vec, axis=1)
                    GT_gaze_vec_mag = np.sqrt(
                        GT_gaze_vec[0] ** 2 + GT_gaze_vec[1] ** 2 + GT_gaze_vec[2] ** 2)
                    GT_gaze_vec = GT_gaze_vec / GT_gaze_vec_mag

                    # Calculate gaze_cam_mm
                    flagBackwardIntersection = 'No'
                    gaze_cam_mm = self.compute_PoR_from_theta_phi(
                        gaze_theta, gaze_phi, gaze_origin_cam_mm,
                        *extrinsics[region_name])
                    flagBackwardIntersection = gaze_cam_mm[3]
                    if flagBackwardIntersection == 'Yes':
                        print('No forward gaze vector screen plane intersection: '
                              '{} {} {} {} {}'
                              .format(sample, set_id, user_id, region_name, frame_id))
                        sample_NA += 1

                    gaze_cam_mm = np.array([gaze_cam_mm[0], gaze_cam_mm[1], gaze_cam_mm[2]])

                    # Mind the indent here, this should not go under elif case above
                    # d-x, d-y, d-z, d-XY, d-XYZ, and degree error calculation.
                    x_cam_err = abs(GT_camera_mm[0] - gaze_cam_mm[0])[0]
                    y_cam_err = abs(GT_camera_mm[1] - gaze_cam_mm[1])[0]
                    z_cam_err = abs(GT_camera_mm[2] - gaze_cam_mm[2])[0]

                    GT_xy_vec = np.array([GT_camera_mm[0], GT_camera_mm[1]])
                    gaze_xy_vec = np.array([gaze_cam_mm[0], gaze_cam_mm[1]])
                    xy_cam_err = abs(GT_xy_vec - gaze_xy_vec)
                    d_xy_cam_err = np.linalg.norm(xy_cam_err)

                    xyz_cam_err = abs(GT_camera_mm - gaze_cam_mm)
                    d_xyz_cam_err = np.linalg.norm(xyz_cam_err)

                    gaze_scr_mm = self.fromCamera2ScreenProj_3inp(gaze_cam_mm,
                                                                  *extrinsics[region_name])
                    xyz_scr_err = abs(GT_screen_mm - gaze_scr_mm[0:2])
                    d_xyz_scr_err = np.linalg.norm(xyz_scr_err)
                    user_dist = np.linalg.norm(gaze_origin_cam_mm - GT_camera_mm)

                    gaze_vec = gaze_cam_mm - gaze_origin_cam_mm
                    gaze_vec = np.squeeze(gaze_vec, axis=1)
                    gaze_vec_mag = np.sqrt(
                        gaze_vec[0] ** 2 + gaze_vec[1] ** 2 + gaze_vec[2] ** 2)
                    gaze_vec = gaze_vec / gaze_vec_mag
                    d_xyz_err_deg = np.arccos(np.dot(GT_gaze_vec, gaze_vec)) * 180 / np.pi

                    if flagBackwardIntersection == 'Yes':
                        d_xyz_err_deg = d_deg_err_direct

                    if abs(d_xyz_err_deg-d_deg_err_direct) > 1.0:
                        print('Error calculation might be problematic -> direct:' +
                              str(d_deg_err_direct) + ', post:' + str(d_xyz_err_deg))

                    # Calculate ref gaze vector for 45-deg cone
                    ref_gaze_vec = np.zeros((3, 1), dtype=float)
                    ref_gaze_vec[2] = -1.0
                    # Calculate angle between GT_gaze_vec and ref_gaze_vec
                    # for 45-deg cone, this should not be larger than 45/2
                    d_cone_deg = np.arccos(np.dot(GT_gaze_vec, ref_gaze_vec)) * 180 / np.pi

                    # if valid_overall_gaze:
                    row = [gt_info[0], gt_info[1]]
                    row += [set_id, user_id, region_name, region_labels[region_name]]
                    row += [str(int(float(gt_info[9]))), str(int(float(gt_info[10])))]
                    row += GT_camera_mm.flatten().tolist() + [np.linalg.norm(GT_camera_mm)]
                    row += [GT_gaze_theta_norm, GT_gaze_phi_norm]
                    row += gaze_origin_cam_mm.flatten().tolist()
                    row += [GT_hp_theta, GT_hp_phi]
                    row += gaze_cam_mm.flatten().tolist() + [np.linalg.norm(gaze_cam_mm)]
                    row += [gaze_theta, gaze_phi]
                    row += [x_cam_err, y_cam_err, z_cam_err, d_xy_cam_err]
                    row += [d_xyz_cam_err, d_xyz_scr_err, user_dist, d_xyz_err_deg,
                            d_deg_err_direct, flagBackwardIntersection, d_cone_deg]

                    final_table.append(row)
                    sample += 1
            print('KPI: ' + kpi_dir + ' is completed!')
            print('No of existing samples:' + str(sample_all))
            print('No of total evaluated samples:' + str(sample))
            print('No of no-forward-intersection samples:' + str(sample_NA))

        return self.ConvertTableToDF(final_table)


class KPIbuckets():
    """KPIbuckets bucketizes predictions once predictions and KPI data frame are linked."""

    centercams = ['s400_KPI', 's457-gaze-kpi-2', 's434-kpi-3', 's435-kpi-3',
                  'germany-2-kpi-2 ', 'germany-right-4-kpi-2', 's464-gaze-kpi-1',
                  's465-gaze-kpi-1']
    badUsers = ['user8_x', 'user9_x', 'user10_x']
    badGTs = ['s434-kpi_TNg6PdPzFMmNKuZgr7BbSdvabP5fFjjqJyXBFQQ-od8=_1905_15',
              's457-gaze-kpi_ZLvDTe7WQmc84UYZ_960_1065']
    car_region_max_x = 700
    car_region_min_x = -1420
    car_region_max_y = 315
    car_region_min_y = -355

    car_center_region_max_x = 700
    car_center_region_min_x = -540  # only change
    car_center_region_max_y = 315
    car_center_region_min_y = -355

    car_right_region_max_x = -540  # only change
    car_right_region_min_x = -1420
    car_right_region_max_y = 315
    car_right_region_min_y = -355

    def __init__(self, df, bad_users):
        """
        Initializes KPIbuckets class.

        DF columns: see kpi_prediction_linker.resultCols
        """

        # Clean L1 and L2
        # remove s400_kpi bad users
        dfclean = self.removeUsers(df, bad_users)
        self.dfclean = dfclean.loc[~(dfclean['unique_cap_id'].isin(self.badGTs))]
        self.df_glare = None

        # KPI all: all frames.
        # This used to be unique frames only (as data collection was not reliable on all frames).
        # Change tfrecord_folder_name_kpi to TfRecords_unique_combined instead.
        self.dfkpiall = self.dfclean

        self.dftop10 = self.filterTopNUsers(self.dfkpiall, N=10)
        self.dfbotTen = self.filterBottomNUsers(self.dfkpiall, N=10)

        self.hppitch30 = self.filterHP_pitch_30(self.dfclean)
        self.hppitch60 = self.filterHP_pitch_60(self.dfclean)
        self.hppitch90 = self.filterHP_pitch_90(self.dfclean)
        self.hpyaw30 = self.filterHP_yaw_30(self.dfclean)
        self.hpyaw60 = self.filterHP_yaw_60(self.dfclean)
        self.hpyaw90 = self.filterHP_yaw_90(self.dfclean)

        self.hp15 = self.filterHP_15(self.dfclean)
        self.hp30 = self.filterHP_30(self.dfclean)
        self.hp45 = self.filterHP_45(self.dfclean)
        self.hp60 = self.filterHP_60(self.dfclean)

        self.coneCenter = self.filter_cone_45_central(self.dfclean)
        self.conePeriphery = self.filter_cone_45_periphery(self.dfclean)
        self.coneCenterTop10 = self.filter_cone_45_central(self.dftop10)
        self.conePeripheryTop10 = self.filter_cone_45_periphery(self.dftop10)

        # Global min-max (of ALL3)
        # Cam X max: 732 Cam X min: -2288
        # Cam Y max: 585 Cam Y min: -1201
        # self.max_cam_x = 700
        # self.min_cam_x = -2000
        # self.max_cam_y = 500
        # self.min_cam_y = -850

        # Decide min and max cam boundaries by min and max ground truth x, y values
        # rounded to nearest hundred.
        self.max_cam_x = int((max(df['GT_cam_x'].max(),
                                  700) // 100) * 100)  # round to nearest hundred
        self.min_cam_x = int((min(df['GT_cam_x'].min(), -2000) // 100) * 100)
        self.max_cam_y = int((max(df['GT_cam_y'].max(), 500) // 100) * 100)
        self.min_cam_y = int((min(df['GT_cam_y'].min(), -850) // 100) * 100)
        print('cam x: {} {}'.format(self.min_cam_x, self.max_cam_x))
        print('cam y: {} {}'.format(self.min_cam_y, self.max_cam_y))

        # Initialize offsets for each in-car regions in heat map
        self.camera_center_x = self.max_cam_x  # 700
        self.camera_center_y = -self.min_cam_y  # 850
        self.car_min_x = self.camera_center_x - 700  # 0
        self.car_max_x = self.camera_center_x + 1420  # 2120
        self.car_min_y = self.camera_center_y - 355  # 495
        self.car_max_y = self.camera_center_y + 315  # 1165
        self.right_mirror_x = self.camera_center_x + 1305
        self.right_mirror_y = self.camera_center_y - 80
        self.left_mirror_x = self.camera_center_x - 585
        self.left_mirror_y = self.camera_center_y - 80
        self.rear_mirror_x = self.camera_center_x + 370
        self.rear_mirror_y = self.camera_center_y - 275
        self.csd_x = self.camera_center_x + 370
        self.csd_y = self.camera_center_y + 115

        self.heat_map_columns = 18
        self.heat_map_rows = 9

        self.heat_map_grid_width = int((self.max_cam_x - self.min_cam_x) / self.heat_map_columns)
        self.heat_map_grid_height = int((self.max_cam_y - self.min_cam_y) / self.heat_map_rows)

    @staticmethod
    def bucketize(df, file_writer, message):
        """Prints and writes the measured error of users in each buckets."""

        '''"cam_err_x", "cam_err_y", "cam_err_z", "cam_err_xy",'''

        unique_frames = len(KPIbuckets.getUniqueFrames(df))
        num_frames = len(df.index)
        num_users = len(df.drop_duplicates(['user_id'], keep='first'))
        avg_cam_err = df['cam_error'].mean()
        stdd_cam_err = df['cam_error'].std()
        degree_err = df['degree_error'].mean()
        stdd_deg_err = df['degree_error'].std()
        x_err = df['cam_err_x'].mean()
        stdd_x_err = df['cam_err_x'].std()
        y_err = df['cam_err_y'].mean()
        stdd_y_err = df['cam_err_y'].std()
        z_err = df['cam_err_z'].mean()
        stdd_z_err = df['cam_err_z'].std()
        xy_err = df['cam_err_xy'].mean()
        stdd_xy_err = df['cam_err_xy'].std()

        print('Num. frames: {}'.format(num_frames))
        print('Unique frames: {}'.format(unique_frames))
        print('Num. users: {}'.format(num_users))
        print('Avg. Cam Error(cm): {}'.format(avg_cam_err))
        print('Stdd. Cam Error(cm): {}'.format(stdd_cam_err))
        print('Avg. Degree Error: {}'.format(degree_err))
        print('Stdd. Degree Error: {}'.format(stdd_deg_err))
        print('Avg. X Y Z Error: {} {} {}'.format(x_err, y_err, z_err))
        print('Stdd. X Y Z Error: {} {} {}'.format(stdd_x_err, stdd_y_err, stdd_z_err))
        print('Avg. XY Error: {}'.format(xy_err))
        print('Stdd. XY Error: {}'.format(stdd_xy_err))

        file_writer.write('{}\n'.format(message))
        file_writer.write('Num frames: {}\n'.format(num_frames))
        file_writer.write('Unique frames: {}\n'.format(unique_frames))
        file_writer.write('Num users: {}\n'.format(num_users))
        file_writer.write('Avg. Cam Error(cm): {}\n'.format(avg_cam_err))
        file_writer.write('Stdd. Cam Error(cm): {}\n'.format(stdd_cam_err))
        file_writer.write('Avg. Degree Error: {}\n'.format(degree_err))
        file_writer.write('Stdd. Degree Error: {}\n'.format(stdd_deg_err))
        file_writer.write('Avg. X Y Z Error: {} {} {}\n'.format(x_err, y_err, z_err))
        file_writer.write('Stdd. X Y Z Error: {} {} {}\n'.format(stdd_x_err, stdd_y_err,
                                                                 stdd_z_err))
        file_writer.write('Avg. XY Error: {}\n'.format(xy_err))
        file_writer.write('Stdd. XY Error: {}\n'.format(stdd_xy_err))
        file_writer.write('\n')

    @staticmethod
    def getUniqueFrames(df):
        """Returns data frame with unique frames."""

        old_sets = ['s400_KPI']
        old_set_filter = df["set_id"].isin(old_sets)
        df_old = df[old_set_filter]
        df_old = df_old.sort_values(by=['orig_frame_id', 'frame_num'])
        df_old = df_old.drop_duplicates(['orig_frame_id'], keep='first')
        df_new = df[~old_set_filter]
        df_new = df_new.sort_values(by=['orig_frame_id', 'frame_num'])
        df_new = df_new.drop_duplicates(['orig_frame_id'], keep='last')
        df2 = df_old.append(df_new)
        return df2

    @staticmethod
    def filterGetOnlyBackwardIntersection(df):
        """Returns data frame with only backward intersections."""

        return df.loc[df['backwardIntersection'] == 'Yes']

    @staticmethod
    def filterGetOnlyForwardIntersection(df):
        """Returns data frame with only forward intersections."""

        return df.loc[df['backwardIntersection'] == 'No']

    @staticmethod
    def filterNoGlasses(df):
        """Returns data frame with users not wearing glasses."""

        return df.loc[df['glasses'] == 'No']

    @staticmethod
    def filterGlasses(df):
        """Returns data frame with users wearing glasses."""

        return df.loc[df['glasses'] == 'Yes']

    @staticmethod
    def filterOccluded(df):
        """Returns data frame with users having either left or right eye occluded."""

        return df[(df['leye_status'] == 'occluded') | (df['reye_status'] == 'occluded')]

    @staticmethod
    def filterCarRegions(df, region='car', margin=0):
        """Returns data frame with specified car region."""

        min_x, min_y, max_x, max_y = 0, 0, 0, 0
        if region == 'car':
            min_x = KPIbuckets.car_region_min_x
            min_y = KPIbuckets.car_region_min_y
            max_x = KPIbuckets.car_region_max_x
            max_y = KPIbuckets.car_region_max_y
        elif region == 'center':
            min_x = KPIbuckets.car_center_region_min_x
            min_y = KPIbuckets.car_center_region_min_y
            max_x = KPIbuckets.car_center_region_max_x
            max_y = KPIbuckets.car_center_region_max_y
        elif region == 'right':
            min_x = KPIbuckets.car_right_region_min_x
            min_y = KPIbuckets.car_right_region_min_y
            max_x = KPIbuckets.car_right_region_max_x
            max_y = KPIbuckets.car_right_region_max_y

        return df[(df['GT_cam_x'] > (min_x - margin)) & (df['GT_cam_x'] < (max_x + margin))
                  & (df['GT_cam_y'] < (max_y + margin)) & (df['GT_cam_y'] > (min_y - margin))]

    @staticmethod
    def filterRegionNames(df, region_names=None):
        """Splits the dataframe into its regions, keeping only region_names.

        If region_names is None, keeps all.

        Returns a dict(region_name->df)
        """
        if region_names is None:
            region_names = df['region_name'].unique()
        regions = {}
        for region_name in region_names:
            regions[region_name] = df.loc[df['region_name'] == region_name]
        return regions

    @staticmethod
    def filterRegionLabels(df, region_labels=None):
        """Splits the dataframe into its regions by label, keeping only region_labels.

        If region_labels is None, keeps all.

        Returns a dict(region_label->df)
        """
        if region_labels is None:
            region_labels = df['region_label'].unique()
        regions = {}
        for region_label in region_labels:
            regions[region_label] = df.loc[df['region_label'] == region_label]
        return regions

    @staticmethod
    def filterCameraIds(df, camera_ids=None):
        """Splits the dataframe into cam ids.

        If camera_ids is None, keeps all.

        Returns a dict(cameras->df)
        """
        df_camera_id = df.set_id.map(kpi_prediction_linker._get_camera_id)
        if camera_ids is None:
            camera_ids = df_camera_id.unique()
        cameras = {}
        for camera_id in camera_ids:
            cameras[camera_id] = df.loc[df_camera_id == camera_id]
        return cameras

    @staticmethod
    def filterCenterCamera(df):
        """Return only frames from the center camera."""
        return df[(df['cam_position'] == 'Center')
                  | (df['cam_position'] == 'Middle Center')
                  | (df['cam_position'] == 'Center Middle ')
                  | (df['cam_position'] == 'Center Middle')]

    @staticmethod
    def filterLeftCenterCamera(df):
        """Return only frames from the left center camera."""
        return df[(df['cam_position'] == 'Left')
                  | (df['cam_position'] == 'Left Center')
                  | (df['cam_position'] == 'Center Left')]

    @staticmethod
    def filterBottomCenterCamera(df):
        """Return only frames from the bottom center camera."""
        return df[(df['cam_position'] == 'Bottom')
                  | (df['cam_position'] == 'Bottom Center')]

    @staticmethod
    def filterTopCenterCamera(df):
        """Return only frames from the top center camera."""
        return df[(df['cam_position'] == 'Top')
                  | (df['cam_position'] == 'Top Center')
                  | (df['cam_position'] == 'Top Centrr')]

    @staticmethod
    def filterRightCenterCamera(df):
        """Return only frames from the right center camera."""
        return df[(df['cam_position'] == 'Right Center')
                  | (df['cam_position'] == 'Center Right')
                  | (df['cam_position'] == 'Right Middle')]

    @staticmethod
    def filterRightCamera(df):
        """Return only frames from the right camera."""
        return df[(df['cam_position'] == 'Far Right Center')
                  | (df['cam_position'] == 'Bottom Right')
                  | (df['cam_position'] == 'Left Right')
                  | (df['cam_position'] == 'Right Right')]

    @staticmethod
    def filterTopNUsers(df, N=10, metric='degree'):
        """Returns top N users with minimal errors."""

        if metric == 'degree':
            dftopusers = df.groupby('user_id')['degree_error'].mean().sort_values(). \
                nsmallest(N)
        elif metric == 'cm':
            dftopusers = df.groupby('user_id')['cam_error'].mean().sort_values(). \
                nsmallest(N)
        topusers = []
        for index, _ in dftopusers.items():
            topusers.append(index)
        return df.loc[df['user_id'].isin(topusers)]

    @staticmethod
    def filterBottomNUsers(df, N=10):
        """Returns bottom N users with maximal errors."""

        dfbotusers = df.groupby('user_id')['degree_error'].mean().sort_values(). \
            nlargest(N)
        botusers = []
        for index, _ in dfbotusers.items():
            botusers.append(index)
        return df.loc[df['user_id'].isin(botusers)]

    @staticmethod
    def removeUsers(df, usernames):
        """Returns a data frame with a specified user removed."""

        if usernames is not None and usernames != '':
            return df[~df['user_id'].isin(usernames)]

        return df

    @staticmethod
    def filterEitherOpenEyes(df):
        """Returns a data frame with users with at least one eye open."""

        return df[(df['leye_status'] == 'open') | (df['reye_status'] == 'open')]

    @staticmethod
    def filterBothOpenEyes(df):
        """Returns a data frame with users with both of their eyes open."""

        return df[(df['leye_status'] == 'open') & (df['reye_status'] == 'open')]

    @staticmethod
    def classifyGlare(df):
        """Classify each frame's user glare status."""

        for index, _ in df.iterrows():
            left_pupil_glare_status = df.at[index, 'left_pupil_glare_status']
            left_eye_glare_status = df.at[index, 'left_eye_glare_status']
            right_pupil_glare_status = df.at[index, 'right_pupil_glare_status']
            right_eye_glare_status = df.at[index, 'right_eye_glare_status']
            left_eye_occlusion = df.at[index, 'leye_status']
            right_eye_occlusion = df.at[index, 'reye_status']
            left_iris_status = df.at[index, 'left_iris_status']
            right_iris_status = df.at[index, 'right_iris_status']

            '''
            Glare Ranks
            1. glare on pupil
            2. glare on eye (but not pupil)
            3. glint on pupil(but no glare on eye or pupil)
            4. glint on eye but not pupil (no glare on eye or pupil)
            5. no glare or glint on eye
            '''

            left_eye_rank = 0
            right_eye_rank = 0
            bright_pupil = 'normal'
            if left_eye_occlusion == 'open':
                if left_pupil_glare_status == 'glare_pupil':
                    left_eye_rank = 1
                if left_eye_glare_status == 'glare_eye' and left_pupil_glare_status == 'no_glare':
                    left_eye_rank = 2
                if left_pupil_glare_status == 'glint_pupil' and left_eye_glare_status == 'no_glare':
                    left_eye_rank = 3
                if left_eye_glare_status == 'glint_eye' and left_pupil_glare_status == 'no_glare':
                    left_eye_rank = 4
                if left_eye_glare_status == 'no_glare' and left_pupil_glare_status == 'no_glare':
                    left_eye_rank = 5

                if left_iris_status == 'bright_pupil':
                    bright_pupil = 'bright_pupil'

            if right_eye_occlusion == 'open':
                if right_pupil_glare_status == 'glare_pupil':
                    right_eye_rank = 1
                if right_eye_glare_status == 'glare_eye' and right_pupil_glare_status == 'no_glare':
                    right_eye_rank = 2
                if right_pupil_glare_status == 'glint_pupil' and \
                        right_eye_glare_status == 'no_glare':
                    right_eye_rank = 3
                if right_eye_glare_status == 'glint_eye' and right_pupil_glare_status == 'no_glare':
                    right_eye_rank = 4
                if right_eye_glare_status == 'no_glare' and right_pupil_glare_status == 'no_glare':
                    right_eye_rank = 5

                if right_iris_status == 'bright_pupil':
                    bright_pupil = 'bright_pupil'

            df.at[index, 'glare_status'] = min(left_eye_rank, right_eye_rank)
            df.at[index, 'bright_pupil'] = bright_pupil

        return df

    @staticmethod
    def filter_cone_45_central(df):
        """Returns a data frame with samples within 45-deg cone of user."""
        return df[df['cone_degree'] <= 22.5]

    @staticmethod
    def filter_cone_45_periphery(df):
        """Returns a data frame with samples outside 45-deg cone of user."""
        return df[df['cone_degree'] > 22.5]

    @staticmethod
    def filterHP_pitch_30(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[(((0 < df['GT_hp_theta']) & (df['GT_hp_theta'] <= np.pi / 6)) |
                   ((0 > df['GT_hp_theta']) & (df['GT_hp_theta'] >= -np.pi / 6)))]

    @staticmethod
    def filterHP_pitch_60(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[(((np.pi / 6 < df['GT_hp_theta']) & (df['GT_hp_theta'] <= np.pi / 3)) |
                   ((-np.pi / 6 > df['GT_hp_theta']) & (df['GT_hp_theta'] >= -np.pi / 3)))]

    @staticmethod
    def filterHP_pitch_90(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[((df['GT_hp_theta'] > np.pi / 3) | (df['GT_hp_theta'] < -np.pi / 3))]

    @staticmethod
    def filterHP_yaw_30(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[(((0 < df['GT_hp_phi']) & (df['GT_hp_phi'] <= np.pi / 6)) |
                   ((0 > df['GT_hp_phi']) & (df['GT_hp_phi'] >= -np.pi / 6)))]

    @staticmethod
    def filterHP_yaw_60(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[(((np.pi / 6 < df['GT_hp_phi']) & (df['GT_hp_phi'] <= np.pi / 3)) |
                   ((-np.pi / 6 > df['GT_hp_phi']) & (df['GT_hp_phi'] >= -np.pi / 3)))]

    @staticmethod
    def filterHP_yaw_90(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[((df['GT_hp_phi'] > np.pi / 3) | (df['GT_hp_phi'] < -np.pi / 3))]

    @staticmethod
    def filterHP_15(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[((df['GT_hp_phi'] < np.pi / 12) & (df['GT_hp_phi'] > -np.pi / 12) &
                   (df['GT_hp_theta'] < np.pi / 12) & (df['GT_hp_theta'] > -np.pi / 12))]

    @staticmethod
    def filterHP_30(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[((df['GT_hp_phi'] < np.pi / 6) & (df['GT_hp_phi'] > -np.pi / 6) &
                   (df['GT_hp_theta'] < np.pi / 6) & (df['GT_hp_theta'] > -np.pi / 6))]

    @staticmethod
    def filterHP_45(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[((df['GT_hp_phi'] < np.pi / 4) & (df['GT_hp_phi'] > -np.pi / 4) &
                   (df['GT_hp_theta'] < np.pi / 4) & (df['GT_hp_theta'] > -np.pi / 4))]

    @staticmethod
    def filterHP_60(df):
        """Returns a data frame with users with hp theta smaller than 30 degrees."""

        return df[((df['GT_hp_phi'] < np.pi / 3) & (df['GT_hp_phi'] > -np.pi / 3) &
                   (df['GT_hp_theta'] < np.pi / 3) & (df['GT_hp_theta'] > -np.pi / 3))]

    def heat_map_visualization(self, df, output_path, file_name):
        """
        Produces an error heat map image.

        The heatmap produced shows the ground truth's xy-coordinates (in camera coordinate system).
        Drawing regions only makes sense with a single camera dataframe and is therefore disabled
        on multiple cameras.
        """

        def color_scale_list(value):
            """Return the (b,g,r) color from the custom color scale.

            For the given value between 0 and 1.
            """
            if value < 1.0/16:
                return (0, 255, 0)  # green -> 0 - 1.5 deg, 0 - 2 cm
            if value < 2.0/16:
                return (0, 150, 128)  # dark-green -> 1.5 -3 deg , 2 - 4 cm
            if value < 4.0/16:
                return (6, 158, 205)  # yellow-green -> 3 - 6, 4 - 8 cm
            if value < 8.0/16:
                return (0, 178, 255)  # orange -> 6 - 12, 8 - 16 cm
            if value < 16.0/16:
                return (66, 112, 243)  # salmon -> 12 - 24, 16 - 32 cm

            return (30, 7, 205)  # red > 24

        ED = 1.5  # min error in degree
        DIST = 75  # cm
        ECM = DIST * np.tan(ED * np.pi / 180)  # min error in cm: 1.9640 cm

        img_xyz_cm = np.zeros((self.max_cam_y - self.min_cam_y,
                               self.max_cam_x - self.min_cam_x, 3), np.uint8)
        img_xyz_deg = np.zeros((self.max_cam_y - self.min_cam_y,
                                self.max_cam_x - self.min_cam_x, 3), np.uint8)

        # dict that keeps track of error in each region of heat map
        heat_map_error_cm_dict = {}
        heat_map_error_degree_dict = {}

        # Grids x has columns+1 entries to allow getting the end of the last column
        # with heat_map_grids_x[self.heat_map_columns].
        heat_map_grids_x = range(0, (self.max_cam_x - self.min_cam_x) + self.heat_map_grid_width,
                                 self.heat_map_grid_width)
        heat_map_grids_y = range(0, (self.max_cam_y - self.min_cam_y) + self.heat_map_grid_height,
                                 self.heat_map_grid_height)

        for _, row in df.iterrows():
            gt_cam_x = row['GT_cam_x']
            gt_cam_y = row['GT_cam_y']
            gaze_error_mm = row['cam_error']
            gaze_error_degree = row['degree_error']

            heat_map_x_index = self.heat_map_columns-1 - min(int(gt_cam_x - self.min_cam_x) //
                                                             self.heat_map_grid_width,
                                                             self.heat_map_columns-1)
            heat_map_y_index = min(int(gt_cam_y - self.min_cam_y) // self.heat_map_grid_height,
                                   self.heat_map_rows-1)

            if (heat_map_y_index, heat_map_x_index) in heat_map_error_cm_dict:
                heat_map_error_cm_dict[(heat_map_y_index, heat_map_x_index)].append(gaze_error_mm)
            else:
                heat_map_error_cm_dict[(heat_map_y_index, heat_map_x_index)] = [gaze_error_mm]
            if (heat_map_y_index, heat_map_x_index) in heat_map_error_degree_dict:
                heat_map_error_degree_dict[(heat_map_y_index,
                                            heat_map_x_index)].append(gaze_error_degree)
            else:
                heat_map_error_degree_dict[(heat_map_y_index,
                                            heat_map_x_index)] = [gaze_error_degree]

        for grid_y in range(self.heat_map_rows):
            for grid_x in range(self.heat_map_columns):
                if (grid_y, grid_x) in heat_map_error_cm_dict:
                    count = len(heat_map_error_cm_dict[(grid_y, grid_x)])
                    cam_xyz_mean_cm = np.mean(heat_map_error_cm_dict[(grid_y, grid_x)])
                    cam_degree_mean = np.mean(heat_map_error_degree_dict[(grid_y, grid_x)])
                else:
                    continue

                if cam_xyz_mean_cm < 0:
                    continue
                xyz_color = color_scale_list(cam_xyz_mean_cm / (16 * ECM))

                if cam_degree_mean < 0:
                    continue
                degree_color = color_scale_list(cam_degree_mean / (16 * ED))

                # Plot xyz cm error heat map regions.
                cv2.rectangle(img_xyz_cm, (int(heat_map_grids_x[grid_x]),
                                           int(heat_map_grids_y[grid_y])),
                              (int(heat_map_grids_x[grid_x + 1]),
                               int(heat_map_grids_y[grid_y + 1])),
                              xyz_color, thickness=-1)
                cv2.rectangle(img_xyz_cm, (int(heat_map_grids_x[grid_x]),
                                           int(heat_map_grids_y[grid_y])),
                              (int(heat_map_grids_x[grid_x + 1]),
                               int(heat_map_grids_y[grid_y + 1])),
                              (255, 0, 0), thickness=1)
                cv2.putText(img_xyz_cm, 'Frames: {}'.format(str(count)),
                            (int(heat_map_grids_x[grid_x] + 5),
                             int(heat_map_grids_y[grid_y] + 2 * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
                cv2.putText(img_xyz_cm, 'Err[cm]: {}'.format(round(cam_xyz_mean_cm, 2)),
                            (int(heat_map_grids_x[grid_x] + 5),
                             int(heat_map_grids_y[grid_y] + 3 * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
                cv2.putText(img_xyz_cm, 'Err[deg]: {}'.format(round(cam_degree_mean, 2)),
                            (int(heat_map_grids_x[grid_x] + 5),
                             int(heat_map_grids_y[grid_y] + 4 * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

                # Plot degrees error heat map regions.
                cv2.rectangle(img_xyz_deg, (int(heat_map_grids_x[grid_x]),
                                            int(heat_map_grids_y[grid_y])),
                              (int(heat_map_grids_x[grid_x + 1]),
                               int(heat_map_grids_y[grid_y + 1])),
                              degree_color, thickness=-1)
                cv2.rectangle(img_xyz_deg, (int(heat_map_grids_x[grid_x]),
                                            int(heat_map_grids_y[grid_y])),
                              (int(heat_map_grids_x[grid_x + 1]),
                               int(heat_map_grids_y[grid_y + 1])),
                              (255, 0, 0), thickness=1)
                cv2.putText(img_xyz_deg, 'Frames: {}'.format(str(count)),
                            (int(heat_map_grids_x[grid_x] + 5),
                             int(heat_map_grids_y[grid_y] + 2 * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
                cv2.putText(img_xyz_deg, 'Err[cm]: {}'.format(round(cam_xyz_mean_cm, 2)),
                            (int(heat_map_grids_x[grid_x] + 5),
                             int(heat_map_grids_y[grid_y] + 3 * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
                cv2.putText(img_xyz_deg, 'Err[deg]: {}'.format(round(cam_degree_mean, 2)),
                            (int(heat_map_grids_x[grid_x] + 5),
                             int(heat_map_grids_y[grid_y] + 4 * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

        # Draw center camera region.
        cv2.circle(img_xyz_cm, (int(self.camera_center_x), int(self.camera_center_y)),
                   int(10), (255, 255, 255), -1)
        cv2.circle(img_xyz_deg, (int(self.camera_center_x), int(self.camera_center_y)),
                   int(10), (255, 255, 255), -1)
        cv2.rectangle(img_xyz_cm, (int(self.camera_center_x - 200),
                                   int(self.camera_center_y - 85)),
                      (int(self.camera_center_x + 200), int(self.camera_center_y + 85)),
                      (255, 255, 255), thickness=2)
        cv2.rectangle(img_xyz_deg, (int(self.camera_center_x - 200),
                                    int(self.camera_center_y - 85)),
                      (int(self.camera_center_x + 200), int(self.camera_center_y + 85)),
                      (255, 255, 255), thickness=2)
        cv2.putText(img_xyz_cm, 'Cam', (int(self.camera_center_x - 15),
                                        int(self.camera_center_y - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255))
        cv2.putText(img_xyz_deg, 'Cam', (int(self.camera_center_x - 15),
                                         int(self.camera_center_y - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255))
        cv2.putText(img_xyz_cm, '(0,0)', (int(self.camera_center_x - 35),
                                          int(self.camera_center_y + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255))
        cv2.putText(img_xyz_deg, '(0,0)', (int(self.camera_center_x - 35),
                                           int(self.camera_center_y + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255))

        camera_ids = df.set_id.map(kpi_prediction_linker._get_camera_id).unique()
        if len(camera_ids) == 1 and camera_ids[0] != 'unknown':
            # a heatmap with only a single camera, enable boxes showing the regions

            # Draw entire car region.
            cv2.rectangle(img_xyz_cm, (int(self.car_min_x), int(self.car_min_y)),
                          (int(self.car_max_x), int(self.car_max_y)),
                          (255, 255, 255), thickness=3)
            cv2.rectangle(img_xyz_deg, (int(self.car_min_x), int(self.car_min_y)),
                          (int(self.car_max_x), int(self.car_max_y)),
                          (255, 255, 255), thickness=3)

            # Draw a box around each region.
            df_regions = self.filterRegionNames(df)
            for region_name, df_region in df_regions.items():
                if region_name == "":
                    region_name = "screen"  # No regions -> on bench setup.
                # x axis is flipped (compare to heat_map_x_index), max and min hence swapped.
                min_x = self.max_cam_x - df_region['GT_cam_x'].max()
                max_x = self.max_cam_x - df_region['GT_cam_x'].min()
                min_y = df_region['GT_cam_y'].min() - self.min_cam_y
                max_y = df_region['GT_cam_y'].max() - self.min_cam_y
                center_x = min_x + (max_x-min_x)/2
                center_y = min_y + (max_y-min_y)/2

                cv2.circle(img_xyz_cm, (int(center_x), int(center_y)),
                           int(10), (255, 255, 255), -1)
                cv2.circle(img_xyz_deg, (int(center_x), int(center_y)),
                           int(10), (255, 255, 255), -1)
                cv2.rectangle(img_xyz_cm, (int(min_x), int(min_y)),
                              (int(max_x), int(max_y)),
                              (255, 255, 255), thickness=2)
                cv2.rectangle(img_xyz_deg, (int(min_x), int(min_y)),
                              (int(max_x), int(max_y)),
                              (255, 255, 255), thickness=2)
                cv2.putText(img_xyz_cm, region_name, (int(center_x - 15),
                                                      int(center_y - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255))
                cv2.putText(img_xyz_deg, region_name, (int(center_x - 15),
                                                       int(center_y - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255))

        # Remember opencv uses BGR colors, not RGB.
        cv2.imwrite(os.path.join(output_path, file_name + '_cm.png'), img_xyz_cm)
        cv2.imwrite(os.path.join(output_path, file_name + '_degree.png'), img_xyz_deg)

    @staticmethod
    def save_images(df, output_path, bucket, num_samples):
        """Outputs sample images from data frame for debug purposes."""

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        bucket_folder = os.path.join(output_path, bucket)
        shutil.rmtree(bucket_folder, ignore_errors=True)
        os.mkdir(bucket_folder)

        num_samples = min(len(df.index), num_samples)
        df = df.sample(num_samples)

        for index, _ in df.iterrows():
            cosmos_path = '/home/copilot.cosmos10/RealTimePipeline/set'
            data_path = 'Data'
            set_id = df.at[index, 'set_id']
            user_id = df.at[index, 'user_id']
            frame_id = df.at[index, 'image_name']

            if set_id >= 's500' and 'germany' not in set_id:
                cosmos_path = '/home/driveix.cosmos639/GazeData/orgData'
                data_path = 'pngData'

            frame_path = os.path.join(cosmos_path, set_id, data_path, user_id, frame_id)
            image_frame = cv2.imread(frame_path)
            output_file = '{}_{}_{}'.format(set_id, user_id, frame_id)
            cv2.imwrite(os.path.join(bucket_folder, output_file), image_frame)


class PredictionVisualizer(object):
    """Visualizes prediction through combining KPIBuckets and kpi_prediction_linker."""

    time_ins_group_cols = ['root_set_id', 'user_id', 'screen_pix_x', 'screen_pix_y', 'frame_num']
    combined_eval_cols = ['num_cameras', 'min_cam_cm_err', 'min_cam_deg_err',
                          'min_GT_cam_norm', 'min_GT_angle_norm']

    def __init__(self, model_type, kpi_sets, bad_users,
                 buckets_to_visualize, path_info,
                 gaze_origin='normal', tp_degrees=False, time_instance_info_path=None):
        """
        Initializes prediction visualizer.

        model_type: xyz, joint, theta-phi
        kpi_sets: kpi sets to visualize
        bad_users: users to be filtered from predictions
        buckets_to_visualize: visualize specified user buckets: glasses, no glasses, occluded eye
        time_instance_info_path: path to dump time instance info
        """
        self._model_type = model_type
        self._bad_users = None
        if bad_users is not None and bad_users != '':
            self._bad_users = bad_users.split(' ')
        self._visualize_functions = buckets_to_visualize.split(' ')

        self.time_instance_info_path = time_instance_info_path
        write_ins = time_instance_info_path is not None
        if write_ins:
            mkdir_p(time_instance_info_path)

        if self._model_type == 'joint':
            self._kpi_prediction_linker = [kpi_prediction_linker(model_type='joint_xyz',
                                                                 kpi_sets=kpi_sets,
                                                                 path_info=path_info,
                                                                 gaze_origin=gaze_origin,
                                                                 theta_phi_degrees=tp_degrees),
                                           kpi_prediction_linker(model_type='joint_tp',
                                                                 kpi_sets=kpi_sets,
                                                                 path_info=path_info,
                                                                 gaze_origin=gaze_origin,
                                                                 theta_phi_degrees=tp_degrees),
                                           kpi_prediction_linker(model_type='joint',
                                                                 kpi_sets=kpi_sets,
                                                                 path_info=path_info,
                                                                 gaze_origin=gaze_origin,
                                                                 theta_phi_degrees=tp_degrees,
                                                                 write_time_instance_info=write_ins)
                                           ]
            self._output_file = ['kpi_bucketize_joint_xyz.txt',
                                 'kpi_bucketize_joint_tp.txt',
                                 'kpi_bucketize_joint.txt']
            self._heat_map_folder = ['heat_map_joint_xyz',
                                     'heat_map_joint_tp',
                                     'heat_map_joint']
            self._joint_extension = ['joint_xyz',
                                     'joint_tp',
                                     'joint']
        else:
            self._kpi_prediction_linker = kpi_prediction_linker(model_type=self._model_type,
                                                                kpi_sets=kpi_sets,
                                                                path_info=path_info,
                                                                gaze_origin=gaze_origin,
                                                                theta_phi_degrees=tp_degrees,
                                                                write_time_instance_info=write_ins)
            self._output_file = 'kpi_bucketize_' + self._model_type + '.txt'
            self._heat_map_folder = 'heat_map_' + self._model_type

    def __call__(self, predictions, kpi_dataframe, output_path):
        """
        Produces bucketized results for predictions and error heat map.

        If model type is joint, bucketized results and heat map will be generated
        for joint_xyz, joint_tp and joint model.
        """

        if kpi_dataframe is None or kpi_dataframe.empty:
            print('Got an empty kpi_dataframe, cannot compute all buckets.')
            # Remove unavailable buckets and warn about them.
            config = set(self._visualize_functions)
            new = config - set(['hard', 'easy', 'center-camera', 'left-center-camera',
                                'bottom-center-camera', 'top-center-camera',
                                'right-center-camera', 'right-camera', 'glasses', 'no-glasses',
                                'occluded', 'glare-pupil', 'glare-eye', 'glint-pupil',
                                'glint-eye', 'no-glare', 'class-0', 'bright-pupil',
                                'non-bright-pupil'])
            diff = config - new
            print('Skipping the following buckets: {}'.format(list(diff)))
            self._visualize_functions = [e for e in self._visualize_functions if e in new]

        if self._model_type == 'joint':
            for linker, output_file, heat_map_folder, \
                joint_extension in zip(self._kpi_prediction_linker,
                                       self._output_file,
                                       self._heat_map_folder,
                                       self._joint_extension):
                self.visualize_kpi(linker, predictions, kpi_dataframe, output_path,
                                   output_file, heat_map_folder, joint_extension)
        else:
            self.visualize_kpi(self._kpi_prediction_linker, predictions, kpi_dataframe,
                               output_path, self._output_file, self._heat_map_folder)

    @staticmethod
    def _add_min_norm(group):
        group['min_cam_cm_err'] = group['cam_error'].min()
        group['min_cam_deg_err'] = group['degree_error'].min()
        group['min_GT_cam_norm'] = group.loc[group['cam_error'].idxmin()]['GT_cam_norm']
        group['min_GT_angle_norm'] = group.loc[group['degree_error'].idxmin()]['GT_angle_norm']
        return group

    def _dump_time_ins_info(self, df_total):
        df_total_no_dup = df_total.drop_duplicates(
            self.time_ins_group_cols + ['set_id'],
            keep='first',
            inplace=False)
        time_ins_group = df_total_no_dup.groupby(
            self.time_ins_group_cols)

        print('There are {} time instances.'.format(time_ins_group.ngroups))

        # Populate combined_eval_cols fields
        df_total_no_dup['num_cameras'] = time_ins_group['frame_num'].transform('count')
        df_total_no_dup = time_ins_group.apply(self._add_min_norm)

        # Flush <time_instance_path>/<root_set_id>
        df_group_root = df_total_no_dup.loc[:, ['root_set_id']]
        for root_set_id, _ in df_group_root.groupby(['root_set_id']):
            root_set_id_path = os.path.join(self.time_instance_info_path, root_set_id)
            if os.path.exists(root_set_id_path):
                print('Deleting files of folder {}'.format(root_set_id_path))
                subprocess.call('rm -rf ' + root_set_id_path + '/*', shell=True)

        df_group_stat = df_total_no_dup.loc[:, self.time_ins_group_cols + self.combined_eval_cols]
        for k, group in df_group_stat.groupby(self.time_ins_group_cols):
            # Every row in the same group has the same vals for combined_eval_cols
            df_group = group.head(1).loc[:, self.combined_eval_cols]
            df_group['time_instance'] = '_'.join(map(str, k))
            df_group = df_group[['time_instance'] + self.combined_eval_cols]

            root_set_id_path = os.path.join(self.time_instance_info_path, k[0])
            mkdir_p(root_set_id_path)
            user = k[1]
            df_group.to_csv(
                os.path.join(root_set_id_path, '{}.txt'.format(user)),
                header=None, index=None, sep=' ', mode='a')

    def visualize_kpi(self, linker, predictions, kpi_dataframe, output_path,
                      output_file, heat_map_folder, joint_extension='joint'):
        """Generate combined Data frame."""

        output_file = os.path.join(output_path, output_file)
        sample_image_dir = os.path.join(output_path, 'sample_images')

        df_total = linker.getDataFrame(predictions, kpi_dataframe)

        # Write csv of the total table.
        df_total.to_csv(os.path.join(output_path, 'df_total_{}.csv'.format(joint_extension)))

        # Non-joint model types have default joint_extension='joint'
        if joint_extension == 'joint' and self.time_instance_info_path:
            self._dump_time_ins_info(df_total)

        # Create buckets / User analysis using DF above
        tx = KPIbuckets(df_total, self._bad_users)

        heat_map_dir = os.path.join(output_path, heat_map_folder)
        if not os.path.exists(heat_map_dir):
            os.mkdir(heat_map_dir)

        with open(output_file, 'w') as f:
            for fn in self._visualize_functions:
                if fn == 'all':
                    print('\nKPI all frames')
                    tx.bucketize(df_total, f, 'KPI all frames')
                    tx.heat_map_visualization(df_total, heat_map_dir, 'kpi_all')

                elif fn == 'only-forward':
                    print('\nKPI only forward intersections')
                    df_forward = tx.filterGetOnlyForwardIntersection(tx.dfkpiall)
                    tx.bucketize(df_forward, f, 'KPI only forward intersections')
                    tx.heat_map_visualization(df_forward, heat_map_dir, 'kpi_forward')

                elif fn == 'only-backward':
                    print('\nKPI only backward intersections')
                    df_backward = tx.filterGetOnlyBackwardIntersection(tx.dfkpiall)
                    tx.bucketize(df_backward, f, 'KPI only backward intersections')
                    tx.heat_map_visualization(df_backward, heat_map_dir, 'kpi_backward')

                elif fn == 'unique':
                    print('\nKPI unique frames')
                    df_unique = tx.getUniqueFrames(tx.dfkpiall)
                    tx.bucketize(df_unique, f, 'KPI unique frames')
                    tx.heat_map_visualization(df_unique, heat_map_dir, 'kpi_unique')

                elif fn == 'clean':
                    print('\nKPI clean')
                    tx.bucketize(tx.dfclean, f, 'KPI clean')
                    tx.heat_map_visualization(tx.dfclean, heat_map_dir, 'kpi_clean')

                elif fn == 'hard':
                    print('\nKPI hard')
                    # KPI hard: preseved frames where user may of one of their eye closed.
                    dfkpihard = tx.filterEitherOpenEyes(tx.dfkpiall)
                    tx.bucketize(dfkpihard, f, 'KPI hard')
                    tx.heat_map_visualization(dfkpihard, heat_map_dir, 'kpi_hard')

                elif fn == 'easy':
                    print('\nKPI easy')
                    # KPI easy: preserved frames of user having both of their eyes present.
                    dfkpieasy = tx.filterBothOpenEyes(tx.dfkpiall)
                    tx.bucketize(dfkpieasy, f, 'KPI easy')
                    tx.heat_map_visualization(dfkpieasy, heat_map_dir, 'kpi_easy')

                elif fn == 'car':
                    print('\nKPI car region')
                    df_car = tx.filterCarRegions(tx.dfkpiall, region='car')
                    tx.bucketize(df_car, f, 'KPI car region')
                    tx.heat_map_visualization(df_car, heat_map_dir, 'kpi_car_region')

                elif fn == 'car_center':
                    print('\nKPI car region center')
                    df_car_center = tx.filterCarRegions(tx.dfkpiall, region='center')
                    tx.bucketize(df_car_center, f, 'KPI car center region')
                    tx.heat_map_visualization(df_car_center, heat_map_dir, 'kpi_car_center')

                elif fn == 'car_right':
                    print('\nKPI car region right')
                    df_car_right = tx.filterCarRegions(tx.dfkpiall, region='right')
                    tx.bucketize(df_car_right, f, 'KPI car right region')
                    tx.heat_map_visualization(df_car_right, heat_map_dir, 'kpi_car_right')

                elif fn == 'regions':
                    print('\nKPI different regions...')
                    df_regions = tx.filterRegionNames(tx.dfkpiall)
                    if len(df_regions) == 1 and '' in df_regions:
                        print('No regions found, skipping regions bucket!')
                    else:
                        for region_name, df_region in df_regions.items():
                            print('\nKPI {}'.format(region_name))
                            tx.bucketize(df_region, f, 'KPI {}'.format(region_name))
                            tx.heat_map_visualization(df_region, heat_map_dir,
                                                      'kpi_{}'.format(region_name))

                elif fn == 'region-labels':
                    print('\nKPI different region-labels...')
                    df_regions = tx.filterRegionLabels(tx.dfkpiall)
                    if len(df_regions) == 1 and 'unknown' in df_regions:
                        print('No regions found, skipping region-labels bucket!')
                    else:
                        for region_label, df_region in df_regions.items():
                            print('\nKPI {}'.format(region_label))
                            tx.bucketize(df_region, f, 'KPI {}'.format(region_label))
                            sane_label = "_".join(region_label.split())
                            tx.heat_map_visualization(df_region, heat_map_dir,
                                                      'kpi_{}'.format(sane_label))

                elif fn == 'cameras':
                    print('\nKPI different cameras...')
                    df_cameras = tx.filterCameraIds(tx.dfkpiall)
                    for camera_id, df_camera in df_cameras.items():
                        print('\nKPI camera {}'.format(camera_id))
                        tx.bucketize(df_camera, f, 'KPI camera {}'.format(camera_id))
                        tx.heat_map_visualization(df_camera, heat_map_dir,
                                                  'kpi_camera_{}'.format(camera_id))

                elif fn == 'center-camera':
                    print('\nKPI center camera')
                    df_center = tx.filterCenterCamera(tx.dfkpiall)
                    tx.bucketize(df_center, f, 'KPI center camera')
                    tx.heat_map_visualization(df_center, heat_map_dir, 'kpi_center_camera')

                elif fn == 'left-center-camera':
                    print('\nKPI left center camera')
                    df_left_center_camera = tx.filterLeftCenterCamera(tx.dfkpiall)
                    tx.bucketize(df_left_center_camera, f, 'KPI left center camera')
                    tx.heat_map_visualization(df_left_center_camera, heat_map_dir,
                                              'kpi_left_center_camera')

                elif fn == 'bottom-center-camera':
                    print('\nKPI bottom center camera')
                    df_bot_center_camera = tx.filterBottomCenterCamera(tx.dfkpiall)
                    tx.bucketize(df_bot_center_camera, f, 'KPI bottom center camera')
                    tx.heat_map_visualization(df_bot_center_camera, heat_map_dir,
                                              'kpi_bot_center_camera')

                elif fn == 'top-center-camera':
                    print('\nKPI top center camera')
                    df_top_center_camera = tx.filterTopCenterCamera(tx.dfkpiall)
                    tx.bucketize(df_top_center_camera, f, 'KPI top center camera')
                    tx.heat_map_visualization(df_top_center_camera, heat_map_dir,
                                              'kpi_top_center_camera')

                elif fn == 'right-center-camera':
                    print('\nKPI right center camera')
                    df_right_center_camera = tx.filterRightCenterCamera(tx.dfkpiall)
                    tx.bucketize(df_right_center_camera, f, 'KPI right center camera')
                    tx.heat_map_visualization(df_right_center_camera, heat_map_dir,
                                              'kpi_right_center_camera')

                elif fn == 'right-camera':
                    print('\nKPI right camera')
                    df_right_camera = tx.filterRightCamera(tx.dfkpiall)
                    tx.bucketize(df_right_camera, f, 'KPI right camera')
                    tx.heat_map_visualization(df_right_camera, heat_map_dir, 'kpi_right_camera')

                elif fn == 'glasses':
                    print('\nKPI glasses')
                    dfglasses = tx.filterGlasses(tx.dfkpiall)
                    tx.bucketize(dfglasses, f, 'KPI glasses')
                    tx.heat_map_visualization(dfglasses, heat_map_dir, 'kpi_glasses')

                elif fn == 'no-glasses':
                    print('\nKPI no glasses')
                    dfnoglasses = tx.filterNoGlasses(tx.dfkpiall)
                    tx.bucketize(dfnoglasses, f, 'KPI no glasses')
                    tx.heat_map_visualization(dfnoglasses, heat_map_dir, 'kpi_no_glasses')

                elif fn == 'occluded':
                    print('\nKPI either eye occluded')
                    dfoccl = tx.filterOccluded(tx.dfkpiall)
                    tx.bucketize(dfoccl, f, 'KPI either eye occluded')
                    tx.heat_map_visualization(dfoccl, heat_map_dir, 'kpi_occluded')

                elif fn == 'topTen':
                    print('\nKPI top ten')
                    tx.bucketize(tx.dftop10, f, 'KPI top ten users')
                    tx.heat_map_visualization(tx.dftop10, heat_map_dir, 'kpi_top_ten')

                elif fn == 'botTen':
                    print('\nKPI bottom ten')
                    tx.bucketize(tx.dfbotTen, f, 'KPI bottom ten users')
                    tx.heat_map_visualization(tx.dfbotTen, heat_map_dir, 'kpi_bottom_ten')

                elif 'top-' in fn:
                    fn = fn.replace('top-', '')
                    nUsers = int(fn)

                    print('\nKPI top {} users'.format(nUsers))
                    dfTopN = tx.filterTopNUsers(tx.dfkpiall, nUsers)
                    tx.bucketize(dfTopN, f, 'KPI top {} users'.format(nUsers))
                    tx.heat_map_visualization(dfTopN, heat_map_dir, 'kpi_top_{}'.format(nUsers))

                    plt.clf()
                    degree_means = []
                    for users in range(1, nUsers):
                        df = tx.filterTopNUsers(tx.dfkpiall, users, metric='degree')
                        degree_means.append(df['degree_error'].mean())
                    plt.figure(1)

                    plt.plot(range(1, nUsers), degree_means)
                    plt.xlabel('KPI top {} users'.format(nUsers))
                    plt.ylabel('Error (degrees)')
                    plt.title('Gaze Top {}'.format(nUsers))
                    plt.savefig(os.path.join(output_path,
                                             "UserErrorAnalysis_top_" + str(nUsers)
                                             + "_degree_" + joint_extension + ".png"))

                    plt.clf()
                    cam_means = []
                    for users in range(1, nUsers):
                        df = tx.filterTopNUsers(tx.dfkpiall, users, metric='cm')
                        cam_means.append(df['cam_error'].mean())
                    plt.figure(1)

                    plt.plot(range(1, nUsers), cam_means)
                    plt.xlabel('KPI top {} users'.format(nUsers))
                    plt.ylabel('Error (cm)')
                    plt.title('Gaze Top {}'.format(nUsers))
                    plt.savefig(os.path.join(output_path,
                                             "UserErrorAnalysis_top_" + str(nUsers)
                                             + "_cm_" + joint_extension + ".png"))
                    plt.clf()

                    with open(os.path.join(output_path,
                                           'top{}_data.txt'.format(nUsers)), 'w+') as t:
                        t.write('{}\n'.format(degree_means))
                        t.write('{}\n'.format(cam_means))

                elif fn == 'glare-pupil':
                    if tx.df_glare is None:
                        tx.df_glare = tx.classifyGlare(tx.dfclean)

                    print('\nKPI glare pupil')
                    df_glare_pupil = tx.df_glare[tx.df_glare['glare_status'] == 1]
                    tx.bucketize(df_glare_pupil, f, 'KPI glare pupil')
                    tx.heat_map_visualization(df_glare_pupil, heat_map_dir, 'kpi_glare_pupil')
                    tx.save_images(df_glare_pupil, sample_image_dir, 'glare_pupil', 50)

                elif fn == 'glare-eye':
                    if tx.df_glare is None:
                        tx.df_glare = tx.classifyGlare(tx.dfclean)

                    print('\nKPI glare eye')
                    df_glare_eye = tx.df_glare[tx.df_glare['glare_status'] == 2]
                    tx.bucketize(df_glare_eye, f, 'KPI glare eye')
                    tx.heat_map_visualization(df_glare_eye, heat_map_dir, 'kpi_glare_eye')
                    tx.save_images(df_glare_eye, sample_image_dir, 'glare_eye', 50)

                elif fn == 'glint-pupil':
                    if tx.df_glare is None:
                        tx.df_glare = tx.classifyGlare(tx.dfclean)

                    print('\nKPI glint pupil')
                    df_glint_pupil = tx.df_glare[tx.df_glare['glare_status'] == 3]
                    tx.bucketize(df_glint_pupil, f, 'KPI glint pupil')
                    tx.heat_map_visualization(df_glint_pupil, heat_map_dir, 'kpi_glint_pupil')
                    tx.save_images(df_glint_pupil, sample_image_dir, 'glint_pupil', 50)

                elif fn == 'glint-eye':
                    if tx.df_glare is None:
                        tx.df_glare = tx.classifyGlare(tx.dfclean)

                    print('\nKPI glint eye')
                    df_glint_eye = tx.df_glare[tx.df_glare['glare_status'] == 4]
                    tx.bucketize(df_glint_eye, f, 'KPI glint eye')
                    tx.heat_map_visualization(df_glint_eye, heat_map_dir, 'kpi_glint_eye')
                    tx.save_images(df_glint_eye, sample_image_dir, 'glint_eye', 50)

                elif fn == 'no-glare':
                    if tx.df_glare is None:
                        tx.df_glare = tx.classifyGlare(tx.dfclean)

                    print('\nKPI No glare eye')
                    df_no_glare_eye = tx.df_glare[tx.df_glare['glare_status'] == 5]
                    tx.bucketize(df_no_glare_eye, f, 'KPI No glare eye')
                    tx.heat_map_visualization(df_no_glare_eye, heat_map_dir, 'kpi_no_glare_eye')
                    tx.save_images(df_no_glare_eye, sample_image_dir, 'no_galre_eye', 50)

                elif fn == 'class-0':
                    if tx.df_glare is None:
                        tx.df_glare = tx.classifyGlare(tx.dfclean)

                    print('\nKPI glare class 0')
                    df_glare_class_0 = tx.df_glare[tx.df_glare['glare_status'] == 0]
                    tx.bucketize(df_glare_class_0, f, 'KPI glare class 0')
                    tx.save_images(df_glare_class_0, sample_image_dir, 'glare_class_0', 50)

                elif fn == 'bright-pupil':
                    if tx.df_glare is None:
                        tx.df_glare = tx.classifyGlare(tx.dfclean)

                    print('\nKPI bright pupil')
                    df_bright_pupil = tx.df_glare[tx.df_glare['bright_pupil'] == 'bright_pupil']
                    tx.bucketize(df_bright_pupil, f, 'KPI bright pupil')
                    tx.save_images(df_bright_pupil, sample_image_dir, 'bright_pupil', 50)

                elif fn == 'non-bright-pupil':
                    if tx.df_glare is None:
                        tx.df_glare = tx.classifyGlare(tx.dfclean)

                    print('\nKPI non-bright pupil')
                    df_non_bright_pupil = tx.df_glare[tx.df_glare['bright_pupil'] == 'normal']
                    tx.bucketize(df_non_bright_pupil, f, 'KPI Non-bright pupil')

                elif fn == 'hpPitch30':
                    print('\nKPI HP Pitch [< 30]')
                    tx.bucketize(tx.hppitch30, f, 'KPI HP Pitch [< 30]')
                    tx.heat_map_visualization(tx.hppitch30, heat_map_dir, 'kpi_hp_pitch_30')

                elif fn == 'hpPitch60':
                    print('\nKPI HP Pitch [30-60]')
                    tx.bucketize(tx.hppitch60, f, 'KPI HP Pitch [30-60]')
                    tx.heat_map_visualization(tx.hppitch60, heat_map_dir, 'kpi_hp_pitch_60')

                elif fn == 'hpPitch90':
                    print('\nKPI HP Pitch [>60]')
                    tx.bucketize(tx.hppitch90, f, 'KPI HP Pitch [> 60]')
                    tx.heat_map_visualization(tx.hppitch90, heat_map_dir, 'kpi_hp_pitch_90')

                elif fn == 'hpYaw30':
                    print('\nKPI HP Yaw [< 30]')
                    tx.bucketize(tx.hpyaw30, f, 'KPI HP Yaw [< 30]')
                    tx.heat_map_visualization(tx.hpyaw30, heat_map_dir, 'kpi_hp_yaw_30')

                elif fn == 'hpYaw60':
                    print('\nKPI HP Yaw [30-60]')
                    tx.bucketize(tx.hpyaw60, f, 'KPI HP Yaw [30-60]')
                    tx.heat_map_visualization(tx.hpyaw60, heat_map_dir, 'kpi_hp_yaw_60')

                elif fn == 'hpYaw90':
                    print('\nKPI HP Yaw [>60]')
                    tx.bucketize(tx.hpyaw90, f, 'KPI HP Yaw [> 60]')
                    tx.heat_map_visualization(tx.hpyaw90, heat_map_dir, 'kpi_hp_yaw_90')

                elif fn == 'hp15':
                    print('\nKPI HP [-15,15]')
                    tx.bucketize(tx.hp15, f, 'KPI HP [-15,15]')
                    tx.heat_map_visualization(tx.hp15, heat_map_dir, 'kpi_hp_15')

                elif fn == 'hp30':
                    print('\nKPI HP [-30,30]')
                    tx.bucketize(tx.hp30, f, 'KPI HP [-30,30]')
                    tx.heat_map_visualization(tx.hp30, heat_map_dir, 'kpi_hp_30')

                elif fn == 'hp45':
                    print('\nKPI HP [-45,45]')
                    tx.bucketize(tx.hp45, f, 'KPI HP [-45,45]')
                    tx.heat_map_visualization(tx.hp45, heat_map_dir, 'kpi_hp_45')

                elif fn == 'hp60':
                    print('\nKPI HP [-60,60]')
                    tx.bucketize(tx.hp60, f, 'KPI HP [-60,60]')
                    tx.heat_map_visualization(tx.hp60, heat_map_dir, 'kpi_hp_60')

                elif fn == 'cone45_center':
                    print('\nKPI cone45_center [-22.5,22.5]')
                    tx.bucketize(tx.coneCenter, f, 'KPI cone45_center [-22.5,22.5]')
                    tx.heat_map_visualization(tx.coneCenter, heat_map_dir,
                                              'kpi_cone45_center')

                elif fn == 'cone45_periphery':
                    print('\nKPI cone45_periphery')
                    tx.bucketize(tx.conePeriphery, f, 'KPI cone45_periphery')
                    tx.heat_map_visualization(tx.conePeriphery, heat_map_dir,
                                              'kpi_cone45_periphery')

                elif fn == 'cone45_center_top10':
                    print('\nKPI cone45_center_top10 [-22.5,22.5]')
                    tx.bucketize(tx.coneCenterTop10, f, 'KPI cone45_center_top10 [-22.5,22.5]')
                    tx.heat_map_visualization(tx.coneCenterTop10, heat_map_dir,
                                              'kpi_cone45_center_top10')

                elif fn == 'cone45_periphery_top10':
                    print('\nKPI cone45_periphery_top10')
                    tx.bucketize(tx.conePeripheryTop10, f, 'KPI cone45_periphery_top10')
                    tx.heat_map_visualization(tx.conePeripheryTop10, heat_map_dir,
                                              'kpi_cone45_periphery_top10')

            f.close()
