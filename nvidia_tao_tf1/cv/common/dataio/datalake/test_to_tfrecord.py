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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from nvidia_tao_tf1.cv.common.dataio.datalake.to_tfrecord import (
    write_from_pandas,
)


class ToTFRecordTest(unittest.TestCase):
    def test_pandas_to_tfrecord(self):
        """Test conversion from Pandas DF to TFRecord."""

        test_tf_folder = '/tmp/'
        test_tf_filepath = "{}/test.tfrecords".format(test_tf_folder)
        test_writer = tf.python_io.TFRecordWriter(test_tf_filepath)

        test_data = {
            'cosmos': 'driveix.cosmos639',
            'facebbx_h': 439.0,
            'facebbx_w': 439.0,
            'facebbx_x': 429.0,
            'facebbx_y': 252.0,
            'gaze_cam_x': -1043.777333,
            'gaze_cam_y': -1.3017040000000009,
            'gaze_cam_z': 539.090724,
            'gaze_leye_phi': 1.366890967655318,
            'gaze_leye_theta': 0.06599873509000821,
            'gaze_mid_phi': 1.3407598348898804,
            'gaze_mid_theta': 0.07245378463757722,
            'gaze_origin_facecenter_x': -51.39234825144855,
            'gaze_origin_facecenter_y': 92.04489429673804,
            'gaze_origin_facecenter_z': 761.1916839669866,
            'gaze_origin_leye_x': -20.84654251367733,
            'gaze_origin_leye_y': 67.7389280534117,
            'gaze_origin_leye_z': 750.6114456760628,
            'gaze_origin_midpoint_x': -44.275693768483954,
            'gaze_origin_midpoint_y': 73.20562575359814,
            'gaze_origin_midpoint_z': 773.155872135953,
            'gaze_origin_reye_x': -67.70484502329059,
            'gaze_origin_reye_y': 78.67232345378457,
            'gaze_origin_reye_z': 795.7002985958433,
            'gaze_phi': 1.340302336944776,
            'gaze_reye_phi': 1.3137137062342341,
            'gaze_reye_theta': 0.0790765716648666,
            'gaze_screen_x': 0.0,
            'gaze_theta': 0.07253765337743741,
            'hp_phi': 0.857525957552,
            'hp_pitch': -37.9900557691,
            'hp_roll': -13.1337805057,
            'hp_theta': 0.49720801543,
            'hp_yaw': -43.1392202242,
            'image_frame_name': '2_0_vc00_52.png',
            'image_frame_path': '/home/driveix.cosmos639/GazeData/orgData/\
                s589-gaze-incar-lincoln-car22-0/pngData/RttjJvFIa5uMqEpo/region_9',
            'is_valid_tp': True,
            'landmarks_source': 'Nvhelnet_v12',
            'leyebbx_h': 443.0,
            'leyebbx_w': 614.0,
            'leyebbx_x': 574.0,
            'leyebbx_y': 403.0,
            'norm_face_gaze_phi': 1.4057945156534788,
            'norm_face_gaze_theta': 0.1869903503541677,
            'norm_face_hp_phi': 0.8400734880627186,
            'norm_face_hp_theta': 0.5104548952594155,
            'norm_face_warping_matrix': np.array([
                0.9901051,
                -0.11478256,
                0.08072733,
                0.12336678,
                0.986143,
                -0.11091729,
                -0.06687732,
                0.11977884,
                0.9905455,
                ]),
            'norm_facebbx_h': 448.0,
            'norm_facebbx_w': 448.0,
            'norm_facebbx_x': 416.0,
            'norm_facebbx_y': 176.0,
            'norm_image_frame_path': '/home/driveix.cosmos639/GazeData/orgData/\
                s589-gaze-incar-lincoln-car22-0/Norm_Data_OldFM_Nvhelnet_v12/\
                    frame/RttjJvFIa5uMqEpo/region_9/2_0_vc00_52.png',
            'norm_leye_gaze_phi': 1.3862417985239928,
            'norm_leye_gaze_theta': 0.1939742569441609,
            'norm_leye_hp_phi': 0.7970397456237868,
            'norm_leye_hp_theta': 0.5465758298139601,
            'norm_leye_warping_matrix': np.array([
                0.9885556,
                -0.14530005,
                0.04056751,
                0.14830144,
                0.9852998,
                -0.08479964,
                -0.02764977,
                0.08984538,
                0.99557185,
                ]),
            'norm_leyebbx_h': 160.0,
            'norm_leyebbx_w': 160.0,
            'norm_leyebbx_x': 664.0,
            'norm_leyebbx_y': 258.0,
            'norm_reye_gaze_phi': 1.3884122795611544,
            'norm_reye_gaze_theta': 0.200662527144421,
            'norm_reye_hp_phi': 0.8544979998120636,
            'norm_reye_hp_theta': 0.5465758298139601,
            'norm_reyebbx_h': 160.0,
            'norm_reyebbx_w': 160.0,
            'norm_reyebbx_x': 522.0,
            'norm_reyebbx_y': 263.0,
            'norm_reyee_warping_matrix': np.array([
                0.98533636,
                -0.13990074,
                0.09767291,
                0.14830144,
                0.9852998,
                -0.08479964,
                -0.08437356,
                0.09804121,
                0.9915992,
                ]),
            'num_landmarks': 80.0,
            'percentile_of_out_of_frame': 0.0,
            'region_id': 'region_9',
            'reyebbx_h': 449.0,
            'reyebbx_w': 534.0,
            'reyebbx_x': 511.0,
            'reyebbx_y': 426.0,
            'sample_type': 'filtered',
            'set_name': 's589-gaze-incar-lincoln-car22-0',
            'setup_type': 'incar',
            'user_id': 'RttjJvFIa5uMqEpo',
            'version': '1586812763',
        }
        test_df = pd.DataFrame([test_data])

        write_from_pandas(test_df, test_writer)
        assert os.path.exists(test_tf_filepath)
