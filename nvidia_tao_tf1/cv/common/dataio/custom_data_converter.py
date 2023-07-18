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

"""Generate tfrecords data for customer input data."""

import six
import tensorflow as tf
from nvidia_tao_tf1.cv.common.dataio.data_converter import DataConverter, TfRecordType


class CustomDataConverter(DataConverter):
    """Converts a dataset to TFRecords."""

    feature_to_type = {
        'train/image_frame_name' : TfRecordType.BYTES,
        'train/image_frame_width' : TfRecordType.INT64,
        'train/image_frame_height' : TfRecordType.INT64,
        'train/facebbx_x' : TfRecordType.INT64,
        'train/facebbx_y' : TfRecordType.INT64,
        'train/facebbx_w' : TfRecordType.INT64,
        'train/facebbx_h' : TfRecordType.INT64,
        'train/lefteyebbx_x' : TfRecordType.INT64,
        'train/lefteyebbx_y' : TfRecordType.INT64,
        'train/lefteyebbx_w' : TfRecordType.INT64,
        'train/lefteyebbx_h' : TfRecordType.INT64,
        'train/righteyebbx_x' : TfRecordType.INT64,
        'train/righteyebbx_y' : TfRecordType.INT64,
        'train/righteyebbx_w' : TfRecordType.INT64,
        'train/righteyebbx_h' : TfRecordType.INT64,
        'train/landmarks' : TfRecordType.DTYPE_FLOAT,
        'train/landmarks_occ' : TfRecordType.DTYPE_INT64,
        'label/left_eye_status' : TfRecordType.BYTES,
        'label/right_eye_status' : TfRecordType.BYTES,
        'train/num_keypoints' : TfRecordType.INT64,
        'train/tight_facebbx_x1' : TfRecordType.INT64,
        'train/tight_facebbx_y1' : TfRecordType.INT64,
        'train/tight_facebbx_x2' : TfRecordType.INT64,
        'train/tight_facebbx_y2' : TfRecordType.INT64,
        'label/hp_pitch': TfRecordType.FLOAT,           # Degrees
        'label/hp_yaw': TfRecordType.FLOAT,             # Degrees
        'label/hp_roll': TfRecordType.FLOAT,            # Degrees
        'label/theta': TfRecordType.FLOAT,              # Radians
        'label/phi': TfRecordType.FLOAT,                # Radians
        'label/mid_cam_x': TfRecordType.FLOAT,          # Mid eye center - x
        'label/mid_cam_y': TfRecordType.FLOAT,          # Mid eye center - y
        'label/mid_cam_z': TfRecordType.FLOAT,          # Mid eye center - z
        'label/lpc_cam_x': TfRecordType.FLOAT,          # Left eye center - x
        'label/lpc_cam_y': TfRecordType.FLOAT,          # Left eye center - y
        'label/lpc_cam_z': TfRecordType.FLOAT,          # Left eye center - z
        'label/rpc_cam_x': TfRecordType.FLOAT,          # Right eye center - x
        'label/rpc_cam_y': TfRecordType.FLOAT,          # Right eye center - y
        'label/rpc_cam_z': TfRecordType.FLOAT,          # Right eye center - z
        'train/valid_theta_phi' : TfRecordType.INT64,   # 1 if valid, 0 otherwise
        'label/theta_le' : TfRecordType.FLOAT,          # In radians
        'label/phi_le' : TfRecordType.FLOAT,            # In radians
        'label/theta_re' : TfRecordType.FLOAT,          # In radians
        'label/phi_re' : TfRecordType.FLOAT,            # In radians
        'label/theta_mid' : TfRecordType.FLOAT,         # In radians
        'label/phi_mid' : TfRecordType.FLOAT,           # In radians
        'label/head_pose_theta' : TfRecordType.FLOAT,   # In radians
        'label/head_pose_phi' : TfRecordType.FLOAT,     # In radians
        'train/eye_features' : TfRecordType.DTYPE_FLOAT,
        'train/source' : TfRecordType.BYTES,
        'train/num_eyes_detected': TfRecordType.INT64,
        'train/norm_frame_path': TfRecordType.BYTES,
        'label/norm_face_gaze_theta': TfRecordType.FLOAT,  # In radians
        'label/norm_face_gaze_phi': TfRecordType.FLOAT,  # In radians
        'label/norm_face_hp_theta': TfRecordType.FLOAT,  # In radians
        'label/norm_face_hp_phi': TfRecordType.FLOAT,  # In radians
        'label/norm_leye_gaze_theta': TfRecordType.FLOAT,  # In radians
        'label/norm_leye_gaze_phi': TfRecordType.FLOAT,  # In radians
        'label/norm_leye_hp_theta': TfRecordType.FLOAT,  # In radians
        'label/norm_leye_hp_phi': TfRecordType.FLOAT,  # In radians
        'label/norm_reye_gaze_theta': TfRecordType.FLOAT,  # In radians
        'label/norm_reye_gaze_phi': TfRecordType.FLOAT,  # In radians
        'label/norm_reye_hp_theta': TfRecordType.FLOAT,  # In radians
        'label/norm_reye_hp_phi': TfRecordType.FLOAT,  # In radians
        'train/norm_facebb_x': TfRecordType.INT64,
        'train/norm_facebb_y': TfRecordType.INT64,
        'train/norm_facebb_w': TfRecordType.INT64,
        'train/norm_facebb_h': TfRecordType.INT64,
        'train/norm_leyebb_x': TfRecordType.INT64,
        'train/norm_leyebb_y': TfRecordType.INT64,
        'train/norm_leyebb_w': TfRecordType.INT64,
        'train/norm_leyebb_h': TfRecordType.INT64,
        'train/norm_reyebb_x': TfRecordType.INT64,
        'train/norm_reyebb_y': TfRecordType.INT64,
        'train/norm_reyebb_w': TfRecordType.INT64,
        'train/norm_reyebb_h': TfRecordType.INT64,
        'train/norm_landmarks': TfRecordType.DTYPE_FLOAT,
        'train/norm_per_oof': TfRecordType.FLOAT,
        'train/landmarks_3D': TfRecordType.DTYPE_FLOAT
    }

    lm_pred_feature_to_type = {k : v for k, v in six.iteritems(feature_to_type)
                               if 'lefteyebbx' not in k and
                                  'righteyebbx' not in k and
                                  'facebbx' not in k and
                                  'num_eyes_detected' not in k}

    # Convert from enum type to read tfrecord type
    enum_to_read_dict = {
        TfRecordType.BYTES : tf.FixedLenFeature([], dtype=tf.string),
        TfRecordType.FLOAT : tf.FixedLenFeature([], dtype=tf.float32),
        TfRecordType.INT64 : tf.FixedLenFeature([], dtype=tf.int64),
        TfRecordType.DTYPE_FLOAT : tf.VarLenFeature(tf.float32),
        TfRecordType.DTYPE_INT64 : tf.VarLenFeature(tf.int64)
    }

    def __init__(self, use_lm_pred):
        """Initialize file paths and features.

        Args:
            tfrecord_files_path (path): Path to dump tfrecord files.
            use_lm_pred (bool): True if using predicted landmarks.
        """

        self._feature_to_type_dict = self.feature_to_type
        if use_lm_pred:
            self._feature_to_type_dict = self.lm_pred_feature_to_type

    def generate_frame_tfrecords(self, frame_dict):
        """Write collected data dict into tfrecords.

        Args:
            frame_dict (dict): dictionary for frame information
        """

        example_array = []
        for frame in frame_dict.keys():
            frame_features = {}
            frame_data_dict = frame_dict[frame]

            for feature in self._feature_to_type_dict.keys():
                self.write_feature(feature, frame_data_dict[feature], frame_features)
            example = tf.train.Example(features=tf.train.Features(feature=frame_features))
            example_array.append(example)

        return example_array
