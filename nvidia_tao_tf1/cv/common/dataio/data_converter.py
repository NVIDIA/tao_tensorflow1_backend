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

"""Write tfrecord files."""

from enum import Enum
import json
import os
import subprocess
import sys
import numpy as np
import six
import tensorflow as tf
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.common.dataio.utils import mkdir


def _convert_unicode_to_str(item):
    if sys.version_info >= (3, 0):
        # unicode() no longer exists in Python 3
        if isinstance(item, str):
            return item.encode()
        return item

    if isinstance(item, unicode):  # noqa: disable=F821 # pylint: disable=E0602
        return item.encode()
    return item


def _bytes_feature(*values):
    """Convert unicode data to string for saving to TFRecords."""
    values = [_convert_unicode_to_str(value) for value in values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _float_feature(*values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(*values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _dtype_feature(ndarray):
    assert isinstance(ndarray, np.ndarray)
    dtype = ndarray.dtype
    if dtype in (np.float32, np.longdouble):
        return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray))
    if dtype == np.int64:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=ndarray))
    return None


class TfRecordType(Enum):
    """Class which bounds Enum values to indicate tfrecord type."""

    BYTES = 1
    FLOAT = 2
    INT64 = 3
    DTYPE_FLOAT = 4
    DTYPE_INT64 = 5


class DataConverter(object):
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
        'label/gaze_cam_x' : TfRecordType.FLOAT,
        'label/gaze_cam_y' : TfRecordType.FLOAT,
        'label/gaze_cam_z' : TfRecordType.FLOAT,
        'label/gaze_screen_x' : TfRecordType.FLOAT,
        'label/gaze_screen_y' : TfRecordType.FLOAT,
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

    read_features_dict = {}

    def __init__(self, tfrecord_files_path, use_lm_pred):
        """Initialize file paths and features.

        Args:
            tfrecord_files_path (path): Path to dump tfrecord files.
            use_lm_pred (bool): True if using predicted landmarks.
        """
        self._tfrecord_files_path = tfrecord_files_path

        self._feature_to_type_dict = self.feature_to_type
        if use_lm_pred:
            self._feature_to_type_dict = self.lm_pred_feature_to_type

        # Overwrite the tfrecords, cleanup
        if os.path.exists(expand_path(tfrecord_files_path)):
            subprocess.run('rm -r ' + tfrecord_files_path, shell=False)

        mkdir(tfrecord_files_path)

    @classmethod
    def enum_to_read_type(cls, enum_type):
        """Return tfrecord type based on Enum."""
        if enum_type not in cls.enum_to_read_dict:
            raise TypeError('Must be an instance of TfRecordType Enum')
        return cls.enum_to_read_dict[enum_type]

    @classmethod
    def get_read_features(cls, use_lm_pred):
        """Return dict of features to values."""
        feature_to_type_dict = cls.feature_to_type

        if use_lm_pred:
            feature_to_type_dict = cls.lm_pred_feature_to_type

        if not cls.read_features_dict:
            for feature in feature_to_type_dict.keys():
                feature_type = feature_to_type_dict[feature]
                cls.read_features_dict[feature] = cls.enum_to_read_type(feature_type)

        return cls.read_features_dict

    @classmethod
    def read_tfrecords(cls, tfrecord_files, use_lm_pred, print_json=True):
        """Read tfrecord files as JSON."""
        records = tf.data.TFRecordDataset(tfrecord_files)
        parsed = records.map(
            lambda x: tf.parse_single_example(x, cls.get_read_features(use_lm_pred)))
        iterator = parsed.make_one_shot_iterator()
        next_elem = iterator.get_next()
        combined_data = []

        feature_to_type_dict = cls.feature_to_type

        if use_lm_pred:
            feature_to_type_dict = cls.lm_pred_feature_to_type

        with tf.train.MonitoredTrainingSession() as sess:
            while not sess.should_stop():
                data = dict(sess.run(next_elem))

                for key, val in six.iteritems(data):
                    if feature_to_type_dict[key] in (
                        TfRecordType.DTYPE_FLOAT,
                        TfRecordType.DTYPE_INT64
                    ):
                        data[key] = (data[key].values).tolist()

                    elif isinstance(val, (np.float32, np.longdouble)):
                        # Keeps around 7 digits of precision of float32
                        py_val = str(np.asarray(val, dtype=np.longdouble))
                        data[key] = float(py_val)

                    elif isinstance(val, np.generic):
                        data[key] = np.asscalar(val)

                combined_data.append(data)

        if print_json:
            print(json.dumps(combined_data, indent=4, sort_keys=True))

        return combined_data

    def write_feature(self, feature, val, feature_dict):
        """Write dict feature and value into tfrecord format."""
        feature_type = self._feature_to_type_dict[feature]
        if feature_type == TfRecordType.BYTES:
            feature_dict[feature] = _bytes_feature(val)
        elif feature_type == TfRecordType.FLOAT:
            feature_dict[feature] = _float_feature(val)
        elif feature_type == TfRecordType.INT64:
            feature_dict[feature] = _int64_feature(val)
        elif feature_type in (TfRecordType.DTYPE_FLOAT, TfRecordType.DTYPE_INT64):
            feature_dict[feature] = _dtype_feature(val)

    def write_user_tfrecords(self, users_dict):
        """Write collected data dict into tfrecords."""
        for user in users_dict.keys():
            tfrecord_file_path = expand_path(f"{self._tfrecord_files_path}/{user}.tfrecords")
            user_writer = tf.python_io.TFRecordWriter(tfrecord_file_path)

            for region in users_dict[user].keys():
                for frame in users_dict[user][region].keys():
                    frame_features = {}
                    frame_data_dict = users_dict[user][region][frame]

                    for feature in self._feature_to_type_dict.keys():
                        self.write_feature(feature, frame_data_dict[feature], frame_features)
                    example = tf.train.Example(features=tf.train.Features(feature=frame_features))
                    user_writer.write(example.SerializeToString())
            user_writer.close()

    def write_combined_tfrecords(self, combined_dict, test_users, validation_users, train_users):
        """Write collected data after splitting into test, train, validation into tfrecords."""
        def _write_category_tfrecords(combined_dict, category_users, tfrecord_file_path):
            category_writer = tf.python_io.TFRecordWriter(tfrecord_file_path)
            for user in category_users:
                for region in combined_dict[user].keys():
                    for frame in combined_dict[user][region].keys():
                        frame_features = {}
                        frame_data_dict = combined_dict[user][region][frame]

                        for feature in self._feature_to_type_dict.keys():
                            self.write_feature(feature, frame_data_dict[feature], frame_features)
                        example = tf.train.Example(
                            features=tf.train.Features(feature=frame_features))
                        category_writer.write(example.SerializeToString())
            category_writer.close()

        test_path = os.path.join(self._tfrecord_files_path, 'test.tfrecords')
        validation_path = os.path.join(self._tfrecord_files_path, 'validate.tfrecords')
        train_path = os.path.join(self._tfrecord_files_path, 'train.tfrecords')

        _write_category_tfrecords(combined_dict, test_users, test_path)
        _write_category_tfrecords(combined_dict, validation_users, validation_path)
        _write_category_tfrecords(combined_dict, train_users, train_path)
