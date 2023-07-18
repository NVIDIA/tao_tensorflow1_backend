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

"""Strategy for tfrecord generation using post JSON labels."""

from collections import defaultdict
import json
import os
import numpy as np
from nvidia_tao_tf1.cv.common.dataio.bbox_strategy import EyeBboxStrategy, FaceBboxStrategy
from nvidia_tao_tf1.cv.common.dataio.eye_features_generator import (
    eye_end_index_diff, eye_index, EyeFeaturesGenerator, num_pts_eye_outline)
from nvidia_tao_tf1.cv.common.dataio.eye_status import EyeStatus
from nvidia_tao_tf1.cv.common.dataio.tfrecordlabels_strategy import TfRecordLabelsStrategy
from nvidia_tao_tf1.cv.common.dataio.utils import get_file_ext, get_file_name_noext


class JsonLabelsStrategy(TfRecordLabelsStrategy):
    """Use JSON post labels to generate tfrecords."""

    def __init__(
        self,
        set_id,
        use_unique,
        logger,
        set_strategy,
        norm_folder_name,
        save_images
    ):
        """
        Initialize parameters.

        Args:
            set_id (str): Set for which to generate tfrecords.
            use_unique (bool): Only create records for first frame in a series if true.
            logger (Logger object): Report failures and number of tfrecords lost for tracking.
            set_strategy (SetStrategy object): Strategy for set type (gaze / eoc).
            norm_folder_name (str): Folder name to save normalized face, eyes and frame images.
            save_images (bool): Whether to generate new folders and images for face crop, eyes, etc.
        """
        super(JsonLabelsStrategy, self).__init__(
            set_id,
            use_unique,
            logger,
            set_strategy,
            norm_folder_name,
            save_images)
        self._landmarks_path = self._paths.landmarks_path

        self._users_json = defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict())))

        self._extract_json()

    def _extract_json(self):
        for user_json_file in os.listdir(self._paths.info_source_path):
            user_path = os.path.join(self._paths.info_source_path, user_json_file)

            if get_file_ext(user_json_file) != '.json':
                continue

            # On bench data format for user_json_file: <set-id>_<user>.json
            user_name = str(get_file_name_noext(user_json_file)[len(self._set_id) + 1:])
            # In-car data format for user_json_file: <set-id>_<user>_<region>.json
            if len(self._paths.regions) > 1:
                user_name = user_name.split('_')[0]

            try:
                with open(user_path, 'r') as user_json:
                    user_read_json = json.load(user_json)
            except Exception:
                self._logger.add_error(
                    'Json file improperly formatted for user {}'.format(user_name))

            for frame_json in user_read_json:
                if 'annotations' not in frame_json:
                    continue

                frame_name = get_file_name_noext(frame_json['filename'].split('/')[-1])
                if len(self._paths.regions) == 1 and self._paths.regions[0] == '':
                    # On bench data collection has no regions.
                    region_name = ''
                else:
                    region_name = frame_json['filename'].split('/')[-2]
                self._users_json[user_name][region_name][frame_name] = frame_json['annotations']

    def _extract_fiducial_points(self, chunk):
        x = [-1] * self.Pipeline_Constants.num_fid_points
        y = [-1] * self.Pipeline_Constants.num_fid_points
        occlusions = [-1] * self.Pipeline_Constants.num_fid_points
        num_landmarks = None

        for point in (
            point for point in chunk if (
                'class' not in point and 'version' not in point)):
            try:
                number = int(''.join(c for c in str(point) if c.isdigit()))

                if num_landmarks is None or number > num_landmarks:
                    num_landmarks = number

                if 'x' in str(point).lower() and number <= self.Pipeline_Constants.num_fid_points:
                    x[number - 1] = str(np.longdouble(chunk[point]))
                if 'y' in str(point).lower() and number <= self.Pipeline_Constants.num_fid_points:
                    y[number - 1] = str(np.longdouble(chunk[point]))
                if (
                    'occ' in str(point).lower() and
                    number <= self.Pipeline_Constants.num_fid_points and
                    chunk[point]
                ):
                    occlusions[number - 1] = 1

                for index in range(num_landmarks):
                    if occlusions[index] == -1:
                        occlusions[index] = 0

            except Exception as e:
                print('Exception occured during parsing')
                print(str(e))
                print(str(point))

        return x, y, occlusions, num_landmarks

    def _extract_landmarks_from_json(self):
        for user in self._users_json.keys():
            for region in self._users_json[user].keys():
                for frame in self._users_json[user][region].keys():
                    json_frame_dict = self._users_json[user][region][frame]
                    frame_dict = self._users[user][region][frame]

                    for chunk in json_frame_dict:
                        if 'class' not in chunk:
                            continue

                        chunk_class = str(chunk['class']).lower()

                        if chunk_class == 'fiducialpoints':
                            x, y, occlusions, num_landmarks = self._extract_fiducial_points(chunk)

                            landmarks_2D = np.asarray([x, y], dtype=np.longdouble).T

                            try:
                                frame_dict['internal/landmarks_2D_distort'] = np.copy(landmarks_2D)
                                landmarks_2D[:num_landmarks] = np.asarray(
                                    self._set_strategy.get_pts(
                                        landmarks_2D[:num_landmarks],
                                        frame_dict['train/image_frame_width'],
                                        frame_dict['train/image_frame_height'])).reshape(-1, 2)
                                frame_dict['internal/landmarks_2D'] = landmarks_2D

                                frame_dict['train/num_keypoints'] = num_landmarks
                                frame_dict['train/landmarks'] = landmarks_2D.reshape(-1)
                                frame_dict['train/landmarks_occ'] = np.asarray(occlusions).T

                                # Note eye_features only dependent on landmarks
                                frame_dict['train/eye_features'] = EyeFeaturesGenerator(
                                    landmarks_2D,
                                    num_landmarks).get_eye_features()
                            except Exception:
                                continue

    def extract_landmarks(self):
        """JSON tfrecord generation read landmarks from json when there is no given path."""
        if self._landmarks_path is None:
            self._extract_landmarks_from_json()
            return

        self._read_landmarks_from_path()

    @staticmethod
    def _get_scaled_facebbx(facex1, facey1, facex2, facey2, frame_w, frame_h):
        def _get_facebbx_legacy(x1, y1, x2, y2):
            h = y2 - y1
            y1 = max(0, y1 - 0.2 * h)
            return x1, y1, x2, y2

        distort_face_coords = [facex1, facey1, facex2, facey2]

        legacy_face_coords = _get_facebbx_legacy(*distort_face_coords)

        x1, y1, x2, y2 = legacy_face_coords
        x1, y1, side_len = FaceBboxStrategy(
                frame_w,
                frame_h,
                x1,
                y1,
                x2 - x1,
                y2 - y1).get_square_bbox()
        scaled_facebbx = x1, y1, side_len, side_len

        return distort_face_coords, legacy_face_coords, scaled_facebbx

    @staticmethod
    def _safeints(x):
        x = int(x)
        x = max(x, 0)
        return int(x)

    @classmethod
    def _extract_from_facebbox(cls, chunk, facex1, facey1, facex2, facey2):
        if (
            'face_tight_bboxx' not in chunk or
            'face_tight_bboxy' not in chunk or
            'face_tight_bboxwidth' not in chunk or
            'face_tight_bboxheight' not in chunk
        ):
            return facex1, facey1, facex2, facey2

        facex1 = cls._safeints(chunk['face_tight_bboxx'])
        facey1 = cls._safeints(chunk['face_tight_bboxy'])
        facex2 = cls._safeints(chunk['face_tight_bboxwidth']) + facex1
        facey2 = cls._safeints(chunk['face_tight_bboxheight']) + facey1
        return facex1, facey1, facex2, facey2

    @classmethod
    def _extract_from_rect(cls, chunk, prevArea, facex1, facey1, facex2, facey2):
        height = chunk['height']
        width = chunk['width']

        if prevArea == 0:
            facex1 = cls._safeints(chunk['x'])
            facey1 = cls._safeints(chunk['y'])
            facex2 = cls._safeints(chunk['width']) + facex1
            facey2 = cls._safeints(chunk['height']) + facey1
            prevArea = height * width
        else:
            if (height * width) < prevArea:
                facex1 = cls._safeints(chunk['x'])
                facey1 = cls._safeints(chunk['y'])
                facex2 = cls._safeints(chunk['width']) + facex1
                facey2 = cls._safeints(chunk['height']) + facey1

        return prevArea, facex1, facey1, facex2, facey2

    def _extract_face_bbox(self):
        for user in self._users_json.keys():
            for region in self._users_json[user].keys():
                for frame in self._users_json[user][region].keys():
                    frame_dict = self._users[user][region][frame]

                    if (
                        'train/image_frame_width' not in frame_dict or
                        'train/image_frame_height' not in frame_dict
                    ):
                        self._logger.add_error(
                            '''Could not find frame width and height.
                            User {} frame {} may not exist'''.format(
                                user, frame))
                        continue

                    prevArea = 0
                    facex1 = -1
                    facey1 = -1
                    facex2 = -1
                    facey2 = -1

                    json_frame_dict = self._users_json[user][region][frame]
                    for chunk in json_frame_dict:
                        if 'class' not in chunk:
                            continue

                        chunk_class = str(chunk['class']).lower()

                        if chunk_class == 'rect':
                            prevArea, facex1, facey1, facex2, facey2 = self._extract_from_rect(
                                chunk, prevArea, facex1, facey1, facex2, facey2)
                        elif chunk_class == 'facebbox':
                            facex1, facey1, facex2, facey2 = self._extract_from_facebbox(
                                chunk, facex1, facey1, facex2, facey2)

                    if -1 in (facex1, facey1, facex2, facey2):
                        self._logger.add_error(
                            'Unable to get face bounding box from json. User {}, frame {}'.format(
                                user, frame))
                        continue  # skip img

                    frame_w = frame_dict['train/image_frame_width']
                    frame_h = frame_dict['train/image_frame_height']

                    face_coords, legacy_face_coords, scaled_facebbx = \
                        self._get_scaled_facebbx(
                            facex1,
                            facey1,
                            facex2,
                            facey2,
                            frame_w,
                            frame_h)

                    self._populate_frame_dict(
                        frame_dict,
                        [
                            'internal/facebbx_x_distort',
                            'internal/facebbx_y_distort',
                            'internal/facebbx_w_distort',
                            'internal/facebbx_h_distort',
                        ],
                        scaled_facebbx)

                    if self._set_strategy.use_undistort():
                        # Recalculate face bounding box so it is undistorted
                        facex1, facey1, facex2, facey2 = self._set_strategy.get_pts(
                            face_coords,
                            frame_w,
                            frame_h)

                        face_coords, legacy_face_coords, scaled_facebbx = \
                            self._get_scaled_facebbx(
                                facex1,
                                facey1,
                                facex2,
                                facey2,
                                frame_w,
                                frame_h)

                    self._populate_frame_dict(
                        frame_dict,
                        [
                            'train/tight_facebbx_x1',
                            'train/tight_facebbx_y1',
                            'train/tight_facebbx_x2',
                            'train/tight_facebbx_y2'
                        ],
                        list(map(int, face_coords)))

                    self._populate_frame_dict(
                        frame_dict,
                        [
                            'internal/facebbx_x1',
                            'internal/facebbx_y1',
                            'internal/facebbx_x2',
                            'internal/facebbx_y2'
                        ],
                        legacy_face_coords)

                    self._populate_frame_dict(
                        frame_dict,
                        [
                            'train/facebbx_x',
                            'train/facebbx_y',
                            'train/facebbx_w',
                            'train/facebbx_h'
                        ],
                        scaled_facebbx)

    def _extract_eye_bbox(self):
        def _format_eye_bbox(x1, y1, x2, y2, frame_dict):
            face_x1, face_y1 = frame_dict['internal/facebbx_x1'], frame_dict['internal/facebbx_y1']

            # Relative to face bbx
            left = np.asarray([x1 - face_x1, y1 - face_y1], dtype=np.longdouble)
            right = np.asarray([x2 - face_x1, y2 - face_y1], dtype=np.longdouble)

            width = np.power(np.sum(np.square(left - right)), 0.5)

            eye_pupil = np.true_divide(np.add(left, right), 2)

            upper_left = np.subtract(eye_pupil, width)
            lower_right = np.add(eye_pupil, np.true_divide(width, 1.5))
            coords = np.asarray([upper_left, lower_right], dtype=np.longdouble)

            # Back to frame coord
            back_global_coord = np.add(coords, np.asarray([face_x1, face_y1]))
            back_global_coord[:, 0] = np.clip(
                back_global_coord[:, 0],
                face_x1,
                frame_dict['internal/facebbx_x2'])
            back_global_coord[:, 1] = np.clip(
                back_global_coord[:, 1],
                face_y1,
                frame_dict['internal/facebbx_y2'])

            [eye_x1, eye_y1], [eye_x2, eye_y2] = back_global_coord.tolist()

            return eye_x1, eye_y1, eye_x2, eye_y2

        for user in self._users.keys():
            for region in self._users[user].keys():
                for frame in self._users[user][region].keys():
                    frame_dict = self._users[user][region][frame]

                    # Landmarks and facebbox should be extracted already
                    if (
                        'internal/landmarks_2D' not in frame_dict or
                        'internal/facebbx_x1' not in frame_dict or
                        'internal/facebbx_y1' not in frame_dict or
                        'internal/facebbx_x2' not in frame_dict or
                        'internal/facebbx_y2' not in frame_dict
                    ):
                        continue

                    landmarks_2D = frame_dict['internal/landmarks_2D']

                    right_eye_begin = eye_index
                    right_eye_end = right_eye_begin + eye_end_index_diff
                    r_x1, r_y1 = landmarks_2D[right_eye_begin].tolist()
                    r_x2, r_y2 = landmarks_2D[right_eye_end].tolist()

                    left_eye_begin = right_eye_begin + num_pts_eye_outline
                    left_eye_end = left_eye_begin + eye_end_index_diff
                    l_x1, l_y1 = landmarks_2D[left_eye_begin].tolist()
                    l_x2, l_y2 = landmarks_2D[left_eye_end].tolist()

                    right_eye_bbx = _format_eye_bbox(r_x1, r_y1, r_x2, r_y2, frame_dict)
                    left_eye_bbx = _format_eye_bbox(l_x1, l_y1, l_x2, l_y2, frame_dict)

                    try:
                        num_eyes_detected = 0
                        right_eye_bbx_processed = self._set_strategy.get_pts(
                            right_eye_bbx,
                            frame_dict['train/image_frame_width'],
                            frame_dict['train/image_frame_height'])
                        right_eye_bbx = EyeBboxStrategy(
                            frame_dict['train/image_frame_width'],
                            frame_dict['train/image_frame_height'],
                            right_eye_bbx_processed).get_square_bbox()
                        frame_dict['train/righteyebbx_x'] = right_eye_bbx[0]
                        frame_dict['train/righteyebbx_y'] = right_eye_bbx[1]
                        frame_dict['train/righteyebbx_w'] = right_eye_bbx[2]
                        frame_dict['train/righteyebbx_h'] = right_eye_bbx[3]
                        if -1 not in right_eye_bbx:
                            num_eyes_detected += 1

                        left_eye_bbx_processed = self._set_strategy.get_pts(
                            left_eye_bbx,
                            frame_dict['train/image_frame_width'],
                            frame_dict['train/image_frame_height'])
                        left_eye_bbx = EyeBboxStrategy(
                            frame_dict['train/image_frame_width'],
                            frame_dict['train/image_frame_height'],
                            left_eye_bbx_processed).get_square_bbox()
                        frame_dict['train/lefteyebbx_x'] = left_eye_bbx[0]
                        frame_dict['train/lefteyebbx_y'] = left_eye_bbx[1]
                        frame_dict['train/lefteyebbx_w'] = left_eye_bbx[2]
                        frame_dict['train/lefteyebbx_h'] = left_eye_bbx[3]
                        if -1 not in left_eye_bbx:
                            num_eyes_detected += 1

                        frame_dict['train/num_eyes_detected'] = num_eyes_detected
                    except Exception as ex:
                        self._logger.add_warning(
                            'User {} frame {} could not draw eye bounding boxes because {}'.format(
                                user,
                                frame,
                                repr(ex)))
                        continue

    def extract_bbox(self):
        """JSON tfrecord generation extract bounding boxes.

        Face bounding box extracted from json.
        Eye bounding boxes extracted from read landmarks.
        """
        if self._landmarks_path is None:
            self._extract_face_bbox()
            self._extract_eye_bbox()

    def extract_eye_status(self):
        """Fill eye status with value in JSON."""
        for user in self._users_json.keys():
            for region in self._users_json[user].keys():
                for frame in self._users_json[user][region].keys():
                    frame_dict = self._users_json[user][region][frame]
                    for chunk in frame_dict:
                        if 'class' not in chunk:
                            continue

                        chunk_class = str(chunk['class']).lower()
                        frame_dict = self._users[user][region][frame]

                        if chunk_class == 'eyes':
                            # JSON labels from the labeller's perspective,
                            # flip for user's perspective
                            if 'r_status' in chunk:
                                frame_dict['label/left_eye_status'] = chunk['r_status']
                            if 'l_status' in chunk:
                                frame_dict['label/right_eye_status'] = chunk['l_status']
                        elif chunk_class == 'eyeopen':
                            frame_dict['label/left_eye_status'] = EyeStatus.open_eye_status
                            frame_dict['label/right_eye_status'] = EyeStatus.open_eye_status
                        elif chunk_class == 'eyeclose':
                            frame_dict['label/left_eye_status'] = EyeStatus.closed_eye_status
                            frame_dict['label/right_eye_status'] = EyeStatus.closed_eye_status
