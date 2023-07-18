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

"""Strategy for tfrecord generation using SDK labels."""

import os
from nvidia_tao_tf1.cv.common.dataio.bbox_strategy import EyeBboxStrategy, FaceBboxStrategy
from nvidia_tao_tf1.cv.common.dataio.eye_status import EyeStatus
from nvidia_tao_tf1.cv.common.dataio.tfrecordlabels_strategy import TfRecordLabelsStrategy
from nvidia_tao_tf1.cv.common.dataio.utils import get_file_name_noext


class SdkLabelsStrategy(TfRecordLabelsStrategy):
    """Use SDK results from pre-labelled data to generate tfrecords."""

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
            norm_folder_name (str): Folder name to save normalized face, eyes and frame images.
            save_images (bool): Whether to generate new folders and images for face crop, eyes, etc.
        """
        super(SdkLabelsStrategy, self).__init__(
            set_id,
            use_unique,
            logger,
            set_strategy,
            norm_folder_name,
            save_images)
        self._eye_status_path = os.path.join(self._paths.info_source_path, 'results')
        self._bounding_box_paths = [
            os.path.join(self._paths.info_source_path, 'two_eyes'),
            os.path.join(self._paths.info_source_path, 'one_eye'),
            os.path.join(self._paths.info_source_path, 'no_eye')
        ]

        self._landmarks_path = os.path.join(self._paths.info_source_path, 'facelandmark')

        # Use given landmarks path if possible
        if self._paths.landmarks_path:
            self._landmarks_path = self._paths.landmarks_path

    def extract_landmarks(self):
        """SDK Nvhelnet tfrecord generation read landmarks from files in landmarks path."""
        self._read_landmarks_from_path()

    def _process_eye_bbox(self, frame_dict, line_split, frame_w, frame_h):
        num_eyes_detected = 0

        left_eye_bbx = list(map(int, line_split[5:9]))
        left_eye_bbx_processed = self._set_strategy.get_pts(
            left_eye_bbx, frame_w, frame_h)
        left_eye_bbx = EyeBboxStrategy(
            frame_w,
            frame_h,
            left_eye_bbx_processed).get_square_bbox()
        frame_dict['train/lefteyebbx_x'] = left_eye_bbx[0]
        frame_dict['train/lefteyebbx_y'] = left_eye_bbx[1]
        frame_dict['train/lefteyebbx_w'] = left_eye_bbx[2]
        frame_dict['train/lefteyebbx_h'] = left_eye_bbx[3]
        if -1 not in left_eye_bbx:
            num_eyes_detected += 1

        right_eye_bbx = list(map(int, line_split[9:13]))
        right_eye_bbx_processed = self._set_strategy.get_pts(
            right_eye_bbx, frame_w, frame_h)
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

        frame_dict['train/num_eyes_detected'] = num_eyes_detected

    @staticmethod
    def _get_scaled_facebbx(x1, y1, x2, y2, frame_w, frame_h):
        distort_face_coords = [x1, y1, x2, y2]
        x1, y1, side_len = FaceBboxStrategy(
                frame_w,
                frame_h,
                x1,
                y1,
                x2 - x1,
                y2 - y1).get_square_bbox()
        scaled_facebbx = x1, y1, side_len, side_len

        return distort_face_coords, scaled_facebbx

    def extract_bbox(self):
        """SDK Nvhelnet tfrecord generation read bounding boxes from paths."""

        # No bounding boxes when using predicted landmarks
        if self._set_strategy._landmarks_folder_name is not None:
            return

        for bounding_box_path in self._bounding_box_paths:
            for user_file in os.listdir(bounding_box_path):
                user_name = get_file_name_noext(user_file)
                user_path = os.path.join(bounding_box_path, user_file)

                with open(user_path, 'r') as user_bbox:
                    for line in user_bbox:
                        line_split = line.rstrip().split(' ')
                        path_split = line_split[0].split('/')
                        frame_name = get_file_name_noext(path_split[-1])

                        if len(self._paths.regions) == 1 and self._paths.regions[0] == '':
                            # On bench data collection has no regions.
                            region_name = ''
                        else:
                            region_name = path_split[-2]

                        frame_dict = self._users[user_name][region_name][frame_name]
                        frame_dict['train/eye_detect_found'] = os.path.basename(
                            os.path.normpath(bounding_box_path))

                        # SDK: person's perspective for left and right eyes
                        try:
                            frame_w = frame_dict['train/image_frame_width']
                            frame_h = frame_dict['train/image_frame_height']

                            self._process_eye_bbox(frame_dict, line_split, frame_w, frame_h)

                            unprocesssed_face_bbx = list(map(int, line_split[1:5]))
                            x1 = unprocesssed_face_bbx[0]
                            y1 = unprocesssed_face_bbx[1]
                            w = unprocesssed_face_bbx[2]
                            h = unprocesssed_face_bbx[3]
                            x2 = x1 + w
                            y2 = y1 + h

                            face_coords, scaled_facebbx = self._get_scaled_facebbx(
                                x1, y1, x2, y2, frame_w, frame_h)

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
                                x1, y1, x2, y2 = self._set_strategy.get_pts(
                                    [x1, y1, x2, y2],
                                    frame_w,
                                    frame_h)
                                w = x2 - x1
                                h = y2 - y1
                                face_coords, scaled_facebbx = self._get_scaled_facebbx(
                                    x1, y1, x2, y2, frame_w, frame_h)

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
                                    'train/facebbx_x',
                                    'train/facebbx_y',
                                    'train/facebbx_w',
                                    'train/facebbx_h'
                                ],
                                scaled_facebbx)
                        except Exception:
                            self._logger.add_warning(
                                'Cannot draw valid eye bounding box {}'.format(user_path))
                            continue

    def extract_eye_status(self):
        """SDK Nvhelnet tfrecord generation read eye status as open / closed."""
        def _map_eye_status(status_val):
            if status_val == 0:
                return EyeStatus.closed_eye_status

            return EyeStatus.open_eye_status

        for user_file in os.listdir(self._eye_status_path):
            file_name_noext = get_file_name_noext(user_file)

            # Nvhelnet wink files store eye open / close status
            if not file_name_noext.endswith('_wink'):
                continue

            user_name = file_name_noext[:-5]
            user_path = os.path.join(self._eye_status_path, user_file)

            with open(user_path, 'r') as user_eye_status:
                for line in user_eye_status:
                    line_split = line.rstrip().split(' ')
                    path_split = line_split[0].split('/')
                    frame_name = get_file_name_noext(path_split[-1])
                    if len(self._paths.regions) == 1 and self._paths.regions[0] == '':
                        # On bench data collection has no regions.
                        region_name = ''
                    else:
                        region_name = path_split[-2]

                    self._users[user_name][region_name][frame_name][
                        'label/left_eye_status'] = _map_eye_status(line_split[1])
                    self._users[user_name][region_name][frame_name][
                        'label/right_eye_status'] = _map_eye_status(line_split[3])
