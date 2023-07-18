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

"""Generate manually engineered eye features."""

import abc
import numpy as np


num_pts_eye_outline = 6
num_pts_pupil_outline = 4
eye_index = 36
eye_end_index_diff = 3
pupil_index = 68


class EyeFeaturesStrategy(object):
    """Abstract class with common methods for eye features generation."""

    __metaclass__ = abc.ABCMeta

    @staticmethod
    def normalize_gaze_coord(coord_row, origin, distance):
        """Normalize coordinate to reduce the range."""
        return (coord_row - origin) / distance

    @staticmethod
    def _get_flattened_list(np_arr):
        assert isinstance(np_arr, np.ndarray)
        return np_arr.reshape(-1).tolist()

    def __init__(self, landmarks_2D):
        """Initialize landmarks."""
        self._landmarks_2D = landmarks_2D

    @abc.abstractmethod
    def get_eye_features(self):
        """Return generated eye features."""
        pass


class PupilStrategy(EyeFeaturesStrategy):
    """Eye features generation with pupils."""

    @staticmethod
    def _get_pupil_center(pupil_coord):
        pupil_remaining = pupil_coord
        separate_by_zero_indices = np.where(pupil_coord == 0)
        pupil_zero_indices = separate_by_zero_indices[0]
        if 0 in pupil_zero_indices or 2 in pupil_zero_indices:
            pupil_remaining = np.delete(pupil_coord, [0, 2], axis=0)
        elif 1 in pupil_zero_indices or 3 in pupil_zero_indices:
            pupil_remaining = np.delete(pupil_coord, [1, 3], axis=0)
        return np.mean(pupil_remaining, axis=0)

    @staticmethod
    def _get_eye_pupil_ratio(eye_coord, pupil_center):
        max_eye_x = np.amax(eye_coord[:, 0])
        max_eye_y = np.amax(eye_coord[:, 1])
        min_eye_x = np.amin(eye_coord[:, 0])
        min_eye_y = np.amin(eye_coord[:, 1])

        max_min_x_to_pupil_ratio = np.abs(
            max_eye_x - pupil_center[0]) / (np.abs(pupil_center[0] - min_eye_x) + 1)
        max_min_y_to_pupil_ratio = np.abs(
            max_eye_y - pupil_center[1]) / (np.abs(pupil_center[1] - min_eye_y) + 1)

        return max_min_x_to_pupil_ratio, max_min_y_to_pupil_ratio

    def _extract_side_eye_features(self, eye_coord, pupil_coord):
        pupil_center = self._get_pupil_center(pupil_coord)

        dist_eye_pupil = pupil_center - eye_coord
        max_min_x_to_pupil_ratio, max_min_y_to_pupil_ratio = \
            self._get_eye_pupil_ratio(eye_coord, pupil_center)

        eye_origin = eye_coord[0]
        eye_dist = np.linalg.norm(eye_origin - eye_coord[3])
        if eye_dist == 0:
            eye_dist += np.finfo(float).eps

        norm_pupil_center = np.apply_along_axis(
            self.normalize_gaze_coord,
            0,
            pupil_center,
            origin=eye_origin,
            distance=eye_dist)
        norm_eye_coord = np.apply_along_axis(
            self.normalize_gaze_coord,
            1,
            eye_coord,
            origin=eye_origin,
            distance=eye_dist)

        eye_features = self._get_flattened_list(norm_eye_coord)
        eye_features.extend(self._get_flattened_list(norm_pupil_center))
        eye_features.extend(self._get_flattened_list(dist_eye_pupil))
        eye_features.append(max_min_x_to_pupil_ratio)
        eye_features.append(max_min_y_to_pupil_ratio)

        return eye_features

    def get_eye_features(self, landmarks_2D):
        """Generate eye features with pupils."""

        # 6 coordinates per eye outline -> 12 coordinates total
        eye_pts_index = eye_index
        n_points_per_eye = num_pts_eye_outline
        right_eye_pts_index = eye_pts_index + n_points_per_eye
        left_eye_pts = landmarks_2D[eye_pts_index:right_eye_pts_index]
        right_eye_pts = \
            landmarks_2D[right_eye_pts_index:right_eye_pts_index + n_points_per_eye]

        # Flip to user's perspective
        left_eye_pts, right_eye_pts = right_eye_pts, left_eye_pts

        # 4 coordinates per pupil outline -> 8 coordinates total
        pupil_pts_index = pupil_index
        n_points_per_pupil = num_pts_pupil_outline
        right_pupil_pts_index = pupil_pts_index + n_points_per_pupil
        left_pupil_pts = landmarks_2D[pupil_pts_index:right_pupil_pts_index]
        right_pupil_pts = \
            landmarks_2D[right_pupil_pts_index:right_pupil_pts_index + n_points_per_pupil]

        # Flip to user's perspective
        left_pupil_pts, right_pupil_pts = right_pupil_pts, left_pupil_pts

        eye_features = []
        eye_features.extend(self._extract_side_eye_features(left_eye_pts, left_pupil_pts))
        eye_features.extend(self._extract_side_eye_features(right_eye_pts, right_pupil_pts))

        return np.asarray(eye_features, dtype=np.longdouble)


class NoPupilStrategy(EyeFeaturesStrategy):
    """Eye features generation without pupils."""

    def _extract_side_eye_features(self, eye_coord):
        eye_origin = eye_coord[0]
        eye_dist = np.linalg.norm(eye_origin - eye_coord[3])
        return np.apply_along_axis(
            self.normalize_gaze_coord,
            1,
            eye_coord,
            origin=eye_origin,
            distance=eye_dist)

    def get_eye_features(self, landmarks_2D):
        """Generate eye features without pupils."""

        # 6 coordinates per eye outline -> 12 coordinates total
        eye_pts_index = eye_index
        n_points_per_eye = num_pts_eye_outline
        right_eye_pts_index = eye_pts_index + n_points_per_eye

        left_eye_pts = self._landmarks_2D[eye_pts_index:right_eye_pts_index]
        right_eye_pts = \
            self._landmarks_2D[right_eye_pts_index:right_eye_pts_index + n_points_per_eye]

        # Flip to user's perspective
        left_eye_pts, right_eye_pts = right_eye_pts, left_eye_pts

        eye_features = np.empty([56, ], dtype=np.longdouble)
        eye_features.fill(-1)

        n_flattened_points_per_eye = n_points_per_eye * 2
        eye_features[:n_flattened_points_per_eye] = \
            self._extract_side_eye_features(left_eye_pts).reshape(-1)
        eye_features[28 : 28 + n_flattened_points_per_eye] = \
            self._extract_side_eye_features(right_eye_pts).reshape(-1)

        return eye_features


class EyeFeaturesGenerator(object):
    """Return generated eye features of an eye features strategy."""

    def __init__(self, landmarks_2D, n_landmarks):
        """Initialize landmarks and strategy."""
        self._landmarks_2D = landmarks_2D

        end_pupil_index = pupil_index + \
            2 * num_pts_eye_outline

        if n_landmarks < end_pupil_index:
            self._strategy = NoPupilStrategy(landmarks_2D)
        else:
            self._strategy = PupilStrategy(landmarks_2D)

    def get_eye_features(self):
        """Return eye features generated by strategy."""
        return self._strategy.get_eye_features(self._landmarks_2D)
