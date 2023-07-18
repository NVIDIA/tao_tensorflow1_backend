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

"""Extract information from Json files."""

import numpy as np


def safeints(x):
    """convert input to int value."""
    x = int(x)
    x = max(x, 0)
    return int(x)


def extract_landmarks_from_json(frame_annotations, num_keypoints):
    """extract landmarks from json file.

    Args:
        frame_annotations (dict): frame annotations
        num_keypoints: number of keypoints to be extracted
    Return:
        landmarks_2D (array): 2D landmarks points
        occlusions (array): occlusion masks
        num_landmarks: number of landmarks points
    """

    for chunk in frame_annotations:
        if 'class' not in chunk:
            continue

        chunk_class = str(chunk['class']).lower()

        landmarks_2D = []
        if chunk_class == 'fiducialpoints':
            x, y, occlusions = \
                    extract_fiducial_points(chunk, num_keypoints)
            landmarks_2D = np.asarray([x, y], dtype=np.longdouble).T
            return landmarks_2D, occlusions

    return None, None


def extract_face_bbox_from_json(frame_annotations):
    """extract landmarks from json file.

    Args:
        frame_annotations (dict): frame annotations

    Return:
        facex1 (int): top left point x
        facey1 (int): top left point y
        facex2 (int): bottom right point x
        facey2 (int): bottom right point y
    """

    for chunk in frame_annotations:
        if 'class' not in chunk:
            continue

        chunk_class = str(chunk['class']).lower()

        facex1 = -1
        facey1 = -1
        facex2 = -1
        facey2 = -1
        if chunk_class == 'facebbox':
            facex1, facey1, facex2, facey2 = extract_from_facebbox(
                        chunk, facex1, facey1, facex2, facey2)
            if -1 in (facex1, facey1, facex2, facey2):
                continue  # skip img

        return facex1, facey1, facex2, facey2


def extract_from_facebbox(chunk, facex1, facey1, facex2, facey2):
    """extract landmarks from json file.

    Args:
        chunk (dict): frame annotations chunk
        facex1 (int): top left point x
        facey1 (int): top left point y
        facex2 (int): bottom right point x
        facey2 (int): bottom right point y
    """

    if (
        'face_tight_bboxx' not in chunk or
        'face_tight_bboxy' not in chunk or
        'face_tight_bboxwidth' not in chunk or
        'face_tight_bboxheight' not in chunk
    ):
        return facex1, facey1, facex2, facey2

    facex1 = safeints(chunk['face_tight_bboxx'])
    facey1 = safeints(chunk['face_tight_bboxy'])
    facex2 = safeints(chunk['face_tight_bboxwidth']) + facex1
    facey2 = safeints(chunk['face_tight_bboxheight']) + facey1
    return facex1, facey1, facex2, facey2


def extract_fiducial_points(chunk, num_keypoints):
    """extract landmarks from json file.

    Args:
        chunk (dict): frame annotations chunk
        num_keypoints: number of keypoints to be extracted
    Return:
        x (float): 2D landmarks x
        y (float): 2D landmarks y
        occlusions (array): occlusion masks
        num_landmarks: number of landmarks points
    """

    x = [-1] * num_keypoints
    y = [-1] * num_keypoints
    occlusions = [-1] * num_keypoints
    num_landmarks = None

    for point in (
        point for point in chunk if (
            'class' not in point and 'version' not in point)):
        try:
            number = int(''.join(c for c in str(point) if c.isdigit()))

            if num_landmarks is None or number > num_landmarks:
                num_landmarks = number

            if 'x' in str(point).lower() and number <= num_keypoints:
                x[number - 1] = str(np.longdouble(chunk[point]))
            if 'y' in str(point).lower() and number <= num_keypoints:
                y[number - 1] = str(np.longdouble(chunk[point]))
            if (
                'occ' in str(point).lower() and
                number <= num_keypoints and
                chunk[point]
            ):
                occlusions[number - 1] = 1

            for index in range(num_landmarks):
                if occlusions[index] == -1:
                    occlusions[index] = 0

        except Exception:
            pass

    return x, y, occlusions


def get_square_bbox(bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_width, image_height):
    """get square bounding box.

    Args:
        bbox_x1 (int): bounding box top left x
        bbox_y1 (int): bounding box top left y
        bbox_x2 (int): bounding box bottom right x
        bbox_y2 (int): bounding box bottom right y
    """

    x = bbox_x1
    y = bbox_y1
    width = bbox_x2 - x
    height = bbox_y2 - y

    # transform it into a square bbox wrt the longer side
    longer_side = max(width, height)
    new_width = longer_side
    new_height = longer_side
    x = int(x - (new_width - width) / 2)
    y = int(y - (new_height - height) / 2)
    x = min(max(x, 0), image_width)
    y = min(max(y, 0), image_height)
    new_width = min(new_width, image_width - x)
    new_height = min(new_height, image_height - y)
    new_width = min(new_width, new_height)
    new_height = new_width  # make it a square bbox

    return x, y, x + new_width, y + new_height
