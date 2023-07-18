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

"""Datastructures and functions for detection ground truths."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from itertools import repeat

import numpy as np
from six.moves import zip

from nvidia_tao_tf1.blocks.multi_source_loader.types import Bbox2DLabel
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.metadata import get_metadata_from_batch_ground_truth

# Python 2 vs Python 3 string.
try:
    # Python 2.
    unicode_func = unicode
except NameError:
    def unicode_func(x):
        """Convert a string to unicode."""
        if isinstance(x, str):
            return x
        return x.decode()

GroundTruth = namedtuple('GroundTruth', [
    'class_name',   # e.g. 'car'
    'bbox',  # (x1, y1, x2, y2)
    'truncation',  # float in KITTI
    'truncation_type',  # int (0 or 1) in Cyclops
    'occlusion',    # int in KITTI
    'is_cvip',  # boolean
    'world_bbox_z',  # float
    'front',  # float.
    'back',  # float.
    'orientation',  # float.
])

DONT_CARE_CLASS_NAME = 'dontcare'


def _populate_ground_truths(object_data):
    """Helper function to populate the GroundTruth instances from dataloader labels.

    Args:
        object_data (iterator): Each element is expected to contain fields in a particular order.

    Returns:
        ground_truths (list): List of GroundTruth instances populated appropriately.
    """
    ground_truths = []
    for object_class, bbox, occlusion, truncation, truncation_type, \
            is_cvip, world_bbox_z, non_facing, front, back, orientation in object_data:
        if non_facing:
            # TODO(@williamz): This is because of metrics code. Properly should be handled
            # differently.
            # Map non-facing road signs to dontcare class.
            object_class = DONT_CARE_CLASS_NAME
        ground_truth = GroundTruth(class_name=unicode_func(object_class), bbox=bbox,
                                   truncation=truncation, truncation_type=truncation_type,
                                   occlusion=occlusion, is_cvip=bool(is_cvip),
                                   world_bbox_z=world_bbox_z, front=front, back=back,
                                   orientation=orientation)
        ground_truths.append(ground_truth)

    return ground_truths


def get_ground_truth_objects_from_batch_ground_truth(batch_data):
    """Parse a batch of ground truth dictionaries to GroundTruth objects.

    Args:
        batch_data: Ground truth data parsed from tfrecords as a list of dicts. Each dict
        represents ground truth values such as bbox coordinates for one frame.

    Returns:
        ground_truths: List of list of GroundTruth objects parsed from this minibatch.
    """
    ground_truths = []

    for frame_data in batch_data:
        frame_ground_truths = []

        object_data = zip(frame_data['target/object_class'],
                          frame_data['target/bbox_coordinates'],
                          # Optional.
                          frame_data.get('target/occlusion', repeat(0)),
                          # Optional.
                          frame_data.get('target/truncation', repeat(0.0)),
                          # Optional.
                          frame_data.get('target/truncation_type', repeat(0)),
                          # Optional.
                          frame_data.get('target/is_cvip', repeat(False)),
                          # Optional.
                          frame_data.get('target/world_bbox_z', repeat(0.0)),
                          # Optional.
                          frame_data.get('target/non_facing', repeat(False)),
                          # Optional.
                          frame_data.get('target/front', repeat(-1.0)),
                          # Optional.
                          frame_data.get('target/back', repeat(-1.0)),
                          frame_data.get('target/orientation', repeat(-1.0))
                          )

        frame_ground_truths = _populate_ground_truths(object_data)

        ground_truths.append(frame_ground_truths)

    return ground_truths


def _get_features_from_bbox_2d_label(bbox_label, feature_name, start_idx, end_idx, default_value):
    """Helper function to extract relevant values in a Bbox2DLabel.

    Args:
        bbox_label (Bbox2DLabel): Label containing all the features for a minibatch.
        feature_name (str): Name of the field to look for in ``bbox_label``. These should be one
            of the fields of the ``Bbox2DLabel`` namedtuple.
        start_idx (int): Start index of the values.
        end_idx (int): End index of the values.
        default_value (variable): If the field is not "present" in ``bbox_label`` (e.g. an optional
            field such as 'front' or 'back' marker), the ``default_value`` iterator to return
            instead.

    Returns:
        If the ``feature_name`` is properly populated in the ``bbox_label``, then the values
            corresponding to the indices provided are returned. Otherwise, an iterator with
            ``default_value`` is returned.
    """
    feature_values = getattr(bbox_label, feature_name, [])

    if hasattr(feature_values, "values"):
        if isinstance(feature_values.values, np.ndarray) and feature_values.values.size > 0:
            return feature_values.values[start_idx:end_idx]
        # TODO(@williamz): consider removing this as only unit tests would realistically populate
        # the fields with lists instead of arrays.
        if type(feature_values.values) == list and len(feature_values.values) > 0:
            return feature_values.values[start_idx:end_idx]

    return repeat(default_value)


def get_ground_truth_objects_from_bbox_label(bbox_label):
    """Parse a Bbox2DLabel to GroundTruth objects.

    Args:
        bbox_label (Bbox2DLabel): Contains all the features for a minibatch.

    Returns:
        ground_truths: List of list of GroundTruth objects parsed from this minibatch.
    """
    ground_truths = []

    batch_size = bbox_label.vertices.coordinates.dense_shape[0]

    # Because the last frame(s) may very well be devoid of any labels, we need to make sure
    # the bincount still has ``batch_size`` entries.
    num_ground_truths_per_image = \
        np.bincount(
            bbox_label.object_class.indices[:, 0], minlength=batch_size)
    # The leading [0] here is to start the cumulative sum at 0 and not
    # num_ground_truths_per_image[0].
    start_end_indices = \
        np.cumsum(np.concatenate(([0], num_ground_truths_per_image)))
    bbox_coords = np.reshape(bbox_label.vertices.coordinates.values, (-1, 4))

    for batch_idx in range(batch_size):
        start_idx = start_end_indices[batch_idx]
        end_idx = start_end_indices[batch_idx+1]
        object_data = zip(
            bbox_label.object_class.values[start_idx:end_idx],
            bbox_coords[start_idx:end_idx, :],
            _get_features_from_bbox_2d_label(
                bbox_label, 'occlusion', start_idx, end_idx, 0),
            _get_features_from_bbox_2d_label(
                bbox_label, 'truncation', start_idx, end_idx, 0.0),
            _get_features_from_bbox_2d_label(
                bbox_label, 'truncation_type', start_idx, end_idx, 0),
            _get_features_from_bbox_2d_label(
                bbox_label, 'is_cvip', start_idx, end_idx, False),
            _get_features_from_bbox_2d_label(
                bbox_label, 'world_bbox_z', start_idx, end_idx, 0.0),
            _get_features_from_bbox_2d_label(
                bbox_label, 'non_facing', start_idx, end_idx, False),
            _get_features_from_bbox_2d_label(
                bbox_label, 'front', start_idx, end_idx, -1.0),
            _get_features_from_bbox_2d_label(
                bbox_label, 'back', start_idx, end_idx, -1.0),
            _get_features_from_bbox_2d_label(
                bbox_label, 'orientation', start_idx, end_idx, -1.0),
        )

        frame_ground_truths = _populate_ground_truths(object_data)

        ground_truths.append(frame_ground_truths)

    return ground_truths


def process_batch_ground_truth(batch_data, num_frames):
    """
    Process the batch ground truth dicts to get GroundTruth objects and frame_metadata dict.

    Args:
        batch_data: Ground truth data parsed from tfrecords as a list of dicts. Each dict
        represents ground truth values such as bbox coordinates for one frame.
        num_frames: Number of frames seen so far. Used as index for frame_metadata.

    Returns:
        ground_truths: List of list of GroundTruth objects parsed from this minibatch.
        frame_metadata: Metadata for the current minibatch. Keys are frame number (integer) and
            values are tuple (frame_identifier, camera_location, image_dimension). camera_location
            and image_dimension are None if not available.
    """
    if isinstance(batch_data, list):
        batch_groundtruth_objects = get_ground_truth_objects_from_batch_ground_truth(
            batch_data)
    elif isinstance(batch_data, Bbox2DLabel):
        batch_groundtruth_objects = get_ground_truth_objects_from_bbox_label(
            batch_data)
    else:
        raise NotImplementedError("Unhandled batch data of type: {}".format(type(batch_data)))

    frame_metadata = get_metadata_from_batch_ground_truth(
        batch_data, num_frames)

    return batch_groundtruth_objects, frame_metadata
