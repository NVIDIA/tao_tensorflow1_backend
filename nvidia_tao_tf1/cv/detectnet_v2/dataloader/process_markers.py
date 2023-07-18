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

"""Functions to process front / back markers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

REFERENCE_ANGLE = 0.0
INVALID_ORIENTATION = 0.0
FRONT_BACK_TOLERANCE = 5.0 * math.pi / 180.
SIDE_ONLY_TOLERANCE = 2.0 * math.pi / 180.


def _round_marker_tf(markers, tolerance=0.01):
    """Round markers to account for potential labeling errors.

    Args:
        markers (tf.Tensor): Either front or back marker values.
        tolerance (float): The tolerance within which we start rounding values.

    Returns:
        rounded_markers (tf.Tensor): <markers> that have been rounded.
    """
    # First, round values close to 0.0.
    rounded_markers = \
        tf.where(tf.logical_and(tf.less(markers, tolerance),
                                tf.greater_equal(markers, 0.0)),
                 tf.zeros_like(markers),  # If condition is True.
                 markers)
    # Then, round values close to 1.0.
    rounded_markers = \
        tf.where(tf.greater(rounded_markers, 1.0 - tolerance),
                 tf.ones_like(rounded_markers),
                 rounded_markers)

    return rounded_markers


def _minus_pi_plus_pi(orientations):
    """Puts orientations values in [-pi; pi[ range.

    Args:
        orientations (tf.Tensor): Contains values for orientation in radians. Shape is (N,).

    Returns:
        new_orientations (tf.Tensor): Same values as <orientations> but in [-pi; pi[ range.
    """
    new_orientations = tf.mod(orientations, 2. * math.pi)
    new_orientations = \
        tf.where(new_orientations > math.pi,
                 new_orientations - 2. * math.pi,
                 new_orientations)
    return new_orientations


def map_markers_to_orientations(front_markers,
                                back_markers,
                                invalid_orientation=INVALID_ORIENTATION,
                                ref_angle=REFERENCE_ANGLE,
                                tolerance=0.01,
                                clockwise=False):
    """Map front / back markers to orientation values.

    An angle of 0.0 corresponds to the scenario where an object is in the same direction as the ego
    camera, its back facing towards the camera. Outputs radian values in the ]-pi; pi] range.

    Args:
        front_markers (tf.Tensor): Denotes the front marker of target objects. Shape is (N,),
            where N is the number of targets in a frame.
        back_markers (tf.Tensor): Likewise, but for the back marker.
        invalid_orientation (float): Value to populate bogus entries correpsonding to bogus
            (<front_markers, back_markers>) combos.
        ref_angle (float): Reference angle corresponding to the scenario where a vehicle is right
            in front of the camera with its back facing towards the camera. For legacy reasons,
            the default is -pi/2.
        clockwise (bool): Whether to count clockwise angles as positive values. False would
            correspond to trigonometric convention.

    Returns:
        orientations (tf.Tensor): Shape (N,) tensor containing the angle corresponding to
            (<front_markers>, <back_markers>). Values are radians.

    Raises:
        ValueError: If parameters are outside accepted ranges.
    """
    if not (0.0 < tolerance < 1.0):
        raise ValueError("map_markers_to_orientations accepts a tolerance in ]0.; 1.[ range only.")
    if not (-math.pi <= ref_angle < math.pi):
        raise ValueError("map_markers_to_orientations accepts a ref_angle in [-pi; pi[ range only.")

    # First, round the markers.
    rounded_front_markers = _round_marker_tf(front_markers)
    rounded_back_markers = _round_marker_tf(back_markers)

    ones = tf.ones_like(rounded_front_markers)  # Used for constants and what not.
    orientations = tf.zeros_like(front_markers)
    # Back only.
    is_back_only = tf.logical_and(
        tf.equal(rounded_front_markers, -1.0),
        tf.logical_or(tf.equal(rounded_back_markers, 0.0),
                      tf.equal(rounded_back_markers, 1.0)))
    orientations = \
        tf.where(is_back_only,
                 tf.zeros_like(rounded_front_markers),
                 orientations)
    # Front only.
    is_front_only = tf.logical_and(
        tf.equal(rounded_back_markers, -1.0),
        tf.logical_or(tf.equal(rounded_front_markers, 0.0),
                      tf.equal(rounded_front_markers, 1.0)))
    orientations = \
        tf.where(is_front_only, math.pi * ones, orientations)
    # Front and right.
    is_front_and_right = \
        tf.logical_and(tf.logical_and(tf.greater(rounded_front_markers, 0.0),
                                      tf.less(rounded_front_markers, 1.0)),
                       tf.equal(rounded_back_markers, 0.0))
    orientations = \
        tf.where(is_front_and_right,
                 -(math.pi / 2.0) * (2.0 * ones - rounded_front_markers),
                 orientations)
    # Front and left.
    is_front_and_left = \
        tf.logical_and(tf.logical_and(tf.greater(rounded_front_markers, 0.0),
                                      tf.less(rounded_front_markers, 1.0)),
                       tf.equal(rounded_back_markers, 1.0))
    orientations = \
        tf.where(is_front_and_left,
                 (math.pi / 2.0) * (ones + rounded_front_markers),
                 orientations)
    # Back + right or left.
    is_back_and_side = \
        tf.logical_and(tf.logical_or(tf.equal(rounded_front_markers, 0.0),  # Left side.
                                     tf.equal(rounded_front_markers, 1.0)),  # Right side.
                       tf.logical_and(tf.greater(rounded_back_markers, 0.0),
                                      tf.less(rounded_back_markers, 1.0)))
    orientations = \
        tf.where(is_back_and_side,
                 (math.pi / 2.0) * (rounded_back_markers - rounded_front_markers),
                 orientations)
    # Finally, only one of the sides is visible (when either (0.0, 1.0) or (1.0, 0.0)).
    is_side_only = tf.logical_or(
        tf.logical_and(tf.equal(rounded_front_markers, 0.0), tf.equal(rounded_back_markers, 1.0)),
        tf.logical_and(tf.equal(rounded_front_markers, 1.0), tf.equal(rounded_back_markers, 0.0)))
    orientations = \
        tf.where(is_side_only,
                 (math.pi / 2.0) * (rounded_back_markers - rounded_front_markers),
                 orientations)

    # Shift and scale.
    if clockwise:
        orientations = -orientations
    orientations = orientations + ref_angle
    # Keep things in [-pi; pi[ range.
    orientations = _minus_pi_plus_pi(orientations)

    # Finally, if none of the cases had hit, set the entries to <invalid_orientation>.
    all_scenarios = tf.stack([is_back_only,
                              is_front_only,
                              is_front_and_right,
                              is_front_and_left,
                              is_back_and_side,
                              is_side_only])
    is_any_scenario = tf.reduce_any(all_scenarios, axis=0)
    orientations = tf.where(is_any_scenario,
                            orientations,  # Keep as is.
                            invalid_orientation * ones)

    return orientations


def augment_orientation_labels(orientation_labels, stm, ref_angle=REFERENCE_ANGLE):
    """Augment orientation labels.

    Why is the below check enough? For Gridbox, all STMs start out as a 3x3 identity matrices M.
    In determining the final STM, input STMS are right multiplied sequentially with a
    flip LR STM, and a combination of translation/zoom STMs which use the same underlying
    representation. A quick matrix multiply will show you that applying both a translate and a
    zoom STM is pretty much the same as applying one such STM with different parameters.
    Furthermore, given that the parameters passed to get the translate and scale STMs are always
    positive, the end result R of multiplying the initial STM M by the flip LR STM x
    translate/zoom STM shows that R[0, 0] is positive if and only if no flip LR STM was applied.

    NOTE: If rotations are introduced, this reasoning is no longer sufficient.

    Args:
        orientation_labels (tf.Tensor): Contains the orientation values for the targets in a single
            frame.
        stm (tf.Tensor): 3x3 spatial transformation matrix.
        ref_angle (float): Reference angle corresponding to the scenario where a vehicle is right
            in front of the camera with its back facing towards the camera. For legacy reasons,
            the default is -pi/2.

    Returns:
        augmented_orientation_labels (tf.Tensor): Contains the orientation values with the spatial
            transformations encapsulated by <stm> applied to them.
    Raises:
        ValueError: If ref_angle is outside accepted range.
    """
    if not (-math.pi <= ref_angle < math.pi):
        raise ValueError("augment_orientation_labels accepts a ref_angle in [-pi; pi[ range only.")

    # Define the callables for tf.cond.
    def no_flip(): return orientation_labels

    def flip(): return _minus_pi_plus_pi(2. * ref_angle - orientation_labels)

    with tf.control_dependencies([tf.assert_equal(stm[0, 1], 0.),
                                  tf.assert_equal(stm[1, 0], 0.)]):
        augmentated_orientation_labels = \
            tf.cond(stm[0, 0] < 0.0, flip, no_flip)

    return augmentated_orientation_labels


def _round_marker(marker, epsilon=0.05):
    """Helper function to round a marker value to either 0.0 or 1.0.

    Args:
        marker (float): Marker value. Expected to be in [0.0, 1.0] range.
        epsilon (float): Value within which to round.

    Returns:
        rounded_marker (float): <marker> rounded to either 0.0 or 1.0 if it is within epsilon of
            one or the other.
    """
    rounded_marker = marker
    if abs(marker) < epsilon:
        rounded_marker = 0.0
    elif abs(marker - 1.0) < epsilon:
        rounded_marker = 1.0

    return rounded_marker


def map_orientation_to_markers(
        orientation,
        ref_angle=REFERENCE_ANGLE,
        clockwise=False,
        front_back_tolerance=FRONT_BACK_TOLERANCE,
        side_only_tolerance=SIDE_ONLY_TOLERANCE):
    """Map orientation value to (front, back) marker values.

    Args:
        orientation (float): Orientation value in radians. Values are expected to be in [-pi; pi[.
        ref_angle (float): Reference angle corresponding to the scenario where a vehicle is right
            in front of the camera with its back facing towards the camera. For legacy reasons,
            the default is -pi/2.
        clockwise (bool): Whether to count clockwise angles as positive values. False would
            correspond to trigonometric convention.
        front_back_tolerance (float): Radian tolerance within which we consider <orientation> to be
            equal to that of a front- / back-only scenarios.
        side_only_tolerance (float): Likewise, but for either of the side-only scenarios.

    Returns:
        front (float): Corresponding front marker value.
        back (float): Idem, but for back marker value.

    Raises:
        ValueError: If ref_angle is outside accepted range.
    """
    if not (-math.pi <= ref_angle < math.pi):
        raise ValueError("augment_orientation_labels accepts a ref_angle in [-pi; pi[ range only.")
    # Adjust orientation coordinate system if need be.
    _orientation = orientation - ref_angle
    if clockwise:
        _orientation *= -1.
    # Put in [-pi, pi[ range.
    _orientation = _orientation % (2. * math.pi)
    _orientation = _orientation - 2. * math.pi if _orientation > math.pi else _orientation

    front = 0.
    back = 0.
    radian_factor = 2. / math.pi
    # For the following scenarios, we allow a certain tolerance on the orientation value:
    # - front or back only: if within <front_back_tolerance> of the exact value.
    # - side only: if within <side_only_tolerance> of the exact value.
    # As such, their corresponding checks will appear first in the following if / elif clause.
    if abs(_orientation) < front_back_tolerance:
        # Back only.
        front = -1.0
        back = 0.0
    elif abs(_orientation - math.pi) < front_back_tolerance or \
            abs(_orientation + math.pi) < front_back_tolerance:
        # Front only.
        front = 0.0
        back = -1.0
    elif abs(_orientation - math.pi / 2.0) < side_only_tolerance:
        # Left only.
        front = 0.0
        back = 1.0
    elif abs(_orientation + math.pi / 2.0) < side_only_tolerance:
        # Right only.
        front = 1.0
        back = 0.0
    elif (-math.pi / 2. < _orientation <= 0.0):
        # ]-pi/2; 0] - back + right.
        front = 1.0
        back = radian_factor * _orientation + 1.
    elif (-math.pi < _orientation <= -math.pi / 2.):
        # ]-pi; -pi/2] - front + right.
        front = radian_factor * _orientation + 2.
        back = 0.
    elif (0. < _orientation <= math.pi / 2.):
        # ]0; pi/2] - back + left.
        front = 0.
        back = radian_factor * _orientation
    elif (math.pi / 2. < _orientation <= math.pi):
        # ]pi/2; pi]. - front + left.
        front = radian_factor * _orientation - 1
        back = 1.

    # Additional rounding. This is to be able to hard classify certain examples as side only, etc.
    front = _round_marker(front)
    back = _round_marker(back)

    return front, back
