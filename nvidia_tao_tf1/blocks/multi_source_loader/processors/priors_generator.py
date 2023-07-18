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

"""Class for priors generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.polyline_clipper import (
    PolylineClipper,
)

COORDINATES_PER_POINT = 4
EDGES_PER_PATH = 2


class PriorsGenerator(object):
    """Class for priors generation.

    The functions in this class deal with programmatically generating the priors
    for path data, which is composed of arbitrary polygons specified by left and right edges (top
    and bottom edges are implied by the bottom and top coordinates of the left and right edges).

    The basic algorithm for generating priors proceeds with the following steps:
    (1) Find the center of the receptive fields for all neurons in the network layer at which you
        are generating priors.
    (2) Generate a set of npriors priors based on points and linear shapes in a normalized
        coordinate frame (centered at origin, spanning from -1 to 1).
        - There can only be 1 point prior per receptive field, located at the center of the
          receptive field.
        - There is only one linear shape: a horizontal line from -1 to 1.
        - Number of point and linear priors are specified in the experiment spec.
    (3) Copy and rotate the generated priors by a set of angles to produce a set of npriors priors.
    (4) Translate the priors to each receptive field center.
    (5) Scale the priors to match the aspect ratio of the image.
    (6) Verify that all points on the prior are within the image, and if not, clip and then inter-
        polate such that the priors are within the image borders but still have points_per_prior
        points.
    """

    def __init__(
        self,
        npoint_priors,
        nlinear_priors,
        points_per_prior,
        prior_threshold,
        feature_map_sizes,
        image_height,
        image_width,
    ):
        """
        Initialize priors related variables.

        Number of point and linear priors, points per prior, prior threshold,
        feature map sizes extracted from the model, and the image height and width.
        """
        if npoint_priors < 0:
            raise ValueError(
                "npoints_priors must be positive, it is {}.".format(npoint_priors)
            )
        self.npoint_priors = npoint_priors
        if nlinear_priors < 0:
            raise ValueError(
                "nlinear_priors must be positive, it is {}.".format(nlinear_priors)
            )
        self.nlinear_priors = nlinear_priors
        self.npriors = npoint_priors + nlinear_priors
        if self.npriors <= 0:
            raise ValueError("npriors must be > 0, it is {}.".format(self.npriors))
        if points_per_prior <= 0:
            raise ValueError(
                "points_per_prior must be positive, not {}.".format(points_per_prior)
            )
        self.points_per_prior = points_per_prior
        self.prior_threshold = prior_threshold
        self.path_width = 100.0
        self.normalized_linear_prior_length = 0.625
        self._polyline_clipper = PolylineClipper(self.points_per_prior)
        self.nall_priors = self._get_nall_priors(feature_map_sizes)
        self.priors = self._get_priors(feature_map_sizes, image_height, image_width)
        if self.nall_priors < 1:
            raise ValueError(
                "There must be at least one prior, instead {}.".format(self.nall_priors)
            )
        if self.priors is None:
            raise ValueError("There is not any prior set.")

    def _transform_prior(
        self, origin, xs, ys, angle, scale_x=1.0, scale_y=1.0, tx=0.0, ty=0.0
    ):
        """
        Rotate, translate and scale points clockwise by a given angle around a given origin.

        Args:
            origin (scalar, scalar): Point about which to rotate the points.
            xs, ys (1D tensor, 1D tensor): x and y coordinates to be rotated.
            angle (scalar): Angle in radians to rotate the points by.
            scale_x (scalar): Multiplier for x coordinates.
            scale_y (scalar): Multiplier for y coordinates.
            tx (scalar): Amount to translate the points in the x-dimension.
            ty (scalar): Amount to translate the points in the y-dimension.

        Returns:
            newx, newy (1D tensor, 1D tensor): Rotated, translated and scaled points.

        """
        ox, oy = tf.split(tf.cast(origin, tf.float32), [1, 1])
        cos_angle = tf.cos(tf.cast(angle, tf.float32))
        sin_angle = tf.sin(tf.cast(angle, tf.float32))

        # Apply 2D rotation matrix with translation.
        newx = ox + cos_angle * (xs - ox) - sin_angle * (ys - oy)
        newy = oy + sin_angle * (xs - ox) + cos_angle * (ys - oy)

        # Scale and translate the resulting points.
        newx = scale_x * newx + tx
        newy = scale_y * newy + ty

        return newx, newy

    def _get_prior_locations(
        self, receptive_field_x, receptive_field_y, image_height, image_width
    ):
        """
        Calculate the locations at which to place each set of priors.

        Args:
            receptive_field_x (int): Size of receptive field in the x-dimension (in pixels).
            receptive_field_y (int): Size of receptive field in the y-dimension (in pixels).
            image_height (int): Height of the image.
            image_width (int): Width of the image.

        Returns:
            prior_locations (matrix of floats): x, y locations of the center of the prior sets
                                                in the image (in pixels).

        """
        prior_locations_x = np.arange(
            (receptive_field_x / 2.0),
            image_width + 1 - (receptive_field_x / 2.0),
            receptive_field_x,
        )
        prior_locations_y = np.arange(
            (receptive_field_y / 2.0),
            image_height + 1 - (receptive_field_y / 2.0),
            receptive_field_y,
        )

        prior_locations = np.meshgrid(prior_locations_x, prior_locations_y)

        return prior_locations

    def _generate_point_priors(self, x, y, image_height, image_width):
        """
        Create the point priors.

        Priors are created by replicating the center point of the receptive field,
        tx and ty, points_per_priors times. This means that each path will be coded
        as an offset from the center of the receptive field.

        Args:
            x (float): Center of receptive field in x-dimension.
            y (float): Center of receptive field in y-dimension.
            image_height (int): Height of the image.
            image_width (int): Width of the image.

        Returns:
            priors (array of floats): Ordered as leftx, lefty, rightx, righty coordinates.

        """
        norm_x = x * (1.0 / image_width)
        norm_y = y * (1.0 / image_height)

        # Convert to tensors and expand number of points.
        priors = tf.tile(
            tf.cast([[norm_x, norm_y]], tf.float32),
            [EDGES_PER_PATH * self.points_per_prior, 1],
        )

        return priors

    def _generate_linear_priors(
        self, angles, scale_x, scale_y, tx, ty, image_height, image_width
    ):
        """
        Create the linear priors.

        Priors are created by generating a horizontal line
        with points_per_prior points then rotating, translating and scaling by the
        amounts specified.

        Args:
            angles (array of floats): Angles to rotate the lines by in radians.
            scale_x (scalar): Extent of priors in x-direction.
            scale_y (scalar): Extent of priors in y-direction.
            tx (scalar): Amount to translate the points in the x-dimension.
            ty (scalar): Amount to translate the points in the y-dimension.
            image_height (int): Height of the image.
            image_width (int): Width of the image.

        Returns:
            priors (array of floats): Ordered as leftx, lefty, rightx, righty coordinates.

        """
        # Generate the base edge priors, which will then be modified based on the spec parameters.
        # The default prior is a horizontal vector that points to the left and is centered at
        # the origin. The angles that this vector will be rotated by are all positive and less
        # than 180 degrees and thus rotate the vector clockwise about the origin. We use a
        # horizontal left pointing vector so that the ordering of the points is always from bottom
        # to top. That is, the vector always starts in the lower half-plane and ends in the upper
        # half plane.
        base_xs = tf.linspace(1.0, -1.0, self.points_per_prior)
        base_ys = tf.zeros(self.points_per_prior, dtype=tf.float32)
        base_widthx = tf.constant([0.0, 0.0], dtype=tf.float32)
        base_widthy = tf.constant([0.0, (self.path_width / 2.0)], dtype=tf.float32)

        for edge in ["Left", "Right"]:
            # Create one prior for each rotation angle.
            prior_xs = []
            prior_ys = []
            for angle in angles:
                assert 0.0 <= angle <= np.pi, (
                    "angle to rotate the linear prior by is %.2f,"
                    "which outside the range 0 to pi." % angle
                )

                # Rotate and scale the prior edges.
                imagex, imagey = self._transform_prior(
                    (0, 0), base_xs, base_ys, angle, scale_x, scale_y, tx, ty
                )
                widthx, widthy = self._transform_prior(
                    (0, 0), base_widthx, base_widthy, angle
                )

                # Convert from centerline priors to edge priors.
                sign = -1 if "Left" == edge else 1.0
                image_x = imagex + sign * widthx[-1]
                image_y = imagey + sign * widthy[-1]

                prior_xs.append(image_x / image_width)
                prior_ys.append(image_y / image_height)

            zipped_priors = tf.stack(
                [
                    tf.reshape(tf.stack(prior_xs), [-1]),
                    tf.reshape(tf.stack(prior_ys), [-1]),
                ],
                axis=1,
            )

            # Clip the priors to the image boundaries. Use image_height and image_width values of 1
            # as the priors are normalized above. Need to maintain the vertex number since all
            # priors are required to have the same number of vertices.
            vertices_per_polyline = tf.constant(
                self.points_per_prior, shape=[len(angles)], dtype=tf.int32
            )
            polygon_mask = tf.constant(
                [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], tf.float32
            )
            clipped_priors, clipped_polyline_index_map = self._polyline_clipper.clip(
                polylines=zipped_priors,
                vertices_per_polyline=vertices_per_polyline,
                class_ids_per_polyline=tf.zeros_like(vertices_per_polyline),
                attributes_per_polyline=tf.zeros_like(vertices_per_polyline),
                maintain_vertex_number=True,
                polygon_mask=polygon_mask,
            )[0:2]

            if "Left" == edge:
                left_clipped_priors = clipped_priors
                left_clipped_map = clipped_polyline_index_map
            else:
                right_clipped_priors = clipped_priors
                right_clipped_map = clipped_polyline_index_map

        # Keep only priors where both edges are present because clipping can remove one or
        # the other edge entirely.
        left_clipped_maps, right_clipped_maps = tf.meshgrid(
            left_clipped_map, right_clipped_map
        )
        valid_priors = tf.compat.v1.where(
            tf.equal(left_clipped_maps, right_clipped_maps)
        )
        right_valid_priors, left_valid_priors = tf.split(valid_priors, 2, axis=1)

        # Get the point indices associated with this prior.
        point_indices = tf.reshape(
            tf.cumsum(tf.ones([len(angles) * self.points_per_prior], tf.int32)) - 1,
            [len(angles), self.points_per_prior],
        )
        left_valid_point_indices = tf.gather(
            point_indices, tf.squeeze(left_valid_priors)
        )
        right_valid_point_indices = tf.gather(
            point_indices, tf.squeeze(right_valid_priors)
        )
        left_valid_prior_points = tf.reshape(
            tf.gather(left_clipped_priors, left_valid_point_indices), [-1, 2]
        )
        right_valid_prior_points = tf.reshape(
            tf.gather(right_clipped_priors, right_valid_point_indices), [-1, 2]
        )

        priors = tf.reshape(
            tf.transpose(
                a=tf.stack(
                    [left_valid_prior_points, right_valid_prior_points], axis=-1
                ),
                perm=[0, 2, 1],
            ),
            [-1, 2],
        )

        return priors

    def _get_nall_priors(self, feature_map_sizes):
        """Get the number of priors."""
        if len(feature_map_sizes) == 0:
            raise ValueError("Feature map sizes not yet set.")
        return np.sum([np.prod(fmaps) * self.npriors for fmaps in feature_map_sizes])

    def _get_priors(self, feature_map_sizes, image_height, image_width):
        """
        Generate path priors based on the image size and constraints.

        Returns:
            priors (array of floats): Ordered as leftx, lefty, rightx, righty coordinates.

        """
        priors = None

        # For each feature map, calculate the priors in image coordinates.
        for feature_map_size in feature_map_sizes:
            # Calculate the receptive field sizes in the original image for this layer.
            receptive_field_x = np.floor(image_width / feature_map_size[1])
            receptive_field_y = np.floor(image_height / feature_map_size[0])

            # Specify the scale ratios for the priors. Based on analysis of ground truth
            # data average widths and heights of paths in the image.
            scale_x = self.normalized_linear_prior_length * receptive_field_x
            scale_y = self.normalized_linear_prior_length * receptive_field_y

            # Enumerate the receptive field center locations.
            prior_locations = self._get_prior_locations(
                receptive_field_x,
                receptive_field_y,
                image_height=image_height,
                image_width=image_width,
            )

            # Calculate the parameters for the linear priors. The rotation angles should all
            # be positive so that the default linear prior vector is rotated clockwise.
            if self.nlinear_priors > 0:
                angle_increment = np.pi / (self.nlinear_priors + 1.0)
                linear_prior_rotation_angles = np.linspace(
                    angle_increment, np.pi - angle_increment, self.nlinear_priors
                )

            # Calculate the priors for each location.
            # TODO(blythe): Determine if can generate all base priors, copy, translate and
            # then clip, rather than doing this loop. Complicated because clipper doesn't
            # support the point priors.
            for tx, ty in zip(
                prior_locations[0].flatten(), prior_locations[1].flatten()
            ):

                if self.npoint_priors > 0:
                    point_priors = self._generate_point_priors(
                        tx, ty, image_height=image_height, image_width=image_width
                    )
                    for _ in range(self.npoint_priors):
                        priors = (
                            tf.concat([priors, point_priors], 0)
                            if priors is not None
                            else point_priors
                        )

                if self.nlinear_priors > 0:
                    linear_priors = self._generate_linear_priors(
                        linear_prior_rotation_angles,
                        scale_x,
                        scale_y,
                        tx,
                        ty,
                        image_height=image_height,
                        image_width=image_width,
                    )
                    priors = (
                        tf.concat([priors, linear_priors], 0)
                        if priors is not None
                        else linear_priors
                    )
        return priors
