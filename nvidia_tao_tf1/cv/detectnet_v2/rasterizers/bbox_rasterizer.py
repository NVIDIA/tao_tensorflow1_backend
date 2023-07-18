# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Bbox rasterizer class that translates labels into ground truth tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import range
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import Bbox2DLabel
import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import UNKNOWN_CLASS
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import filter_labels


class BboxRasterizerInput(object):
    """Encapsulate some of the lower level details needed by BboxRasterizer from the user."""

    __slots__ = ["num_bboxes", "bbox_class_ids", "bbox_matrices", "bbox_coverage_radii",
                 "bbox_flags", "bbox_sort_values", "gradient_info"]

    def __init__(self,
                 num_bboxes,
                 bbox_class_ids,
                 bbox_matrices,
                 bbox_coverage_radii,
                 bbox_flags,
                 bbox_sort_values,
                 gradient_info):
        """Constructor.

        Args:
            num_bboxes (tf.Tensor): 0-D Tensor with the number of bboxes in this frame.
            bbox_class_ids (tf.Tensor): 1-D int32 Tensor indicating of which class each bbox is.
            bbox_matrices (tf.Tensor): 3-D float32 Tensor of shape (N, 3, 3) where N is the number
                of bboxes in this frame. Each element [i, :, :] is a row major matrix specifying the
                shape of the corresponding bbox.
            bbox_coverage_radii (tf.Tensor): 2-D float32 Tensor of shape (N, 2). Each element [i, :]
                contains the radii (along each dimension x and y) of the coverage region to be drawn
                for the corresponding bbox.
            bbox_flags (tf.Tensor): 1-D uint8 tensor. Each element indicates how the corresponding
                bbox's coverage region should be filled. Hardcoded to 'DRAW_MODE_ELLIPSE'.
            gradient_info (dict): Contains output space coordinates, inverse bbox area, and various
                other fields needed to calculate the objective-specific target gradients.
        """
        self.num_bboxes = num_bboxes
        self.bbox_class_ids = bbox_class_ids
        self.bbox_matrices = bbox_matrices
        self.bbox_coverage_radii = bbox_coverage_radii
        self.bbox_flags = bbox_flags
        self.bbox_sort_values = bbox_sort_values
        self.gradient_info = gradient_info


class BboxRasterizer(object):
    """Takes care of rasterizing labels into ground truth tensors for DetectNet V2 detection."""

    def __init__(self, input_width, input_height, output_width, output_height,
                 target_class_names, bbox_rasterizer_config, target_class_mapping,
                 output_type=None):
        """Constructor.

        Args:
            input_width/height (int): Input images' width / height in pixel space.
            output_width/height (int): Output rasters' width / height.
            target_class_names (list of str): List of target class names for which to generate
                rasters.
            bbox_rasterizer_config (BboxRasterizerConfig): Maps from target class names to
                BboxRasterizerConfig.TargetClassConfig.

        Raises:
            AssertionError: If certain target classes do not have corresponding parameters.
        """
        if not target_class_mapping:
            raise ValueError("BboxRasterizer expected a valid class mapping, instead got: {}".
                             format(target_class_mapping))
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height
        self.target_class_names = target_class_names
        self.bbox_rasterizer_config = bbox_rasterizer_config
        self.output_type = output_type
        self.deadzone_radius = self.bbox_rasterizer_config.deadzone_radius

        self._target_class_lookup = nvidia_tao_tf1.core.processors.LookupTable(
            keys=list(target_class_mapping.keys()),
            values=list(target_class_mapping.values()),
            default_value=tf.constant(UNKNOWN_CLASS)
        )
        # Check that each target class has corresponding rasterization parameters.
        for target_class_name in self.target_class_names:
            assert target_class_name in self.bbox_rasterizer_config
        self._target_class_indices = list(range(len(self.target_class_names)))

        # Get strides.
        self._scale_x = self.output_width / self.input_width
        self._scale_y = self.output_height / self.input_height

        # Get lookup tables for rasterization parameters.
        self._cov_center_x, self._cov_center_y, self._cov_radius_x, self._cov_radius_y, \
            self._bbox_min_radius = self._construct_lookup_tables()

        self._rasterizer = nvidia_tao_tf1.core.processors.BboxRasterizer()

    def _construct_lookup_tables(self):
        """Construct LUTs for mapping class names into ground truth parameters.

        Returns:
            cov_center_x/y (list of float): Follows the indexing of self.target_class_names. Each
                element corresponds to the x / y coordinate of where the center of the coverage
                region should be drawn, relative to ecah bounding box (e.g. midpoint is 0.5).
            cov_radius_x/y (list of float): Follows the indexing of self.target_class_names. Each
                element corresponds to the x / y extent of the coverage region, relative to the
                bbox dimensions (e.g. full bbox dimension is 1.0).
            bbox_min_radius (list of float): Follows the indexing of self.target_class_names. Each
                element corresponds to the minimum radius each coverage region should have.
        """
        cov_center_x = []
        cov_center_y = []
        cov_radius_x = []
        cov_radius_y = []
        bbox_min_radius = []
        # Go in order of self.target_class_names.
        for target_class_name in self.target_class_names:
            target_class_config = self.bbox_rasterizer_config[target_class_name]
            # Find a matching class from bbox_rasterizer_spec and append the values into lists.
            cov_center_x.append(target_class_config.cov_center_x)
            cov_center_y.append(target_class_config.cov_center_y)
            cov_radius_x.append(target_class_config.cov_radius_x)
            cov_radius_y.append(target_class_config.cov_radius_y)
            bbox_min_radius.append(target_class_config.bbox_min_radius)

        return cov_center_x, cov_center_y, cov_radius_x, cov_radius_y, bbox_min_radius

    def _lookup(self, values):
        """Create a lookup function for rasterization parameters.

        Args:
            values (list): Contains arbitrary elements as constructed by e.g.
                self._construct_lookup_tables.

        Returns:
            (nvidia_tao_tf1.core.processors.LookupTable) Callable with target class name (str) that returns
                the corresponding entry in <values>.
        """
        return nvidia_tao_tf1.core.processors.LookupTable(keys=self.target_class_names, values=values,
                                              default_value=-1)

    @staticmethod
    def bbox_from_rumpy_params(
            xmin, ymin, xmax, ymax,
            cov_center_x, cov_center_y, cov_radius_x, cov_radius_y, bbox_min_radius,
            deadzone_radius):
        """Compute bbox matrix and coverage radii based on input coords and Rumpy style parameters.

        Args:
            The first 4 arguments are all in the model output space.

            xmin (1-D tf.Tensor of float): Contains the left-most coordinates of bboxes.
            ymin (1-D tf.Tensor of float): Contains the top-most coordinates of bboxes.
            xmax / ymax (1-D tf.Tensor of float): Same but right- and bottom-most coordinates.
            cov_center_x (1-D tf.Tensor of float): Contains the x-coordinates of the centers of the
                coverage regions to be drawn for bboxes. Same indexing as e.g. xmin.
            cov_center_y (1-D tf.Tensor of float): Likewise, but for the y-axis.
            cov_radius_x (1-D tf.Tensor of float): Contains the radii along the x-axis of the
                coverage regions to be drawn for bboxes. Same indexing as e.g. xmin.
            cov_radius_y (1-D tf.Tensor of float): Likewise, but for the y-axis.
            bbox_min_radius (1-D tf.Tensor of float): Contains the minimum radii for the coverage
                regions to be drawn for bboxes. Same indexing as e.g. xmin.
            deadzone_radius (float): Radius of the deadzone region to be drawn between bboxes that
                have overlapping coverage regions.

        Returns:
            mat (3-D tf.Tensor of float): A matrix that maps from ground truth image space to the
                rasterization space, where transformed coordinates that fall within [-1.0, 1.0]
                are inside the deadzone. The shape of this tensor is (N, 3, 3) where N is the
                number of elements in <xmin>.
            cov_radius (2-D tf.Tensor of float): A (N, 2) tensor whose values contains the ratios
                of coverage to deadzone radii.
            inv_bbox_area (1-D tf.Tensor of float): Contains the values of the inverse bbox area,
                with the indexing corresponding to that of for instance <xmin>.
        """
        # Center of the coverage region in gt space
        # TODO is cov_center always [0.5, 0.5]?
        cx = xmin + cov_center_x * (xmax - xmin)
        cy = ymin + cov_center_y * (ymax - ymin)

        # Ellipse's semi-diameters (i.e. semi-major and semi-minor axes)
        # Picking the distance to the closest edge of the bbox as the radius so the generated
        # ellipse never spills outside of the bbox, unless possibly when too small.
        # Note: this is in abs gt-pixel coordinate space.
        sx = tf.where(tf.less(cov_center_x, 0.5), cx - xmin, xmax - cx)
        sy = tf.where(tf.less(cov_center_y, 0.5), cy - ymin, ymax - cy)

        # Compute coverage radii as fractions of bbox radii
        csx = cov_radius_x * sx
        csy = cov_radius_y * sy

        # Constrain absolute minimum size to avoid numerical problems below. Tenth of a pixel
        # should be small enough to allow almost non-visible bboxes if so desired, while large
        # enough to avoid problems. Note that this is just a safety measure: bbox_min_radius
        # below provides user controlled clamping (but can't guard against zero-sized bboxes),
        # and dataset converters should have removed degeneracies (but augmentation might
        # produce small bboxes).
        csx = tf.maximum(csx, 0.1)
        csy = tf.maximum(csy, 0.1)

        # Constrain X dimension, keeping aspect ratio
        rx = tf.maximum(csx, bbox_min_radius)
        ry = tf.where(tf.less(csx, bbox_min_radius), bbox_min_radius * csy / csx, csy)
        csx = rx
        csy = ry

        # Constrain Y dimension, keeping aspect ratio
        rx = tf.where(tf.less(csy, bbox_min_radius), bbox_min_radius * csx / csy, csx)
        ry = tf.maximum(csy, bbox_min_radius)
        csx = rx
        csy = ry

        # Compute deadzone radii by interpolating between coverage zone and original bbox size
        dsx = (1.0 - deadzone_radius) * csx + deadzone_radius * sx
        dsy = (1.0 - deadzone_radius) * csy + deadzone_radius * sy

        # Constrain deadzone to be larger than coverage zone
        dsx = tf.maximum(dsx, csx)
        dsy = tf.maximum(dsy, csy)

        # Construct a matrix that maps from ground truth image space to rasterization space
        # where transformed coordinates that are within [-1,1] range are inside deadzone
        oodsx = 1. / dsx
        oodsy = 1. / dsy
        zero = tf.zeros(shape=[tf.size(xmin)])
        one = tf.ones(shape=[tf.size(xmin)])
        mat = [[oodsx,     zero,      zero],
               [zero,      oodsy,     zero],
               [-cx*oodsx, -cy*oodsy, one]]

        # Convert from matrix of arrays to array of matrices.
        mat = tf.transpose(mat, (2, 0, 1))

        # Compute the ratio of coverage and deadzone radii
        cov_radius = tf.transpose([csx * oodsx, csy * oodsy])

        # Compute coverage area based normalization factor to be used for cost function weighting
        # Clamp to ensure the value is always <= 1.0
        inv_bbox_area = 1. / tf.maximum(csx * csy * 4., 1.)

        return mat, cov_radius, inv_bbox_area

    def _prepare_labels(self, labels):
        """Prepare labels by keeping only those with mapped classes, and then sorting them.

        Filter out source classes that are not mapped to any target class.

        Args:
            labels (variable type):
                * If a dict, then it contains various label features for a single frame. Maps from
                feature name (str) to tf.Tensor. This corresponds to the old (DefaultDataloader)
                path.
                * Otherwise, expects a Bbox2DLabel with all the features for a minibatch.

        Returns:
            output_labels (dict of tf.Tensors): Contains the same label features as ``labels``,
                but with unmapped classes filtered out.
            class_ids (tf.Tensor): 1-D Tensor containing integer indices corresponding to each
                label value's class in ``output_labels``.
            num_bboxes (tf.Tensor): 1-D Tensor containing the number of bounding boxes per frame.
        """
        output_labels = dict()
        if isinstance(labels, dict):
            # Filter out unmapped labels.
            mapped_labels = dict()
            mapped_labels.update(labels)
            target_classes = self._target_class_lookup(labels['target/object_class'])
            valid_indices = tf.not_equal(target_classes, UNKNOWN_CLASS)
            mapped_labels['target/object_class'] = target_classes
            mapped_labels = filter_labels(mapped_labels, valid_indices)

            object_classes = mapped_labels['target/object_class']
            class_ids = self._lookup(self._target_class_indices)(object_classes)

            num_bboxes = tf.size(class_ids)
            for feature_name, feature_tensor in six.iteritems(mapped_labels):
                if feature_name.startswith('target/'):
                    output_labels[feature_name] = feature_tensor
                elif feature_name.startswith('frame/'):
                    output_labels[feature_name] = mapped_labels[feature_name]
        elif isinstance(labels, Bbox2DLabel):
            # TODO(@williamz): This feature needs to be ported into ObstacleNet version once
            # temporal models become a thing there.
            if self.output_type == 'last':
                # Filter out labels belonging to other than the last frame.
                def _filter_labels(labels):
                    """Helper function to filter labels other than the last frame."""
                    valid_indices = tf.equal(labels.object_class.indices[:, 1],
                                             labels.object_class.dense_shape[1]-1)
                    filtered_labels = labels.filter(valid_indices)

                    return filtered_labels

                labels = tf.cond(labels.object_class.dense_shape[1] > 1,
                                 lambda: _filter_labels(labels),
                                 lambda: labels)

            # Filter out unmapped labels.
            source_classes = labels.object_class
            mapped_classes = tf.SparseTensor(
                values=self._target_class_lookup(source_classes.values),
                indices=source_classes.indices,
                dense_shape=source_classes.dense_shape)
            mapped_labels = labels._replace(object_class=mapped_classes)
            valid_indices = tf.not_equal(mapped_classes.values, UNKNOWN_CLASS)

            filtered_labels = mapped_labels.filter(valid_indices)

            valid_classes = filtered_labels.object_class.values
            valid_coords = tf.reshape(filtered_labels.vertices.coordinates.values, [-1, 4])
            valid_sparse_indices = filtered_labels.object_class.indices

            class_ids = self._lookup(self._target_class_indices)(valid_classes)
            if self.output_type == 'all':
                num_frames = tf.cast(source_classes.dense_shape[0] *
                                     source_classes.dense_shape[1], dtype=tf.int32)
                frame_indices = tf.cast(
                    valid_sparse_indices[:, 0] *
                    source_classes.dense_shape[1] + valid_sparse_indices[:, 1], dtype=tf.int32)
            elif self.output_type in [None, 'last']:
                num_frames = tf.cast(source_classes.dense_shape[0], dtype=tf.int32)
                frame_indices = tf.cast(valid_sparse_indices[:, 0], dtype=tf.int32)
            else:
                raise ValueError("Unsupported output_type: {}".format(self.output_type))

            output_labels['target/bbox_coordinates'] = valid_coords
            for feature_name in filtered_labels.TARGET_FEATURES:
                feature_tensor = getattr(filtered_labels, feature_name)
                if feature_name == 'vertices' or \
                        not isinstance(feature_tensor, tf.SparseTensor):
                    continue
                output_labels['target/' + feature_name] = feature_tensor.values

            # Calculate number of bboxes per image.
            # NOTE: the minlength arg is required because the above filtering mechanism may have
            # led to the last frames in the batch being completely void of labels.
            num_bboxes = tf.bincount(frame_indices, minlength=num_frames)
        else:
            raise ValueError("Unsupported variable type for labels ({}).".format(type(labels)))

        return output_labels, class_ids, num_bboxes

    def get_target_gradient_info(self, frame_labels):
        """Translate labels.

        Computes the information necessary to calculate target gradients.

        Args:
            frame_labels (dict of tf.Tensors): Contains various label features for a single frame.

        Returns:
            bbox_rasterizer_input (BboxRasterizerInput): Encapsulate all the lower level arguments
                needed by the call to the SDK.
        """
        filtered_labels, class_ids, num_bboxes = self._prepare_labels(frame_labels)

        object_classes = filtered_labels['target/object_class']
        coordinates = filtered_labels['target/bbox_coordinates']

        # Find appropriate scaling factors to go from input image pixel space to network output
        #  / 'rasterization' space, i.e. divide by stride.
        xmin = coordinates[:, 0] * self._scale_x
        ymin = coordinates[:, 1] * self._scale_y
        xmax = coordinates[:, 2] * self._scale_x
        ymax = coordinates[:, 3] * self._scale_y

        # Compute bbox matrices based on bbox coordinates.
        matrices, coverage_radii, inv_bbox_area = \
            self.bbox_from_rumpy_params(
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                cov_center_x=self._lookup(self._cov_center_x)(object_classes),
                cov_center_y=self._lookup(self._cov_center_y)(object_classes),
                cov_radius_x=self._lookup(self._cov_radius_x)(object_classes),
                cov_radius_y=self._lookup(self._cov_radius_y)(object_classes),
                bbox_min_radius=self._lookup(self._bbox_min_radius)(object_classes),
                deadzone_radius=self.deadzone_radius)

        flags = tf.fill([tf.size(xmin)],
                        tf.cast(nvidia_tao_tf1.core.processors.BboxRasterizer.DRAW_MODE_ELLIPSE,
                                tf.uint8))

        # Sort bboxes by ascending ymax to approximate depth sorting.
        sort_value = ymax

        gradient_info = dict()
        gradient_info.update(filtered_labels)
        # Make a label info dictionary for use in gradient construction
        gradient_info['target/inv_bbox_area'] = inv_bbox_area
        # Update label info with the coordinates to be used for "gradient" calculation.
        gradient_info['target/output_space_coordinates'] = tf.stack([xmin, ymin, xmax, ymax])

        return BboxRasterizerInput(
            num_bboxes=num_bboxes,
            bbox_class_ids=class_ids,
            bbox_matrices=matrices,
            bbox_coverage_radii=coverage_radii,
            bbox_flags=flags,
            bbox_sort_values=sort_value,
            gradient_info=gradient_info)

    def rasterize_labels(self,
                         batch_bbox_rasterizer_input,
                         batch_gradients,
                         num_gradients,
                         gradient_flag):
        """Rasterize a batch of labels for a given Objective.

        Args:
            batch_bbox_rasterizer_input (list): Each element is a BboxRasterizerInput containing
                the information for a frame.
            batch_gradients (list): Each element is a 3-D tf.Tensor of type float32. Each tensor is
                of shape (N, G, 3) where N is the number of bboxes in the corresponding frame, G
                the number of output channels the rasterized tensor will have for this objective.
            num_gradients (int): Number of gradients (output channels).
            gradient_flag: One of the draw modes under nvidia_tao_tf1.core.processors.BboxRasterizer.

        Returns:
            target_tensor (tf.Tensor): Rasterized ground truth tensor for one single objective.
                Shape is (N, C, G, H, W) where C is the number of target classes, and H and W are
                respectively the height and width in the model output space.
        """
        if isinstance(batch_bbox_rasterizer_input, list):
            bboxes_per_image = [item.num_bboxes for item in batch_bbox_rasterizer_input]
            # Concatenate the inputs that need it.
            bbox_class_ids = tf.concat(
                [item.bbox_class_ids for item in batch_bbox_rasterizer_input], axis=0)
            bbox_matrices = tf.concat(
                [item.bbox_matrices for item in batch_bbox_rasterizer_input], axis=0)
            bbox_coverage_radii = tf.concat(
                [item.bbox_coverage_radii for item in batch_bbox_rasterizer_input], axis=0)
            bbox_flags = tf.concat(
                [item.bbox_flags for item in batch_bbox_rasterizer_input], axis=0)
            bbox_sort_values = tf.concat(
                [item.bbox_sort_values for item in batch_bbox_rasterizer_input], axis=0)

            bbox_gradients = tf.concat(batch_gradients, axis=0)
            num_images = len(batch_bbox_rasterizer_input)
        else:
            bboxes_per_image = batch_bbox_rasterizer_input.num_bboxes
            bbox_class_ids = batch_bbox_rasterizer_input.bbox_class_ids
            bbox_matrices = batch_bbox_rasterizer_input.bbox_matrices
            bbox_gradients = batch_gradients
            bbox_coverage_radii = batch_bbox_rasterizer_input.bbox_coverage_radii
            bbox_flags = batch_bbox_rasterizer_input.bbox_flags
            bbox_sort_values = batch_bbox_rasterizer_input.bbox_sort_values
            num_images = tf.size(bboxes_per_image)

        num_target_classes = len(self.target_class_names)
        gradient_flags = [gradient_flag] * num_gradients
        target_tensor = \
            self._rasterizer(num_images=num_images,
                             num_classes=num_target_classes,
                             num_gradients=num_gradients,
                             image_height=self.output_height,
                             image_width=self.output_width,
                             bboxes_per_image=bboxes_per_image,
                             bbox_class_ids=bbox_class_ids,
                             bbox_matrices=bbox_matrices,
                             bbox_gradients=bbox_gradients,
                             bbox_coverage_radii=bbox_coverage_radii,
                             bbox_flags=bbox_flags,
                             bbox_sort_values=bbox_sort_values,
                             gradient_flags=gradient_flags)

        return target_tensor
