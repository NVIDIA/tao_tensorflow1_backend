# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Loss mask rasterizer class that translates labels to rasterized tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import Bbox2DLabel
from nvidia_tao_tf1.core.processors import PolygonRasterizer

import six
import tensorflow as tf


class LossMaskRasterizer(object):
    """Handle the logic of translating labels to ground truth tensors.

    Much like the LossMaskFilter object, this class holds model-specific information in a
    'hierarchy'.
    It is for now comprised of two levels: [target_class_name][objective_name], although in the
    future it is quite likely an additional [head_name] level will be pre-pended to it.
    """

    def __init__(self, input_width, input_height, output_width, output_height):
        """Constructor.

        Args:
            input_width/height (int): Model input dimensions.
            output_width/height (int): Model output dimensions.
        """
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height
        # Get the rasterizer from the SDK.
        self._rasterizer = \
            PolygonRasterizer(width=output_width, height=output_height,
                              one_hot=False, data_format='channels_first')
        # Setup some private attributes to use for label-to-rasterizer-input translation.
        self._scale_x = self.output_width / self.input_width
        self._scale_y = self.output_height / self.input_height

    def translate_frame_labels_bbox_2d_label(self, ground_truth_labels):
        """Translate a frame's ground truth labels to the inputs necessary for the rasterizer.

        Args:
            ground_truth_labels (Bbox2DLabel): Filtered labels, which only incorporates
                bboxes matching filters for all frames in a batch.

        Returns:
            polygon_vertices (tf.Tensor of float): 2-D tensor of shape (N, 2) where entry [n - 1, 0]
                corresponds to the n-th vertex's x coordinate, and [n - 1, 1] to its y coordinate.
            vertex_counts_per_polygon (tf.Tensor of int): 1-D tensor where each entry holds the
                number of vertices for a single polygon. As an example, if entries at indices 0 and
                1 are 3 and 4, that means the first 3 entries in <polygon_vertices> describe one
                polygon, and the next 4 entries in <polygon_vertices> describe another.
                However, in this special case, ALL polygons are bboxes and hence have 4 vertices.
            class_ids_per_polygon (tf.Tensor of int): 1-D tensor that has the same length as
                <vertex_counts_per_polygon>. Contains the class ID of each corresponding polygon.
                In this special case, they are assumed to all belong to the same class.
            polygons_per_image (tf.Tensor of int): 1-D tensor that describes how many polygons
                there are in this image.
        """
        source_classes = ground_truth_labels.object_class
        frame_indices = tf.cast(source_classes.indices[:, 0], dtype=tf.int32)
        num_frames = tf.cast(source_classes.dense_shape[0], dtype=tf.int32)
        # Step 1: we separate coords into x1,y1,x2,y2.
        coords = tf.reshape(ground_truth_labels.vertices.coordinates.values, [-1, 4])
        x1 = tf.reshape(coords[:, 0] * self._scale_x, [-1])
        y1 = tf.reshape(coords[:, 1] * self._scale_y, [-1])
        x2 = tf.reshape(coords[:, 2] * self._scale_x, [-1])
        y2 = tf.reshape(coords[:, 3] * self._scale_y, [-1])
        # Step 2: compose the vertices of polygon.
        coordinates_x = tf.stack([x1, x2, x2, x1], axis=0)
        coordinates_x = tf.reshape(tf.transpose(coordinates_x, perm=[1, 0]), [-1])
        coordinates_y = tf.stack([y1, y1, y2, y2], axis=0)
        coordinates_y = tf.reshape(tf.transpose(coordinates_y, perm=[1, 0]), [-1])
        polygon_vertices = tf.stack([coordinates_x, coordinates_y], axis=1)
        # Step 3: compose vertex counts, and we assume each polygon has 4 vertices.
        vertex_counts_per_polygon = tf.cast(tf.ones_like(x1) * 4, dtype=tf.int32)
        # Step 4: Compose class ids, and they are all the same class.
        class_ids_per_polygon = tf.zeros_like(vertex_counts_per_polygon)
        # Step 5: polygons per image.
        polygons_per_image = tf.bincount(frame_indices, minlength=num_frames)

        return polygon_vertices, vertex_counts_per_polygon, \
            class_ids_per_polygon, polygons_per_image

    def translate_frame_labels_dict(self, frame_ground_truth_labels):
        """Translate a frame's ground truth labels to the inputs necessary for the rasterizer.

        Args:
            frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.

        Returns:
            polygon_vertices (tf.Tensor of float): 2-D tensor of shape (N, 2) where entry [n - 1, 0]
                corresponds to the n-th vertex's x coordinate, and [n - 1, 1] to its y coordinate.
            vertex_counts_per_polygon (tf.Tensor of int): 1-D tensor where each entry holds the
                number of vertices for a single polygon. As an example, if entries at indices 0 and
                1 are 3 and 4, that means the first 3 entries in <polygon_vertices> describe one
                polygon, and the next 4 entries in <polygon_vertices> describe another.
                However, in this special case, ALL polygons are bboxes and hence have 4 vertices.
            class_ids_per_polygon (tf.Tensor of int): 1-D tensor that has the same length as
                <vertex_counts_per_polygon>. Contains the class ID of each corresponding polygon.
                In this special case, they are assumed to all belong to the same class.
            polygons_per_image (tf.Tensor of int): 1-D tensor that describes how many polygons
                there are in this image.
        """
        # TODO(@williamz): again, some hardcoded BS that is likely to lead to some problems.
        # Get polygon coordinates.
        coordinates_x = frame_ground_truth_labels['target/coordinates/x'] * self._scale_x
        coordinates_y = frame_ground_truth_labels['target/coordinates/y'] * self._scale_y
        # Setup vertices as (x1, y1), (x2, y1), (x2, y2), (x1, y2).
        polygon_vertices = tf.stack([coordinates_x, coordinates_y], axis=1)

        # Intermediate step.
        coordinates_per_polygon = tf.bincount(tf.cast(
            frame_ground_truth_labels['target/coordinates/index'], dtype=tf.int32))

        # All the same class.
        class_ids_per_polygon = tf.zeros_like(coordinates_per_polygon)
        # reshape is needed here because scalars don't play along nicely with concat ops.
        polygons_per_image = tf.reshape(tf.size(coordinates_per_polygon), shape=(1,))

        return polygon_vertices, coordinates_per_polygon, class_ids_per_polygon, \
            polygons_per_image

    def rasterize_labels_bbox_2d_label(self, batch_labels, mask=None, mask_multiplier=1.0):
        """Setup the rasterized loss mask for a given set of ground truth labels.

        Args:
            batch_labels (Bbox2DLabel): Filtered labels, which only incorporates bboxes matching
                filters for all frames in a batch.
            mask (Tensor): Where nonzero, the mask_multiplier is ignored (mask multiplier is set
                to the background value, 1.0). Default None, the mask_multiplier is never ignored.
            mask_multiplier (float): Scalar value that will be assigned to each region in a set
                of ground truth labels. Default value of 1.0 means the output is all filled with
                ones, essentially meaning all regions of the network's output are treated equally.

        Returns:
            loss_mask (tf.Tensor): rasterized loss mask corresponding to the input labels.
        """
        vertices, vertex_counts, ids, polygons_per_image =  \
            self.translate_frame_labels_bbox_2d_label(batch_labels)

        polygon_raster = self._rasterizer(
            polygon_vertices=vertices,
            vertex_counts_per_polygon=vertex_counts,
            class_ids_per_polygon=ids,
            polygons_per_image=polygons_per_image
        )
        # Outside the input labels, the loss mask should have a value of 1.0 (i.e. the loss will
        #  be treated as usual in those cells).
        ones, zeros = tf.ones_like(polygon_raster), tf.zeros_like(polygon_raster)

        # If a mask exists, zero the polygon raster where the mask is nonzero.
        if mask is not None:
            objective_mask = tf.where(mask > 0., zeros, ones)
            polygon_raster *= objective_mask

        # Set all foreground values to the value of mask_multiplier.
        background = tf.where(polygon_raster > 0., zeros, ones)
        loss_mask = background + mask_multiplier * polygon_raster

        return loss_mask

    def rasterize_labels_dict(self, batch_labels, mask=None, mask_multiplier=1.0):
        """Setup the rasterized loss mask for a given set of ground truth labels.

        Args:
            batch_labels (list of dicts of Tensors): contains the labels for a batch of frames.
            mask (Tensor): Where nonzero, the mask_multiplier is ignored (mask multiplier is set
                to the background value, 1.0). Default None, the mask_multiplier is never ignored.
            mask_multiplier (float): Scalar value that will be assigned to each region in a set
                of ground truth labels. Default value of 1.0 means the output is all filled with
                ones, essentially meaning all regions of the network's output are treated equally.

        Returns:
            loss_mask (tf.Tensor): rasterized loss mask corresponding to the input labels.
        """
        batch_polygon_vertices = []
        batch_vertex_counts_per_polygon = []
        batch_class_ids_per_polygon = []
        batch_polygons_per_image = []
        for frame_labels in batch_labels:
            # Get the rasterizer inputs for the new frame.
            _polygon_vertices, _vertex_counts_per_polygon, _class_ids_per_polygon, \
                _polygons_per_image = self.translate_frame_labels_dict(frame_labels)
            # Update the batch's inputs.
            batch_polygon_vertices.append(_polygon_vertices)
            batch_vertex_counts_per_polygon.append(_vertex_counts_per_polygon)
            batch_class_ids_per_polygon.append(_class_ids_per_polygon)
            batch_polygons_per_image.append(_polygons_per_image)

        # Concatenate them to pass as single tensors to the rasterizer.
        polygon_vertices = tf.concat(batch_polygon_vertices, axis=0)
        vertex_counts_per_polygon = tf.concat(batch_vertex_counts_per_polygon, axis=0)
        class_ids_per_polygon = tf.concat(batch_class_ids_per_polygon, axis=0)
        polygons_per_image = tf.concat(batch_polygons_per_image, axis=0)

        polygon_raster = self._rasterizer(polygon_vertices=polygon_vertices,
                                          vertex_counts_per_polygon=vertex_counts_per_polygon,
                                          class_ids_per_polygon=class_ids_per_polygon,
                                          polygons_per_image=polygons_per_image)
        # Outside the input labels, the loss mask should have a value of 1.0 (i.e. the loss will
        #  be treated as usual in those cells).
        ones, zeros = tf.ones_like(polygon_raster), tf.zeros_like(polygon_raster)

        # If a mask exists, zero the polygon raster where the mask is nonzero.
        if mask is not None:
            objective_mask = tf.where(mask > 0., zeros, ones)
            polygon_raster *= objective_mask

        # Set all foreground values to the value of mask_multiplier.
        background = tf.where(polygon_raster > 0., zeros, ones)
        loss_mask = background + mask_multiplier * polygon_raster

        return loss_mask

    def rasterize_labels(self, batch_labels, mask=None, mask_multiplier=1.0):
        """Setup the rasterized loss mask for a given set of ground truth labels.

        Args:
            batch_labels (list of dicts of Tensors or Bbox2DLabel): If it were list of dicts of
                tensors, it contains the labels for a batch of frames. If it were Bbox2DLabel,
                it contains filtered labels for all frames in a batch.
            mask (Tensor): Where nonzero, the mask_multiplier is ignored (mask multiplier is set
                to the background value, 1.0). Default None, the mask_multiplier is never ignored.
            mask_multiplier (float): Scalar value that will be assigned to each region in a set
                of ground truth labels. Default value of 1.0 means the output is all filled with
                ones, essentially meaning all regions of the network's output are treated equally.

        Returns:
            loss_mask (tf.Tensor): rasterized loss mask corresponding to the input labels.
        """
        loss_mask_tensors = None
        if isinstance(batch_labels, list):
            loss_mask_tensors = self.rasterize_labels_dict(batch_labels, mask, mask_multiplier)
        elif isinstance(batch_labels, Bbox2DLabel):
            loss_mask_tensors = self.rasterize_labels_bbox_2d_label(batch_labels, mask,
                                                                    mask_multiplier)
        else:
            raise ValueError("Unsupported type.")
        return loss_mask_tensors

    def __call__(self, loss_mask_batch_labels, ground_truth_tensors=None, mask_multiplier=1.0):
        """Method that users will call to generate necessary loss masks.

        Args:
            loss_mask_batch_labels (nested dict): for now, has two levels:
                [target_class_name][objective_name]. The leaf values are the corresponding filtered
                ground truth labels in tf.Tensor for a batch of frames.
            mask_multiplier (float): Scalar value that will be assigned to each region in a set
                of ground truth labels. Default value of 1.0 means the output is all filled with
                ones, essentially meaning all regions of the network's output are treated equally.

        Returns:
            loss_masks (nested dict): Follows the same hierarchy as the input. Each leaf value
                is the loss mask in tf.Tensor form for the corresponding filter.
        """
        loss_masks = dict()
        for target_class_name in loss_mask_batch_labels:
            if target_class_name not in loss_masks:
                loss_masks[target_class_name] = dict()
            for objective_name, batch_labels in \
                    six.iteritems(loss_mask_batch_labels[target_class_name]):
                ground_truth_mask = ground_truth_tensors[target_class_name]['cov'] \
                    if ground_truth_tensors is not None else None
                loss_masks[target_class_name][objective_name] = \
                    self.rasterize_labels(batch_labels,
                                          mask=ground_truth_mask,
                                          mask_multiplier=mask_multiplier)

        return loss_masks
