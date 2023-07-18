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

"""Test loss mask rasterizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader import types
import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.loss_mask_rasterizer import LossMaskRasterizer

Canvas2D = nvidia_tao_tf1.core.types.Canvas2D


def _get_batch_vertices(batch_x, batch_y, batch_index):
    """Generate a set of labels for a batch of frames.

    Args:
        batch_x/batch_y/batch_index (list of lists): Outer list is for each frame, inner lists
            contain the coordinates of vertices and their polygon indices in that frame.

    Returns:
        batch_labels (list): Each element is a ground truth labels dict.
    """
    # Check they have the same number of 'frames'.
    assert len(batch_x) == len(batch_y)
    batch_labels = []
    for frame_idx in range(len(batch_x)):
        coordinates_x = batch_x[frame_idx]
        coordinates_y = batch_y[frame_idx]
        coordinates_index = batch_index[frame_idx]
        # Check the coordinate lists have the same number of elements.
        assert len(coordinates_x) == len(
            coordinates_y) == len(coordinates_index)
        _coordinates_x = tf.constant(coordinates_x, dtype=tf.float32)
        _coordinates_y = tf.constant(coordinates_y, dtype=tf.float32)
        _coordinates_index = tf.constant(coordinates_index, dtype=tf.int64)
        batch_labels.append({
            'target/coordinates/x': _coordinates_x,
            'target/coordinates/y': _coordinates_y,
            'target/coordinates/index': _coordinates_index
        })

    return batch_labels


def _get_bbox_2d_labels():
    """Bbox2DLabel for test preparation."""
    frame_indices = [0, 1, 2, 3, 3]
    object_class = tf.constant(
        ['pedestrian', 'unmapped', 'automobile', 'truck', 'truck'])
    bbox_coordinates = tf.constant(
        [7.0, 6.0, 8.0, 9.0,
         2.0, 3.0, 4.0, 5.0,
         0.0, 0.0, 3.0, 4.0,
         1.2, 3.4, 5.6, 7.8,
         4.0, 4.0, 10.0, 10.0])
    world_bbox_z = tf.constant([1.0, 2.0, 3.0, -1.0, -2.0])
    front = tf.constant([0.5, 1.0, -0.5, -1.0, 0.5])
    back = tf.constant([-1.0, 0.0, 0.0, 0.63, -1.0])

    canvas_shape = Canvas2D(height=tf.ones([1, 12]), width=tf.ones([1, 12]))

    sparse_coordinates = tf.SparseTensor(
        values=bbox_coordinates,
        dense_shape=[5, 5, 2, 2],
        indices=[[f, 0, j, k]
                 for f in frame_indices
                 for j in range(2)
                 for k in range(2)])
    sparse_object_class = tf.SparseTensor(
        values=object_class,
        dense_shape=[5, 5, 1],
        indices=[[f, 0, 0]
                 for f in frame_indices])
    sparse_world_bbox_z = tf.SparseTensor(
        values=world_bbox_z,
        dense_shape=[5, 5, 1],
        indices=[[f, 0, 0]
                 for f in frame_indices])
    sparse_front = tf.SparseTensor(
        values=front,
        dense_shape=[5, 5, 1],
        indices=[[f, 0, 0]
                 for f in frame_indices])
    sparse_back = tf.SparseTensor(
        values=back,
        dense_shape=[5, 5, 1],
        indices=[[f, 0, 0]
                 for f in frame_indices])

    source_weight = [tf.constant(2.0, tf.float32)]

    # Initialize all fields to empty lists (to signify 'optional' fields).
    bbox_2d_label_kwargs = {field_name: []
                            for field_name in types.Bbox2DLabel._fields}

    bbox_2d_label_kwargs.update({
        'frame_id': tf.constant('bogus'),
        'object_class': sparse_object_class,
        'vertices': types.Coordinates2D(
            coordinates=sparse_coordinates, canvas_shape=canvas_shape),
        'world_bbox_z': sparse_world_bbox_z,
        'front': sparse_front,
        'back': sparse_back,
        'source_weight': source_weight})

    return types.Bbox2DLabel(**bbox_2d_label_kwargs)


class TestLossMaskRasterizer:
    def test_loss_mask_rasterizer_setup(self):
        """Test that the LossMaskRasterizer setup follows the input hierarchy."""
        # Instantiate a LossMaskRasterizer.
        loss_mask_rasterizer = LossMaskRasterizer(
            input_width=1,
            input_height=2,
            output_width=3,
            output_height=4
        )
        # Get some dummy labels for old data format.
        batch_x = [[1., 7., 7., 1., 2., 8., 8., 2., 3., 9., 9., 3.]]
        batch_y = [[4., 4., 10., 10., 5., 5., 11., 11., 6., 6., 12., 12.]]
        batch_idx = [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]]
        loss_mask_batch_labels = \
            {'depth':  # Outer level is 'objective'.
                {'person':  # Second level is 'target class'.
                    _get_batch_vertices(batch_x, batch_y, batch_idx),
                 'car': _get_batch_vertices(batch_x, batch_y, batch_idx)},
             'bbox': {'road_sign': _get_batch_vertices(batch_x, batch_y, batch_idx)}}

        loss_mask_tensors = loss_mask_rasterizer(loss_mask_batch_labels)

        # Check that the output keeps the same 'hierarchy' on rasterizing labels of old data.
        assert set(loss_mask_batch_labels.keys()) == set(
            loss_mask_tensors.keys())
        for objective_name in loss_mask_batch_labels:
            assert set(loss_mask_batch_labels[objective_name].keys()) == \
                set(loss_mask_tensors[objective_name].keys())

        # Re-instantiate the rasterizer for larger input and output size.
        loss_mask_rasterizer2 = LossMaskRasterizer(
            input_width=20,
            input_height=22,
            output_width=10,
            output_height=11
        )
        # Get dummy labels for bbox2d_label.
        loss_mask_batch_labels2 = \
            {'depth':  # Outer level is 'objective'.
                {'person':  # Second level is 'target class'.
                    _get_bbox_2d_labels(),
                 'car': _get_bbox_2d_labels()},
             'bbox': {'road_sign': _get_bbox_2d_labels()}}
        loss_mask_tensors2 = loss_mask_rasterizer2(loss_mask_batch_labels2)
        # Check that the output keeps the same 'hierarchy' on rasterizing labels of bbox2d_label.
        assert set(loss_mask_batch_labels.keys()) == set(
            loss_mask_tensors2.keys())
        for objective_name in loss_mask_batch_labels:
            assert set(loss_mask_batch_labels[objective_name].keys()) == \
                set(loss_mask_tensors2[objective_name].keys())

    def _get_expected_rasterizer_args(self, coords_x, coords_y, coords_idx):
        """Helper function that generates the expected inputs to the rasterizer.

        Args:
            coords_x/coords_y/coords_idx (list): Contain the coordinates and index of polygons in a
             frame.

        Returns:
            polygon_vertices:
            vertex_counts_per_polygon:
            class_ids_per_polygon:
            polygons_per_image:
        """
        polygon_vertices = []
        for i in range(len(coords_x)):
            polygon_vertices.extend([[coords_x[i], coords_y[i]]])
        vertex_counts_per_polygon = np.bincount(coords_idx)
        polygons_per_image = [len(vertex_counts_per_polygon)]
        class_ids_per_polygon = [0] * len(vertex_counts_per_polygon)

        return polygon_vertices, vertex_counts_per_polygon, class_ids_per_polygon, \
            polygons_per_image

    @pytest.mark.parametrize(
        "input_width,input_height,output_width,output_height,batch_x,batch_y,batch_idx",
        [
            # First test case is without scaling.
            (10, 10, 10, 10,
             [1., 7., 7., 1., 2., 8., 8., 2., 3.,
                 9., 9., 3.],  # batch x-coordinates
                [4., 4., 10., 10., 5., 5., 11., 11., 6.,
                 6., 12., 12.],  # batch y-coordinates
                # batch coordinate indices
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
             ),
            # Now scale by half along both dimensions.
            (10, 10, 5, 5,
                [1., 7., 7., 1., 2., 8., 8., 2., 3.,
                    9., 9., 3.],  # batch x-coordinates
                [4., 4., 10., 10., 5., 5., 11., 11., 6.,
                 6., 12., 12.],  # batch y-coordinates
                # batch coordinate indices
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
             ),
            # Now scale by 2.2 along both dimensions.
            (10, 10, 22, 22,
                [1., 7., 7., 1., 2., 8., 8., 2., 3.,
                    9., 9., 3.],  # batch x-coordinates
                [4., 4., 10., 10., 5., 5., 11., 11., 6.,
                 6., 12., 12.],  # batch y-coordinates
                # batch coordinate indices
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
             ),
            # Now scale by different factors along each dimensions.
            (10, 10, 22, 8,
                [1., 7., 7., 1., 2., 8., 8., 2., 3.,
                    9., 9., 3.],  # batch x-coordinates
                [4., 4., 10., 10., 5., 5., 11., 11., 6.,
                 6., 12., 12.],  # batch y-coordinates
                # batch coordinate indices
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
             )
        ]
    )
    def test_loss_mask_label_translation_dict(self,
                                              input_width,
                                              input_height,
                                              output_width,
                                              output_height,
                                              batch_x, batch_y, batch_idx):
        """Test that the args passed to the PolygonRasterizer are sane.

        Args:
            input_width/height (int): Input image dimensions.
            output_width/height (int): Output raster dimensions.
            x1/y1/x2/y2 (list): Contain the coordinates of bboxes in a frame.
        """
        loss_mask_rasterizer = LossMaskRasterizer(
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height
        )
        # Get some dummy labels for a single frame.
        loss_mask_frame_labels = _get_batch_vertices(
            [batch_x], [batch_y], [batch_idx])[0]
        scale_x = output_width / input_width
        scale_y = output_height / input_height
        scaled_x = [_x * scale_x for _x in batch_x]
        scaled_y = [_y * scale_y for _y in batch_y]
        expected_args = \
            self._get_expected_rasterizer_args(scaled_x, scaled_y, batch_idx)
        computed_args = loss_mask_rasterizer.translate_frame_labels_dict(
            loss_mask_frame_labels)
        # Check that the translation works as expected.
        with tf.compat.v1.Session() as sess:
            for i in range(len(expected_args)):
                np.testing.assert_allclose(
                    np.array(expected_args[i]), sess.run(computed_args[i]))

    @pytest.mark.parametrize(
        "input_width,input_height,output_width,output_height,exp_vertx,exp_verty,exp_polygon_num",
        [
            # First test case is without scaling.
            (11, 13, 11, 13,
             [7., 8., 8., 7., 2., 4., 4., 2., 0., 3., 3.,
              0., 1.2, 5.6, 5.6, 1.2, 4., 10., 10., 4.],
                [6., 6., 9., 9., 3., 3., 5., 5., 0., 0., 4.,
                 4., 3.4, 3.4, 7.8, 7.8, 4., 4., 10., 10.],
                [1, 1, 1, 2, 0],
             ),
            # Now scale by 2 along both dimensions(output_width/input_width=22/11=2).
            (11, 11, 22, 22,
                [14., 16., 16., 14., 4., 8., 8., 4., 0., 6., 6., 0., 2.4, 11.2, 11.2, 2.4, 8.,
                 20., 20., 8.],
                [12., 12., 18., 18., 6., 6., 10., 10., 0., 0., 8., 8., 6.8, 6.8, 15.6, 15.6,
                 8., 8., 20., 20.],
                [1, 1, 1, 2, 0],
             ),
        ]
    )
    def test_loss_mask_label_translation_bbox_2d_label(self,
                                                       input_width,
                                                       input_height,
                                                       output_width,
                                                       output_height,
                                                       exp_vertx,
                                                       exp_verty,
                                                       exp_polygon_num):
        """Test that the args passed to the PolygonRasterizer are sane.

        Args:
            input_width/height (int): Input image dimensions.
            output_width/height (int): Output raster dimensions.
            exp_vertx (float): expected x1/x2 list for polygons
            exp_verty (float): expected y1/y2 list for polygons
            exp_polygon_num (int): expected polygon num for each frames
        """
        loss_mask_rasterizer = LossMaskRasterizer(
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height
        )
        # Get some dummy labels for a single frame.
        loss_mask_frame_labels = _get_bbox_2d_labels()
        polygon_vertices, vertex_counts_per_polygon, class_ids_per_polygon, polygons_per_image = \
            loss_mask_rasterizer.translate_frame_labels_bbox_2d_label(
                loss_mask_frame_labels)

        # Check that the translation works as expected.
        with tf.compat.v1.Session() as sess:
            polygon_vertices_output = sess.run(polygon_vertices)
            np.testing.assert_allclose(
                polygon_vertices_output[:, 0], np.array(exp_vertx))
            np.testing.assert_allclose(
                polygon_vertices_output[:, 1], np.array(exp_verty))
            vertex_counts_output = sess.run(vertex_counts_per_polygon)
            np.testing.assert_equal(
                vertex_counts_output, np.array([4] * len(exp_polygon_num)))
            class_ids_output = sess.run(class_ids_per_polygon)
            np.testing.assert_equal(
                class_ids_output, np.array([0] * len(exp_polygon_num)))
            polygons_per_image_output = sess.run(polygons_per_image)
            np.testing.assert_equal(
                polygons_per_image_output, np.array(exp_polygon_num))

    # TODO(@williamz): Could consider saving the rasters like in maglev/processors/
    #  test_bbox_rasterizer_ref?
    @pytest.mark.parametrize(
        "input_width,input_height,output_width,output_height,mask_multiplier,"
        "batch_x,batch_y,batch_idx,expected_mask",
        [
            # Case 1: First, use a single frame.
            (10, 10, 5, 5, 0.0,
             # batch x-coordinates
             [[1., 7., 7., 1., 2., 8., 8., 2., 3., 9., 9., 3.]],
                # batch y-coordinates
                [[4., 4., 10., 10., 5., 5., 11., 11., 6., 6., 12., 12.]],
                # batch coordinate indices
                [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]],
                np.array([[[[1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [0., 0., 0., 0., 1.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]]], dtype=np.float32)
             ),  # ----- End case 1.
            # Case 2: Now has two frames. All boxes in the second frame are out of bound.
            (10, 10, 5, 5, 1.5,
                [[1., 7., 7., 1., 2., 8., 8., 2., 3., 9., 9., 3.],
                 # batch x-coordinates
                 [13., 19., 19., 13., 14., 20., 20., 14., 15., 21., 21., 15.]],
                [[4., 4., 10., 10., 5., 5., 11., 11., 6., 6., 12., 12.],
                 # batch y-coordinates
                 [16., 16., 22., 22., 17., 17., 23., 23., 18., 18., 24., 24.]],
                [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                 [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]],  # batch coordinate indices
                np.array([[[[1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            # Reminder: 1.5 is the multiplier for this case.
                            [1.5, 1.5, 1.5, 1.5, 1.],
                            [1.5, 1.5, 1.5, 1.5, 1.5],
                            [1.5, 1.5, 1.5, 1.5, 1.5]]],  # end of first frame.
                          # Nothing in second frame.
                          [np.ones((5, 5)).tolist()]
                          ], dtype=np.float32)
             ),  # ----- End case 2.
            # Case 3: Test empty frame.
            (10, 10, 5, 5, 2.0,  # Should not matter.
                [[]], [[]], [[]],  # Empty batch_x/y/coordinates.
                np.ones((5, 5), dtype=np.float32).reshape(1, 1, 5, 5)
             )  # ----- End case 3.
        ]
    )
    def test_loss_mask_rasters_dict(self,
                                    input_width,
                                    input_height,
                                    output_width,
                                    output_height,
                                    mask_multiplier,
                                    batch_x, batch_y, batch_idx,
                                    expected_mask):
        """Test that the masks produced by the LossMaskRasterizer are sane.

        Args:
            input_width/height (int): Input image dimensions.
            output_width/height (int): Output raster dimensions.
            mask_multiplier (float): Value that should be present in the loss masks.
            batch_x1/y1/x2/y2 (list of lists): Outer list is for each frame, inner lists contain
                the coordinates of bboxes in that frame.
            expected_mask (np.array): of shape [len(batch_x1), 1, output_height, output_width] which
                is the 'golden' truth against which the raster will be compared.
        """
        loss_mask_rasterizer = LossMaskRasterizer(
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
        )
        # Get some dummy labels.
        loss_mask_batch_labels = \
            {'car': {'bbox': _get_batch_vertices(batch_x, batch_y, batch_idx)}}
        loss_mask_tensor_dict = loss_mask_rasterizer(loss_mask_batch_labels,
                                                     mask_multiplier=mask_multiplier)
        # Run the rasterization.
        with tf.compat.v1.Session() as sess:
            loss_mask_rasters = sess.run(loss_mask_tensor_dict)
            # Check dict structure.
            assert set(loss_mask_tensor_dict.keys()) == set(
                loss_mask_batch_labels.keys())
            for target_class_name in loss_mask_tensor_dict:
                assert set(loss_mask_tensor_dict[target_class_name].keys()) == \
                    set(loss_mask_batch_labels[target_class_name].keys())
                for obj_name in loss_mask_tensor_dict[target_class_name]:
                    # Compare with golden value.
                    np.testing.assert_allclose(loss_mask_rasters[target_class_name][obj_name],
                                               expected_mask)

    def test_loss_mask_rasters_bbox_2d_label(self):
        """Test that LossMaskRasterizer works correctly with bbox 2d labels."""
        loss_mask_rasterizer = LossMaskRasterizer(
            input_width=13,
            input_height=11,
            output_width=13,
            output_height=11)

        # Get all labels.
        all_labels = _get_bbox_2d_labels()

        # Empty groundtruth.
        empty_loss_mask_tensor = np.ones(shape=(11, 13), dtype=np.float32)

        # Case 1: activate bbox 4 and 5.
        gt_rast_tensor1 = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                               2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                           [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                               2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                           [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                               2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                           [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                               2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                               2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                               2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        mask1 = tf.constant(
            np.array([False, False, False, True, True]), dtype=tf.bool)
        filtered_labels1 = all_labels.filter(mask1)
        loss_mask_batch_labels1 = \
            {'car': {'bbox': filtered_labels1}}
        loss_mask_tensor_dict1 = loss_mask_rasterizer(loss_mask_batch_labels1,
                                                      mask_multiplier=2.0)
        with tf.compat.v1.Session() as sess:
            output_loss_mask_tensor_dict1 = sess.run(loss_mask_tensor_dict1)
            output_loss_mask_tensor = output_loss_mask_tensor_dict1['car']['bbox']
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[0, :, :]),
                                       empty_loss_mask_tensor)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[1, :, :]),
                                       empty_loss_mask_tensor)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[2, :, :]),
                                       empty_loss_mask_tensor)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[3, :, :]),
                                       gt_rast_tensor1)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[4, :, :]),
                                       empty_loss_mask_tensor)

        # Case 2: activate bbox 3,4,5.
        gt_rast_tensor2 = np.ones(shape=(11, 13), dtype=np.float32)
        gt_rast_tensor2[0:4, 0:3] = 0.0
        gt_rast_tensor3 = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        mask2 = tf.constant(
            np.array([False, False, True, True, True]), dtype=tf.bool)
        filtered_labels2 = all_labels.filter(mask2)
        loss_mask_batch_labels2 = \
            {'car': {'bbox': filtered_labels2}}
        loss_mask_tensor_dict2 = loss_mask_rasterizer(loss_mask_batch_labels2,
                                                      mask_multiplier=0.0)
        with tf.compat.v1.Session() as sess:
            output_loss_mask_tensor_dict2 = sess.run(loss_mask_tensor_dict2)
            output_loss_mask_tensor = output_loss_mask_tensor_dict2['car']['bbox']
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[0, :, :]),
                                       empty_loss_mask_tensor)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[1, :, :]),
                                       empty_loss_mask_tensor)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[2, :, :]),
                                       gt_rast_tensor2)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[3, :, :]),
                                       gt_rast_tensor3)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[4, :, :]),
                                       empty_loss_mask_tensor)

    def test_loss_mask_raster_with_groundtruth_mask_dict(self):
        """Test that masks produced by LossMaskRasterizer when not ignoring groundtruth are sane."""
        loss_mask_rasterizer = LossMaskRasterizer(
            input_width=4, input_height=4, output_width=4, output_height=4)
        batch_x, batch_y, batch_idx = \
            [[1., 4., 4., 1.]], [[1., 1., 4., 4.]], [[0, 0, 0, 0]]
        loss_mask_batch_labels = \
            {'car': {'bbox': _get_batch_vertices(batch_x, batch_y, batch_idx)}}
        car_cov = [
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 1., 1.],
            [0., 0., 1., 0.],
        ]
        ground_truth_tensors = \
            {'car': {'cov': tf.constant(
                car_cov, dtype=tf.float32, shape=(1, 1, 4, 4))}}
        expected_mask = [[[
            [1., 1., 1., 1.],
            [1., 2., 1., 2.],
            [1., 1., 1., 1.],
            [1., 2., 1., 2.],
        ]]]
        loss_mask_tensor_dict = loss_mask_rasterizer(loss_mask_batch_labels,
                                                     ground_truth_tensors=ground_truth_tensors,
                                                     mask_multiplier=2.)

        with tf.compat.v1.Session() as sess:
            loss_mask_rasters = sess.run(loss_mask_tensor_dict)

            np.testing.assert_allclose(
                loss_mask_rasters['car']['bbox'], expected_mask)

    def test_loss_mask_raster_with_groundtruth_mask_bbox_2d_label(self):
        """Test groundtruth mask works well for rasterizer with bbox_2d_label type."""
        loss_mask_rasterizer = LossMaskRasterizer(
            input_width=13,
            input_height=11,
            output_width=13,
            output_height=11)

        # Get all labels.
        all_labels = _get_bbox_2d_labels()

        # Empty groundtruth.
        empty_loss_mask_tensor = np.ones(shape=(11, 13), dtype=np.float32)

        # Final rasterized groundtruth with mask.
        gt_rast1 = np.ones(shape=(11, 13), dtype=np.float32)
        gt_rast1[3:5, 2:4] = 2.0
        gt_rast1[3, 2] = 1.0

        # Ground truth mask.
        car_cov = np.zeros(shape=(5, 11, 13), dtype=np.float32)
        car_cov[1, 0:4, 0:3] = 1.0
        ground_truth_tensors = \
            {'car': {'cov': tf.constant(
                car_cov, dtype=tf.float32, shape=(5, 1, 11, 13))}}

        # Only select bbox 1 for rasterization.
        bbox_indices = tf.constant(
            np.array([False, True, False, False, False]), dtype=tf.bool)
        filtered_labels = all_labels.filter(bbox_indices)
        loss_mask_batch_labels = \
            {'car': {'cov': filtered_labels}}
        loss_mask_tensor_dict = loss_mask_rasterizer(loss_mask_batch_labels,
                                                     ground_truth_tensors=ground_truth_tensors,
                                                     mask_multiplier=2.0)

        with tf.compat.v1.Session() as sess:
            output_loss_mask_tensor_dict = sess.run(loss_mask_tensor_dict)
            output_loss_mask_tensor = output_loss_mask_tensor_dict['car']['cov']
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[0, :, :]),
                                       empty_loss_mask_tensor)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[1, :, :]),
                                       gt_rast1)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[2, :, :]),
                                       empty_loss_mask_tensor)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[3, :, :]),
                                       empty_loss_mask_tensor)
            np.testing.assert_allclose(np.squeeze(output_loss_mask_tensor[4, :, :]),
                                       empty_loss_mask_tensor)
