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

import itertools
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import Bbox2DLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types import Coordinates2D
import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer import BboxRasterizer
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer import BboxRasterizerInput
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer_config import BboxRasterizerConfig


Canvas2D = tao_core.types.Canvas2D
INPUT_HEIGHT = 6
INPUT_WIDTH = 15


class TestBboxRasterizer:
    @pytest.fixture(scope='function')
    def bbox_rasterizer(self):
        """Instantiate a BboxRasterizer."""
        bbox_rasterizer_config = BboxRasterizerConfig(deadzone_radius=0.67)
        bbox_rasterizer_config['car'] = \
            BboxRasterizerConfig.TargetClassConfig(
                cov_center_x=0.5, cov_center_y=0.5, cov_radius_x=1.0, cov_radius_y=1.0,
                bbox_min_radius=1.0)
        bbox_rasterizer_config['person'] = \
            BboxRasterizerConfig.TargetClassConfig(
                cov_center_x=0.5, cov_center_y=0.5, cov_radius_x=0.5, cov_radius_y=0.5,
                bbox_min_radius=1.0)
        bbox_rasterizer = BboxRasterizer(
            input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, output_width=5, output_height=3,
            target_class_names=['car', 'person'], bbox_rasterizer_config=bbox_rasterizer_config,
            target_class_mapping={'pedestrian': 'person', 'automobile': 'car', 'van': 'car'})

        return bbox_rasterizer

    def test_bbox_from_rumpy_params(self, bbox_rasterizer):
        """Test that the bbox matrix, coverage radius, and inverse bbox area are correct.

        Args:
            bbox_rasterizer: BboxRasterizer obtained from above fixture.
        """
        xmin, ymin = tf.constant([1.0]), tf.constant([2.0])
        xmax, ymax = tf.constant([3.0]), tf.constant([4.0])
        cov_center_x, cov_center_y = tf.constant([0.5]), tf.constant([0.5])
        cov_radius_x, cov_radius_y = tf.constant([0.6]), tf.constant([0.6])
        bbox_min_radius = tf.constant([0.5])
        deadzone_radius = 1.0

        mat, cov_radius, inv_bbox_area = bbox_rasterizer.bbox_from_rumpy_params(
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
            cov_center_x=cov_center_x, cov_center_y=cov_center_y,
            cov_radius_x=cov_radius_x, cov_radius_y=cov_radius_y,
            bbox_min_radius=bbox_min_radius, deadzone_radius=deadzone_radius)

        with tf.compat.v1.Session() as sess:
            mat, cov_radius, inv_bbox_area = sess.run(
                [mat, cov_radius, inv_bbox_area])

        # Check values are as expected.
        assert np.allclose(cov_radius, np.array([0.6, 0.6], dtype=np.float32))
        # bbox area = 2 * cov_radius_x * 2 * cov_radius_y in this case.
        assert np.allclose(inv_bbox_area, np.array(
            [1.0 / (4.0 * 0.36)], dtype=np.float32))
        # These should the center coordinates * -1.0 / deadzone_radius.
        assert np.allclose(mat, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-2.0, -3.0, 1.0]],
                                         dtype=np.float32))

    def labels(self, dataloader_type, sequence_length, output_type):
        """Create test labels.

        Args:
            dataloader_type (str): Dataloader type: 'old' or 'common'.
            sequence_length (int): Sequence length.
            output_type (str): Output type for sequence models: 'last' or 'all'.

        Returns:
            dict or Bbox2DLabel-namedtuple depending on the dataloader_type.
        """
        object_class = tf.constant(
            ['pedestrian', 'automobile', 'van', 'unmapped'])
        bbox_coordinates = tf.constant(
            [[7.0, 6.0, 8.0, 9.0],
             [2.0, 3.0, 4.0, 5.0],
             [0.0, 0.0, 3.0, 4.0],
             [1.2, 3.4, 5.6, 7.8]])
        world_bbox_z = tf.constant([1.0, 2.0, 3.0, -1.0])

        if sequence_length == 1 or output_type == 'last':
            sequence_range = [sequence_length-1]
        else:
            sequence_range = range(sequence_length)

            object_class_2 = tf.constant(
                ['automobile', 'pedestrian', 'unmapped', 'unmapped'])
            bbox_coordinates_2 = tf.constant(
                [[2.0, 3.0, 4.0, 5.0],
                 [7.0, 6.0, 8.0, 9.0],
                 [0.0, 0.0, 3.0, 4.0],
                 [1.2, 3.4, 5.6, 7.8]])
            world_bbox_z_2 = tf.constant([1.0, 2.0, 3.0, -1.0])

            object_class = tf.concat([object_class_2, object_class], 0)
            bbox_coordinates = tf.concat(
                [bbox_coordinates_2, bbox_coordinates], 0)
            world_bbox_z = tf.concat([world_bbox_z_2, world_bbox_z], 0)

        if dataloader_type == 'old':
            labels = {
                'target/object_class': object_class,
                'target/bbox_coordinates': bbox_coordinates,
                'target/world_bbox_z': world_bbox_z}
        elif dataloader_type == 'common':
            canvas_shape = Canvas2D(height=tf.ones([1, sequence_length, INPUT_HEIGHT]),
                                    width=tf.ones([1, sequence_length, INPUT_WIDTH]))

            sparse_coordinates = tf.SparseTensor(
                values=tf.reshape(bbox_coordinates, [-1]),
                dense_shape=[1, sequence_length, 4, 2, 2],
                indices=[[0, s, i, j, k]
                         for s in sequence_range
                         for i in range(4)
                         for j in range(2)
                         for k in range(2)])
            sparse_object_class = tf.SparseTensor(
                values=object_class,
                dense_shape=[1, sequence_length, 4],
                indices=[[0, s, i]
                         for s in sequence_range
                         for i in range(4)])
            sparse_world_bbox_z = tf.SparseTensor(
                values=world_bbox_z,
                dense_shape=[1, sequence_length, 4],
                indices=[[0, s, i]
                         for s in sequence_range
                         for i in range(4)])

            # Initialize all fields to empty lists (to signify 'optional' fields).
            bbox_2d_label_kwargs = {field_name: []
                                    for field_name in Bbox2DLabel._fields}

            bbox_2d_label_kwargs.update({
                'frame_id': tf.constant('bogus'),
                'object_class': sparse_object_class,
                'vertices': Coordinates2D(
                    coordinates=sparse_coordinates, canvas_shape=canvas_shape),
                'world_bbox_z': sparse_world_bbox_z})

            labels = Bbox2DLabel(**bbox_2d_label_kwargs)

        return labels

    @pytest.mark.parametrize(
        "dataloader_type,sequence_length,output_type",
        [('old', 1, None), ('common', 1, None),
         ('common', 1, 'last'), ('common', 1, 'all'),
         ('common', 2, 'last'), ('common', 2, 'all')])
    @pytest.mark.parametrize(
        "exp_num_bboxes,exp_bbox_class_ids,exp_bbox_matrices,exp_bbox_coverage_radii,"
        "exp_bbox_flags,exp_inv_bbox_area,exp_output_space_coordinates,exp_object_class,"
        "exp_world_bbox_z",
        # Define sequence of two outputs. If sequence_length == 1, only the latter test
        # case is used.
        [
            (
                [2, 3],
                [[0, 1], [1, 0, 0]],
                [np.array([[[1.0,         0.0, 0.0],
                            [0.0,  0.66666675, 0.0],
                            [-1.0, -1.3333335, 1.0]],
                           [[1.0,         0.0, 0.0],
                            [0.0,  0.26666668, 0.0],
                            [-2.5,       -1.0, 1.0]],
                           ], dtype=np.float32),
                    np.array([[[1.0,         0.0, 0.0],
                               [0.0,  0.26666668, 0.0],
                               [-2.5,       -1.0, 1.0]],
                              [[1.0,         0.0, 0.0],
                               [0.0,  0.66666675, 0.0],
                               [-1.0, -1.3333335, 1.0]],
                              [[1.0,   0.0, 0.0],
                               [0.0,   0.5, 0.0],
                               [-0.5, -0.5, 1.0]],
                              ], dtype=np.float32)],  # end exp_bbox_matrices
                [np.ones((2, 2), dtype=np.float32),
                 np.ones((3, 2), dtype=np.float32)],  # exp_bbox_coverage_radii
                [np.ones((2,), dtype=np.uint8),
                 np.ones((3,), dtype=np.uint8)],  # exp_bbox_flags
                [np.array([0.16666669, 0.06666667], dtype=np.float32),
                 # end exp_inv_bbox_area
                 np.array([0.06666667, 0.16666669, 0.125], dtype=np.float32)],
                [np.array([[0.6666667, 2.3333335],
                           [1.5,              3.],
                           [1.3333334, 2.6666667],
                           [2.5,            4.5]], dtype=np.float32),
                 np.array([[2.3333335, 0.6666667, 0.0],
                           [3.,        1.5,       0.0],
                           [2.6666667, 1.3333334, 1.0],
                           [4.5,       2.5,       2.0]],
                          dtype=np.float32)],  # exp_output_space_coordinates
                [['car', 'person'],
                 # exp_object_class. These should now be mapped and filtered.
                 ['person', 'car', 'car']],
                [np.array([1.0, 2.0], dtype=np.float32),
                 np.array([1.0, 2.0, 3.0], dtype=np.float32)],  # exp_world_bbox_z
            )
        ]
    )
    def test_get_target_gradient_info(
            self, bbox_rasterizer, dataloader_type, sequence_length, output_type,
            exp_num_bboxes, exp_bbox_class_ids, exp_bbox_matrices, exp_bbox_coverage_radii,
            exp_bbox_flags, exp_inv_bbox_area, exp_output_space_coordinates, exp_object_class,
            exp_world_bbox_z):
        """Test that the inputs for the SDK are correctly computed.

        Args:
            bbox_rasterizer: BboxRasterizer obtained from above fixture.
            dataloader_type (str): Dataloader type: 'old' or 'common'.
            sequence_length (int): Sequence length.
            output_type (str): Output type for sequence models: 'last' or 'all'.
            exp_num_bboxes (int): Expected number of bboxes.
            exp_bbox_class_ids (list): Expected class ids (int).
            exp_bbox_matrices (np.array): Expected bbox matrices.
            exp_bbox_coverage_radii (list): Expected coverage radii (float).
            exp_bbox_flags (list): Expected bbox flags.
            exp_inv_bbox_area (np.array): Expected inverse bbox areas.
            exp_output_space_coordinates (np.array): Expected coordinates of the bboxes in the
                model output space.
            exp_object_class (list): Expected class names.
            exp_world_bbox_z (np.array): Expected depth coordinates.
        """
        labels = self.labels(dataloader_type, sequence_length, output_type)

        # Expected label sequence length.
        exp_sequence_length = sequence_length if output_type == 'all' else 1

        bbox_rasterizer.output_type = output_type
        _inputs = bbox_rasterizer.get_target_gradient_info(labels)
        # Need to initialize lookup tables.
        tables_initializer = tf.compat.v1.tables_initializer()

        with tf.compat.v1.Session() as sess:
            sess.run(tables_initializer)
            num_bboxes, bbox_class_ids, bbox_matrices, bbox_coverage_radii, bbox_flags, \
                gradient_info = sess.run([_inputs.num_bboxes, _inputs.bbox_class_ids,
                                          _inputs.bbox_matrices, _inputs.bbox_coverage_radii,
                                          _inputs.bbox_flags, _inputs.gradient_info])

        assert (num_bboxes == np.array(
            exp_num_bboxes[-exp_sequence_length:])).all()
        assert (bbox_class_ids == np.array(list(itertools.chain.from_iterable(
            exp_bbox_class_ids[-exp_sequence_length:])))).all()
        assert np.allclose(bbox_matrices,
                           np.concatenate(exp_bbox_matrices[-exp_sequence_length:], axis=0))
        assert np.allclose(bbox_coverage_radii,
                           np.concatenate(exp_bbox_coverage_radii[-exp_sequence_length:], axis=0))
        assert (bbox_flags == np.concatenate(
            exp_bbox_flags[-exp_sequence_length:], axis=0)).all()
        assert np.allclose(gradient_info['target/inv_bbox_area'],
                           np.concatenate(exp_inv_bbox_area[-exp_sequence_length:], axis=0))
        assert np.allclose(gradient_info['target/output_space_coordinates'],
                           np.concatenate(exp_output_space_coordinates[-exp_sequence_length:],
                                          axis=1))
        assert gradient_info['target/object_class'].astype(str).tolist() == \
            list(itertools.chain.from_iterable(
                exp_object_class[-exp_sequence_length:]))
        assert np.allclose(gradient_info['target/world_bbox_z'],
                           np.concatenate(exp_world_bbox_z[-exp_sequence_length:], axis=0))

    @pytest.fixture(scope='function', params=['old', 'common'])
    def rasterize_labels_input(self, request):
        """Prepare inputs to the rasterize_labels() method."""
        if request.param == 'old':
            batch_bbox_rasterizer_input, batch_gradients = [], []
            batch_bbox_rasterizer_input.append(
                BboxRasterizerInput(
                    num_bboxes=tf.constant(1),
                    bbox_class_ids=tf.constant([1]),  # person
                    bbox_matrices=tf.constant(
                        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-2.0, -3.0, 1.0]]]),
                    bbox_coverage_radii=tf.constant([[0.5, 0.5]]),
                    bbox_flags=tf.fill(
                        [1], tf.cast(tao_core.processors.BboxRasterizer.DRAW_MODE_ELLIPSE,
                                     tf.uint8)),
                    bbox_sort_values=tf.constant([0.]),
                    # Not needed since we are bypassing ObjectiveSet.
                    gradient_info=[]
                ))
            # Like cov objective.
            batch_gradients.append(tf.constant([[[0., 0., 1.]]]))
            batch_bbox_rasterizer_input.append(
                BboxRasterizerInput(
                    num_bboxes=tf.constant(1),
                    bbox_class_ids=tf.constant([0]),  # car
                    bbox_matrices=tf.constant(
                        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-3.0, -2.0, 1.0]]]),
                    bbox_coverage_radii=tf.constant([[0.5, 0.5]]),
                    bbox_flags=tf.fill(
                        [1], tf.cast(tao_core.processors.BboxRasterizer.DRAW_MODE_ELLIPSE,
                                     tf.uint8)),
                    bbox_sort_values=tf.constant([0.]),
                    # Not needed since we are bypassing ObjectiveSet.
                    gradient_info=[]
                ))
            # Like cov objective.
            batch_gradients.append(tf.constant([[[0., 0., 1.]]]))
        else:
            batch_bbox_rasterizer_input = BboxRasterizerInput(
                num_bboxes=tf.constant([1, 1]),
                bbox_class_ids=tf.constant([1, 0]),  # person, car.
                bbox_matrices=tf.constant(
                    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-2.0, -3.0, 1.0]],
                     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-3.0, -2.0, 1.0]]]),
                bbox_coverage_radii=tf.constant([[0.5, 0.5], [0.5, 0.5]]),
                bbox_flags=tf.fill([2], tf.cast(tao_core.processors.BboxRasterizer.DRAW_MODE_ELLIPSE,
                                                tf.uint8)),
                bbox_sort_values=tf.constant([0., 0.]),
                gradient_info=[])
            batch_gradients = tf.constant([[[0., 0., 1.]], [[0., 0., 1.]]])

        return batch_bbox_rasterizer_input, batch_gradients

    def test_rasterize_labels(self, bbox_rasterizer, rasterize_labels_input):
        """Test the rasterize_labels method."""
        batch_bbox_rasterizer_input, batch_gradients = rasterize_labels_input

        rasterized_tensors = bbox_rasterizer.rasterize_labels(
            batch_bbox_rasterizer_input=batch_bbox_rasterizer_input,
            batch_gradients=batch_gradients,
            num_gradients=1,
            gradient_flag=tao_core.processors.BboxRasterizer.GRADIENT_MODE_MULTIPLY_BY_COVERAGE,
        )

        expected_raster = np.zeros((2, 2, 1, 3, 5), dtype=np.float32)
        # Only a few output indices are non zero, and all equal to 0.1875.
        expected_value = 0.1875
        expected_raster[0, 1, 0, 2, 1:3] = expected_value
        expected_raster[1, 0, 0, 1:3, 2:4] = expected_value

        with tf.compat.v1.Session() as sess:
            raster = sess.run(rasterized_tensors)

        assert np.allclose(raster, expected_raster)
