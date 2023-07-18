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

"""DetectNet V2 visualization utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.cv.common.visualizer.base_visualizer import Descriptor
from nvidia_tao_tf1.cv.common.visualizer.tensorboard_visualizer import TensorBoardVisualizer
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer import BboxRasterizer

logger = logging.getLogger(__name__)


class TargetClassConfig(object):
        """Target class config."""

        def __init__(self, coverage_threshold):
            """Constructor."""
            self.coverage_threshold = coverage_threshold


class DetectNetTBVisualizer(TensorBoardVisualizer):
    """Visualizer implemented as a static class."""

    target_class_config = Descriptor("_target_class_config")

    @classmethod
    def build(cls, coverage_thresholds, enabled, num_images):
        """Build the Visualizer.

        Arguments:
            enabled (bool): Boolean to enabled visualizer.
            num_images (int): Number of images to be rendered.
        """
        # Create / set the properties.
        cls._target_class_config = {
            class_name: TargetClassConfig(coverage_threshold)
            for class_name, coverage_threshold in coverage_thresholds.items()
        }
        super().build(enabled,
                      num_images=num_images)

    @classmethod
    def build_from_config(cls, visualizer_config):
        """Build visualizer from config.

        Arguments:
            visualizer_config (visualizer_config_pb2.VisualizerConfig).
        """
        coverage_thresholds = {
            class_name: visualizer_config.target_class_config[class_name].coverage_threshold
            for class_name in visualizer_config.target_class_config
        }
        cls.build(
            coverage_thresholds, visualizer_config.enabled,
            visualizer_config.num_images
        )

    @classmethod
    def visualize_elliptical_bboxes(cls, target_class_names, input_images, coverage,
                                    abs_bboxes):
        """Visualize bboxes as ellipses on tensorboard.

        Args:
            target_class_names: List of target class names.
            input_images: Input images to visualize.
            coverage: Coverage predictions, shape
                [batch_size, num_classes, 1, grid_height, grid_width].
            abs_bboxes: Bounding box predictions in absolute coordinates, shape
                [batch_size, num_classes, 4, grid_height, grid_width].
        """
        # Compute the number of images to visualize as the minimum of the user
        # parameter and the actual minibatch size.
        batch_size = min(cls.num_images, input_images.shape[0])

        # Initially we have one bbox per grid cell.
        num_bboxes = [tf.cast(abs_bboxes.shape[3] * abs_bboxes.shape[4], tf.int32)]

        # Get visualization image size.
        image_height = tf.cast(input_images.shape[2], tf.int32)
        image_width = tf.cast(input_images.shape[3], tf.int32)

        # Constants.
        deadzone_radius = 1.0
        draw_mode = tao_core.processors.BboxRasterizer.DRAW_MODE_ELLIPSE

        # Loop over each image, and add predicted bboxes for each class to lists, sorted
        # by ascending coverage value.
        bboxes_per_image = []
        bbox_class_ids = []
        bbox_matrices = []
        bbox_gradients = []
        bbox_coverage_radii = []
        bbox_flags = []

        for n in range(batch_size):
            bboxes_per_class = 0
            for target_class_index, target_class_name in enumerate(target_class_names):
                # Extract input arrays and flatten.
                coverages = tf.reshape(coverage[n, target_class_index], num_bboxes)
                xmin = tf.reshape(abs_bboxes[n, target_class_index, 0], num_bboxes)
                ymin = tf.reshape(abs_bboxes[n, target_class_index, 1], num_bboxes)
                xmax = tf.reshape(abs_bboxes[n, target_class_index, 2], num_bboxes)
                ymax = tf.reshape(abs_bboxes[n, target_class_index, 3], num_bboxes)

                zero = tf.zeros(shape=num_bboxes)
                one = tf.zeros(shape=num_bboxes)
                # Bbox color comes from its coverage value.
                gradients = tf.transpose([[zero, zero, coverages]], (2, 0, 1))
                # Compute bbox matrices based on bbox coordinates.
                # Use constants for bbox params.
                matrices, coverage_radii, _ =\
                    BboxRasterizer.bbox_from_rumpy_params(xmin=xmin, ymin=ymin,
                                                          xmax=xmax, ymax=ymax,
                                                          cov_radius_x=tf.fill(num_bboxes, 1.0),
                                                          cov_radius_y=tf.fill(num_bboxes, 1.0),
                                                          bbox_min_radius=tf.fill(num_bboxes, 0.0),
                                                          cov_center_x=tf.fill(num_bboxes, 0.5),
                                                          cov_center_y=tf.fill(num_bboxes, 0.5),
                                                          deadzone_radius=deadzone_radius)

                flags = tf.fill(num_bboxes, tf.cast(draw_mode, tf.uint8))

                # Filter out bboxes with min > max.
                xdiff_mask = tf.cast(xmax > xmin, tf.float32)
                ydiff_mask = tf.cast(ymax > ymin, tf.float32)
                coverages *= xdiff_mask * ydiff_mask

                # Sort bboxes by ascending coverage.
                sort_value = -coverages
                _, sorted_indices = tf.nn.top_k(input=sort_value, k=num_bboxes[0])

                # Cut down work by throwing away bboxes with too small coverage.
                coverage_threshold = 0.
                if target_class_name in cls.target_class_config:
                    coverage_threshold =\
                        cls.target_class_config[target_class_name].coverage_threshold
                half = tf.cast(tf.reduce_sum(tf.where(tf.less(coverages, coverage_threshold),
                                                      one, zero)), tf.int32)
                sorted_indices = sorted_indices[half:]

                # Rearrange data arrays into sorted order, and append to lists.
                bboxes_per_class += tf.size(sorted_indices)
                bbox_class_ids.append(tf.fill(num_bboxes, target_class_index))
                bbox_matrices.append(tf.gather(matrices, sorted_indices))
                bbox_gradients.append(tf.gather(gradients, sorted_indices))
                bbox_coverage_radii.append(tf.gather(coverage_radii, sorted_indices))
                bbox_flags.append(tf.gather(flags, sorted_indices))

            bboxes_per_image += [bboxes_per_class]

        # Rasterize everything in one go.
        gradient_flags = [tao_core.processors.BboxRasterizer.GRADIENT_MODE_MULTIPLY_BY_COVERAGE]
        rasterizer = tao_core.processors.BboxRasterizer()
        images = rasterizer(num_images=batch_size,
                            num_classes=len(target_class_names),
                            num_gradients=1,
                            image_height=image_height,
                            image_width=image_width,
                            bboxes_per_image=bboxes_per_image,
                            bbox_class_ids=tf.concat(bbox_class_ids, axis=0),
                            bbox_matrices=tf.concat(bbox_matrices, axis=0),
                            bbox_gradients=tf.concat(bbox_gradients, axis=0),
                            bbox_coverage_radii=tf.concat(bbox_coverage_radii, axis=0),
                            bbox_flags=tf.concat(bbox_flags, axis=0),
                            gradient_flags=gradient_flags)

        # Add target classes dimension and tile it as many times as there are classes.
        inputs_tiled = tf.tile(tf.stack([input_images], axis=1),
                               [1, len(target_class_names), 1, 1, 1])

        # Show image through at spots where we have a strong prediction.
        images += inputs_tiled * images

        # Add images to Tensorboard.
        for target_class_index, target_class_name in enumerate(target_class_names):
            cls.image(
                '%s_preds' % target_class_name,
                images[:, target_class_index],
                collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])

    @classmethod
    def _draw_bboxes(cls, input_images, coverage, abs_bboxes, coverage_threshold=0.005):
        """
        Visualize bbox rectangles.

        Args:
            input_images: Tensor holding the input images (NCHW).
            coverage: Coverage predictions, shape [batch_size, 1, grid_height, grid_width].
            abs_bboxes: Bounding box predictions in absolute coordinates, shape
                [batch_size, 4, grid_height, grid_width].
            coverage_threshold (float32): Threshold value for coverage values to visualize.
        Returns:
            Images with drawn bounding boxes in NWHC format.
        """
        # Reshape the bbox predictions into [batch_size, num_bboxes, 4].
        batch_size = tf.cast(abs_bboxes.shape[0], tf.int32)
        num_bboxes = tf.cast(abs_bboxes.shape[2] * abs_bboxes.shape[3], tf.int32)
        bboxes = tf.transpose(tf.reshape(abs_bboxes, [batch_size, 4, num_bboxes]), [0, 2, 1])

        # Normalize bboxes to [0,1] range.
        height = tf.cast(input_images.shape[2], tf.float32)
        width = tf.cast(input_images.shape[3], tf.float32)
        xmin = bboxes[:, :, 0] / width
        ymin = bboxes[:, :, 1] / height
        xmax = bboxes[:, :, 2] / width
        ymax = bboxes[:, :, 3] / height
        # Convert to [y_min, x_min, y_max, x_max] order. Bboxes tensor shape is
        # [batch_size, num_bboxes, 4].
        bboxes = tf.stack([ymin, xmin, ymax, xmax], 2)

        # Mask out bboxes with coverage below a threshold or min > max.
        coverage = tf.reshape(coverage, [batch_size, num_bboxes, 1])
        coverage_mask = tf.cast(coverage > coverage_threshold, tf.float32)
        xdiff_mask = tf.reshape(tf.cast(xmax > xmin, tf.float32), [batch_size, num_bboxes, 1])
        ydiff_mask = tf.reshape(tf.cast(ymax > ymin, tf.float32), [batch_size, num_bboxes, 1])
        mask = coverage_mask * xdiff_mask * ydiff_mask
        bboxes = tf.multiply(bboxes, mask)

        # Convert input image to NHWC.
        input_images = tf.transpose(input_images, [0, 2, 3, 1])

        # Draw bboxes.
        output_images = tf.image.draw_bounding_boxes(input_images, bboxes)

        return output_images

    @classmethod
    def visualize_rectangular_bboxes(cls, target_class_names, input_images, coverage,
                                     abs_bboxes):
        """Visualize bboxes as rectangles on tensorboard.

        Args:
            target_class_names: List of target class names.
            input_images: Input images to visualize.
            coverage: Coverage predictions, shape [batch_size, 1, grid_height, grid_width].
            abs_bboxes: Bounding box predictions in absolute coordinates, shape
                [batch_size, 4, grid_height, grid_width].
        """
        # Loop over each target class and call visualization of the bounding boxes.
        for target_class_index, target_class_name in enumerate(target_class_names):

            # Look up coverage threshold for this class.
            coverage_threshold = 0.
            if target_class_name in cls.target_class_config:
                coverage_threshold =\
                    cls.target_class_config[target_class_name].coverage_threshold

            output_images = cls._draw_bboxes(input_images,
                                             coverage[:, target_class_index],
                                             abs_bboxes[:, target_class_index],
                                             coverage_threshold)

            cls.image(
                '%s_rectangle_bbox_preds' % target_class_name,
                output_images,
                data_format='channels_last',
                collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY]
            )
