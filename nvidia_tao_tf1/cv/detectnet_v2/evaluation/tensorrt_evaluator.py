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

"""A class to evaluate a DriveNet TensorRT engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from timeit import default_timer

import six
from six.moves import range as xrange
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import Bbox2DLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types import Coordinates2D
import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.evaluation import Evaluator

Canvas2D = nvidia_tao_tf1.core.types.Canvas2D

logger = logging.getLogger(__name__)


class TensorRTEvaluator(Evaluator):
    """Class for running evaluation using a TensorRT (TRT) engine."""

    def __init__(self,
                 postprocessing_config,
                 evaluation_config,
                 gridbox_model,
                 images,
                 ground_truth_labels,
                 steps,
                 confidence_models=None,
                 target_class_mapping=None,
                 sqlite_paths=None):
        """Constructor.

        Args:
            postprocessing_config (PostProcessingConfig): Object holding postprocessing parameters.
            evaluation_config: evaluation_config_pb2.EvaluationConfig object
            gridbox_model (GridboxModel): A GridboxModel instance.
            images: Dataset input tensors to be used for validation.
            ground_truth_labels (list): Each element is a dict of target features (each
                a tf.Tensor).
            steps (int): Number of minibatches to loop the validation dataset once.
            confidence_models (dict): A dict of ConfidenceModel instances, indexed by
                target class name. Can be None.
            target_class_mapping (dict): Maps from source class to target class (both str). Defaults
                to ``None``. If provided, forwards the information to dnn_metrics.
            sqlite_paths (list): If provided, is expected to be a list of paths (str) to HumanLoop
                sqlite exports. The reason this exists is to support the usecase where one has
                access to a sqlite file that contains labels for both detection ('BOX') and
                lanes ('POLYLINE').
        """
        super(TensorRTEvaluator, self).__init__(postprocessing_config=postprocessing_config,
                                                evaluation_config=evaluation_config,
                                                gridbox_model=gridbox_model,
                                                images=images,
                                                ground_truth_labels=ground_truth_labels,
                                                steps=steps,
                                                confidence_models=confidence_models,
                                                target_class_mapping=target_class_mapping,
                                                sqlite_paths=sqlite_paths)
        self._ground_truth_labels_placeholders = self._get_ground_truth_labels_placeholders()

    @property
    def keras_models(self):
        """Point to Keras Model objects with which to initialize the weights.

        Since the underlying models for TensorRT are not Keras models, return None.
        """
        return None

    def _get_ground_truth_labels_placeholders(self):
        """Create a placeholder for each ground truth tensor.

        Returns:
            placeholders (list or Bbox2DLabel): If self._ground_truth_labels is a list: List of
                dicts of tf.placeholders for ground truth labels. Each key in each dict is a
                ground truth label feature.
                If self._ground_truth_labels is a Bbox2DLabel: A Bbox2DLabel with placeholders.
        """
        if isinstance(self._ground_truth_labels, list):
            placeholders = []
            for frame_labels in self._ground_truth_labels:
                framewise_placeholders = dict()
                for label_name, tensor in six.iteritems(frame_labels):
                    framewise_placeholders[label_name] = tf.compat.v1.placeholder(
                        tensor.dtype)

                placeholders.append(framewise_placeholders)
        elif isinstance(self._ground_truth_labels, Bbox2DLabel):
            kwargs_bbox2dlabel = dict()
            labels_as_dict = self._ground_truth_labels._asdict()
            for field_name in Bbox2DLabel._fields:
                field_value = labels_as_dict[field_name]
                if field_name == 'vertices':
                    placeholder = Coordinates2D(
                        coordinates=tf.compat.v1.sparse.placeholder(
                            field_value.coordinates.values.dtype),
                        canvas_shape=Canvas2D(
                            height=tf.compat.v1.placeholder(
                                field_value.canvas_shape.height.dtype),
                            width=tf.compat.v1.placeholder(
                                field_value.canvas_shape.width.dtype)
                        ))
                elif isinstance(field_value, tf.SparseTensor):
                    placeholder = tf.compat.v1.sparse.placeholder(
                        field_value.values.dtype)
                elif isinstance(field_value, tf.Tensor):
                    placeholder = tf.compat.v1.placeholder(field_value.dtype)
                else:
                    raise TypeError("Unknown ground truth label field type")

                kwargs_bbox2dlabel[field_name] = placeholder

            placeholders = Bbox2DLabel(**kwargs_bbox2dlabel)
        else:
            raise TypeError("Unknown ground truth label type")

        return placeholders

    def _get_ground_truth_labels_feed_dict(self, ground_truth_labels):
        """Construct a feed dict for ground truth placeholders given a list of dicts of labels.

        Returns:
            feed_dict (dict): Keys are placeholders for the ground truth label features, values
                are the corresponding feature values.
        """
        feed_dict = dict()

        if isinstance(ground_truth_labels, list):
            for values, tensors in zip(ground_truth_labels, self._ground_truth_labels_placeholders):
                for name, tensor in six.iteritems(tensors):
                    feed_dict[tensor] = values[name]
        elif isinstance(ground_truth_labels, Bbox2DLabel):
            label_dict = ground_truth_labels._asdict()
            placeholders_dict = self._ground_truth_labels_placeholders._asdict()

            for field_name in Bbox2DLabel._fields:
                placeholder = placeholders_dict[field_name]
                field_value = label_dict[field_name]
                if field_name == 'vertices':
                    feed_dict[placeholder.coordinates] = field_value.coordinates
                    feed_dict[placeholder.canvas_shape.width] = field_value.canvas_shape.width
                    feed_dict[placeholder.canvas_shape.height] = field_value.canvas_shape.height
                else:
                    feed_dict[placeholder] = field_value
        else:
            raise TypeError("Unknown ground truth label type")

        return feed_dict

    @property
    def ground_truth_labels(self):
        """Return labels placeholders to be used for validation."""
        return self._ground_truth_labels_placeholders

    @staticmethod
    def get_session_config():
        """Constrain TensorFlow to use CPU to avoid conflicting with TensorRT on the GPU.

        Returns:
            Tensorflow session config.
        """
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.33)
        session_config = tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            device_count={'GPU': 0, 'CPU': 1})
        return session_config

    def _get_validation_iterator(self, session, dataset_percentage=100.0):
        """Generator that yields batch predictions, labels, and cost.

        Args:
            session (tf.Session): Session to be used for evaluation.
            dataset_percentage (float): % of the dataset to evaluate.

        Returns:
            predictions_batch: Raw predictions for current batch.
            gt_batch: List of ground truth labels dicts for current batch.
            batch_val_cost: Validation cost for current batch.
            inference_time (float): Inference time for one image
        """
        num_steps = int(self._steps * dataset_percentage / 100.0)
        log_steps = 10
        prev_start = default_timer()
        for step in xrange(num_steps):
            im_batch, gt_batch = \
                session.run([self._images, self._ground_truth_labels])

            feed_dict = dict()

            ground_truth_labels_feed_dict = \
                self._get_ground_truth_labels_feed_dict(
                    gt_batch)
            predictions_feed_dict = self.gridbox.get_predictions_feed_dict(
                im_batch)

            feed_dict.update(ground_truth_labels_feed_dict)
            feed_dict.update(predictions_feed_dict)

            start = default_timer()
            if (step % log_steps) == 0:
                logger.info("step %d / %d, %.2fs/step" %
                            (step, num_steps, (start-prev_start)/log_steps))
                prev_start = start
            predictions_batch, batch_val_cost = \
                session.run(self.gridbox.get_validation_tensors(),
                            feed_dict=feed_dict)
            end = default_timer()
            batch_size = len(list(predictions_batch.values())[0]['bbox'])
            inference_time = (end - start) / batch_size

            yield predictions_batch, gt_batch, batch_val_cost,\
                inference_time
