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

"""Postprocess for Detections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.cluster import cluster_predictions


def _bbox_xywh_image(bbox, image_size):
    """Convert bbox from LTRB to normalized XYWH.

    Arguments:
        bbox: Bbox in LTRB format.
        image_size: Range of bbox coordinates.
    Returns:
        Bbox in XYWH format, normalized to [0,1] range.
    """
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x /= float(image_size[0])
    y /= float(image_size[1])
    w /= float(image_size[0])
    h /= float(image_size[1])
    return x, y, w, h


def detections_to_confidence_model_input(detections, image_size):
    """Construct an input batch of detections.

    Arguments:
        detections: A list of Detections.
        image_size: Detection bbox resolution as tuple (width, height).
    Returns:
        A list of confidence model input vectors.
    """
    detection_tensors = []
    for sample in detections:
        for detection in sample:
            bbox_x, bbox_y, bbox_width, bbox_height = _bbox_xywh_image(detection.bbox, image_size)
            det = [detection.confidence,
                   detection.bbox_variance,
                   bbox_height,
                   bbox_width,
                   detection.num_raw_bboxes,
                   bbox_x,
                   bbox_y]
            detection_tensors.append(np.array(det))

    return detection_tensors


def _patch_detections(detections, confidences):
    """Reconstruct Detections with the computed confidence values.

    Arguments:
        detections: A list of list of Detections.
        confidences: A list of confidence values, one for each Detection.
    Returns:
        A list of list of Detections with patched confidence values.
    """
    index = 0
    updated_detections = []
    for sample in detections:
        updated_sample = []
        for detection in sample:
            updated_detection = \
                detection._replace(confidence=confidences[index][0])
            updated_sample.append(updated_detection)
            index = index + 1
        updated_detections.append(updated_sample)

    return updated_detections


def _filter_by_confidence(detections, confidence_threshold):
    """Filter list of detections by given confidence threshold.

    Args:
        detections (list): List of list of detections. Each outer list indexes frames, and each
            inner list contains the Detection instances for a given frame.
        confidence_threshold (float): Confidence threshold to use for filtering.

    Returns:
        filtered_detections (list): Filtered detections in the same format as detections.
    """
    filtered_detections = [list([det for det in detections_list if det.confidence >=
                           confidence_threshold]) for detections_list in detections]
    return filtered_detections


class PostProcessor(object):
    """Hold all the pieces of the DetectNet V2 postprocessing pipeline."""

    def __init__(self, postprocessing_config,
                 confidence_models=None, image_size=None):
        """Constructor.

        Args:
            postprocessing_config (dict): Each key is a target class name (str), and value a
                PostProcessingConfig object.
            confidence_models (dict): Each key is a target class name (str), and value a
                ConfidenceModel. Can be None.
            image_size (tuple): Dimensions of the input to the detector (width, height). If
                <confidence_models> are supplied, this must also be supplied.

        Raises:
            ValueError: If <confidence_models> are supplied, but <image_size> is not.
        """
        if confidence_models is not None:
            raise ValueError("PostProcessor: Confidence Model is currently not supported")
        self._postprocessing_config = postprocessing_config
        if confidence_models is None:
            self._confidence_models = dict()
        else:
            self._confidence_models = confidence_models
        self._image_size = image_size

    def cluster_predictions(self, predictions, postprocessing_config=None):
        """Cluster raw predictions into detections.

        Args:
            predictions (dict): Nested dictionary with structure [target_class_name][objective_name]
                and values the corresponding 4-D (N, C, H, W) np.array as produced by the detector.
                N is the number of images in a batch, C the number of dimension that objective has
                (e.g. 4 coordinates for 'bbox'), and H and W are the spatial dimensions of the
                detector's output.
            postprocessing_config (dict of PostProcessingConfigs): Dictionary of postprocessing
                parameters per class, which, if provided, override existing clustering parameters
                for this call only.

        Returns:
            detections (dict): Keys are target class names, values are lists of lists of Detection
                instances. Each outer list indexes frames, each inner list, the detections for that
                frame.
        """
        if postprocessing_config is None:
            postprocessing_config = self._postprocessing_config

        detections = cluster_predictions(predictions, postprocessing_config)

        return detections

    def postprocess_predictions(self, predictions, target_class_names,
                                postprocessing_config=None, session=None):
        """Cluster predictions into Detections.

        Optionally apply confidence models, and filter by confidence.

        Args:
            predictions (dict): Nested dictionary with structure [target_class_name][objective_name]
                and values the corresponding 4-D (N, C, H, W) np.array as produced by the detector.
                N is the number of images in a batch, C the number of dimension that objective has
                (e.g. 4 coordinates for 'bbox'), and H and W are the spatial dimensions of the
                detector's output.
            target_class_names (list): A list of target class names.
            postprocessing_config (dict of PostProcessingConfigs): Dictionary of postprocessing
                parameters per class, which, if provided, override existing clustering parameters
                for this call only.
            session (tf.Session): A session for confidence model inference. If
                `self._confidence_models` is not None, this must also be supplied.

        Returns:
            detections (dict): Keys are target class names, values are lists of lists of Detection
                instances. Each outer list indexes frames, each inner list, the detections for that
                frame.
        """
        detections = self.cluster_predictions(predictions, postprocessing_config)

        if self._confidence_models:
            detections = self.apply_confidence_models(
                detections=detections,
                session=session,
                target_class_names=target_class_names)

            # Now, filter by confidence.
            detections = self.filter_by_confidence(detections)

        return detections

    def filter_by_confidence(self, detections, confidence_threshold=None):
        """Filter list of detections by given confidence threshold.

        Args:
            detections (dict): Keys are target class names, values are lists of lists of Detection
                instances. Each outer list indexes frames, each inner list, the detections for that
                frame.
            confidence_threshold (float): Confidence threshold to use for filtering. Can be None.
                If not supplied, the one defined in `self._postprocessing_config` is used.

        Returns:
            filtered_detections (dict): Filtered detections in the same format as <detections>.
        """
        filtered_detections = dict()
        for target_class_name in detections:
            if confidence_threshold is None:
                confidence_threshold = self._postprocessing_config[target_class_name].\
                    confidence_config.confidence_threshold

            filtered_detections[target_class_name] = _filter_by_confidence(
                detections[target_class_name],
                confidence_threshold=confidence_threshold)

        return filtered_detections
