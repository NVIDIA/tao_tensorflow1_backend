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

"""Apply clustering to prediction tensors and create Detection objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from sklearn.cluster import DBSCAN as dbscan

from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.detection import Detection
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.utilities import get_keep_indices


def cluster_predictions(predictions, postprocessing_config):
    """Cluster bounding boxes from raw predictions, with some other preprocessing options.

    Args:
        predictions: Nested dictionary of prediction tensors with the structure
            {'car': 'bbox': 4D tensor}
        postprocessing_config: A dict in which keys are target class names and values
            PostProcessingConfig objects.
    Returns:
        clustered_detections: A dict of list of lists, which contains all detections for each frame.
            Keys are target class names.
    Raises:
        AssertionError: When target_class does not exist in postprocessing_config.
    """
    clustered_detections = {}

    # Cluster each class separately.
    for target_class in predictions:

        def flatten_spatial(array):
            return array.reshape(array.shape[:-2] + (-1, ))

        # Grab coverage and absolute bbox predictions.
        prediction = {}
        for objective in predictions[target_class]:
            prediction[objective] = flatten_spatial(predictions[target_class][objective])
            assert prediction[objective].ndim == 3

        assert target_class in postprocessing_config
        class_clustering_config = postprocessing_config[target_class].clustering_config

        clustered_detections[target_class] = cluster_bboxes(
            target_class, prediction,
            class_clustering_config,
            algo=class_clustering_config.clustering_algorithm)

    return clustered_detections


def cluster_bboxes(target_class, raw_detections, clustering_config, algo="dbscan"):
    """
    Cluster bboxes with a clustering algorithm.

    Args:
        target_class (str):
        raw_detections: dictionary with keys:
          bbox: rectangle coordinates in absolute image space, (num_imgs, 4, H*W) array.
          cov: weights for the rectangles, (num_imgs, 1, H*W) array.
          [other objectives similarly as the above]
        clustering_config: ClusteringConfig object.
        algo (str): The algorithm to be used for clustering.
            choices: "nms", "dbscan".

    Returns:
        detections_per_image: a list of lists of Detection objects, one list for each input frame.
    """
    db = None
    if algo == "dbscan":
        db = dbscan(
            eps=clustering_config.dbscan_eps,
            min_samples=max(int(clustering_config.dbscan_min_samples), 1),
            metric='precomputed'
        )

    num_images = len(raw_detections['cov'])
    # Initialize output detections to empty lists.
    detections_per_image = [[] for _ in range(num_images)]
    # Loop images.
    for image_idx in range(num_images):
        detection_data = threshold_data(
            raw_detections,
            clustering_config.coverage_threshold,
            image_idx
        )
        # make sure boxes exist after preliminary filtering.
        if detection_data is None:
            continue
        # Cluster boxes based on the clustering algorithm.
        if algo == "dbscan":
            detections_per_image[image_idx] += cluster_with_dbscan(
                detection_data,
                db,
                target_class,
                clustering_config.minimum_bounding_box_height,
                threshold=clustering_config.dbscan_confidence_threshold
            )
        elif algo == "nms":
            detections_per_image[image_idx] += cluster_with_nms(
                detection_data,
                target_class,
                clustering_config.minimum_bounding_box_height,
                nms_iou_threshold=clustering_config.nms_iou_threshold,
                confidence_threshold=clustering_config.nms_confidence_threshold)
        else:
            raise NotImplementedError(
                "Invalid clustering algorithm requested: {}".format(algo)
            )

    # Sort in descending order of confidence.
    detections_per_image = [sorted(image_detections, key=lambda det: -det.confidence)
                            for image_detections in detections_per_image]

    return detections_per_image


def cluster_with_dbscan(detection_data,
                        db,
                        target_class,
                        min_bbox_height,
                        threshold=0.01):
    """Clustering bboxes with DBSCAN.

    Args:
        detection_data (dict): Dictionary of thresholded predicitions.
        db (sklearn.dbscan): Scikit learn dbscan object:
        target_class (str): Target class string to compile clustered detections.
        min_bbox_height (float): Minimum height of a bbox to be considered a valid detection.

    Returns:
        detections_per_image (list): List of clustered detections per image.
    """
    detections_per_image = []
    # Compute clustering data.
    clustering_data = compute_clustering_data(detection_data)

    sample_weight_data = detection_data['cov'].flatten()
    labeling = db.fit_predict(
        X=clustering_data, sample_weight=sample_weight_data)

    # Ignore detections which don't belong to any cluster (i.e., noisy samples).
    labels = np.unique(labeling[labeling >= 0])

    for label in labels:
        detection_indices = labeling == label
        detection = create_detection(
            target_class, detection_data, detection_indices)
        # Filter out too small bboxes.
        if bbox_height_image(detection.bbox) <= min_bbox_height:
            continue
        if detection.confidence < threshold:
            continue
        detections_per_image += [detection]
    return detections_per_image


def cluster_with_nms(detection_data,
                     target_class,
                     min_bbox_height,
                     nms_iou_threshold=0.2,
                     confidence_threshold=0.01):
    """Clustering raw detections with NMS."""
    bboxes = detection_data["bbox"]
    covs = detection_data["cov"][:, 0]
    keep_indices = get_keep_indices(bboxes, covs, min_bbox_height,
                                    Nt=nms_iou_threshold,
                                    threshold=confidence_threshold)
    if keep_indices.size == 0:
        return []
    filterred_boxes = np.take_along_axis(bboxes, keep_indices, axis=0)
    filterred_coverages = covs[keep_indices]
    assert filterred_boxes.shape[0] == filterred_coverages.shape[0], (
        "The number of boxes and covs after filtering must be the same: "
        "{} != {}".format(filterred_boxes.shape[0], filterred_coverages.shape[0])
    )
    clustered_boxes_per_image = []
    for idx in range(len(filterred_boxes)):
        clustered_boxes_per_image.append(Detection(
            class_name=target_class,
            bbox_variance=None,
            num_raw_bboxes=None,
            bbox=filterred_boxes[idx, :],
            confidence=filterred_coverages[idx][0],
            cov=filterred_coverages[idx][0]))
    return clustered_boxes_per_image


def threshold_data(raw_detections, coverage_threshold, image_idx):
    """Threshold output detections based on clustering_config.

    Args:
        raw_detections (dict): Dictionary of raw predictions.
        coverage_threshold (float): Minimum confidence in the cov blob
            to filter bboxes.
        image_idx (int): Id of the image in the batch being processed.

    Returns:
        detection_data (dict): Dictionary of thresholded predictions per image.
    """
    covs = raw_detections['cov'][image_idx][0]
    # Check if the input was empty.
    if not covs.size:
        return None
    # Discard too low coverage detections.
    valid_indices = covs > coverage_threshold
    if not valid_indices.any():
        # Filtered out everything, continue.
        return None

    # Filter and reshape bbox data so that clustering data can be calculated.
    detection_data = {}
    for objective in raw_detections:
        detection_data[objective] = raw_detections[objective][image_idx][:,
                                                                         valid_indices].T
    return detection_data


def compute_clustering_data(detection_data):
    """
    Compute data required by the clustering algorithm.

    Args:
        detection_data: Values for bbox coordinates in the image plane.
    Returns:
        clustering_data: Numpy array which contains data for the clustering algorithm to use.
    """
    clustering_data = 1.0 - compute_iou(detection_data['bbox'])

    return clustering_data


def bbox_height_image(bbox):
    """Height of an bbox in (x1, y1, x2, y2) or LTRB format on image plane."""
    return bbox[3] - bbox[1]


def compute_iou(rectangles):
    """Intersection over union (IOU) among a list of rectangles in (x1, y1, x2, y2) format.

    Args:
      rectangles: numpy array of shape (N, 4), (x1, y1, x2, y2) format, assumes x1 < x2, y1 < y2
    Returns:
      iou: numpy array of shape (N, N) of the IOU between all pairs of rectangles
    """
    # Get coordinates
    x1, y1, x2, y2 = rectangles.T

    # Form intersection coordinates
    intersection_x1 = np.maximum(x1[:, None], x1[None, :])
    intersection_y1 = np.maximum(y1[:, None], y1[None, :])
    intersection_x2 = np.minimum(x2[:, None], x2[None, :])
    intersection_y2 = np.minimum(y2[:, None], y2[None, :])

    # Form intersection areas
    intersection_width = np.maximum(0, intersection_x2 - intersection_x1)
    intersection_height = np.maximum(0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    # Original rectangle areas
    areas = (x2 - x1) * (y2 - y1)

    # Union area is area_a + area_b - intersection area
    union_area = (areas[:, None] + areas[None, :] - intersection_area)

    # Return IOU regularized with a small constant to avoid outputing NaN in pathological
    # cases (area_a = area_b = isect = 0)
    iou = intersection_area / (union_area + 1e-5)

    return iou


def _bbox_area_image(bbox):
    """Bounding box area for LTRB image plane bboxes."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def mean_angle(angles, weights=None):
    """
    Compute the (weighted) average of given angles.

    The average is computed taking wrap-around into account. If weights are given, compute a
    weighted average.

    Args:
        angles: The angles in radians
        weights: The corresponding weights

    Returns: The mean angle
    """
    if weights is None:
        # Note: since np.arctan2 does an element wise quotient, the weights need not sum to 1.0.
        weights = np.ones_like(angles)

    cos_sum = np.sum(np.cos(angles) * weights)
    sin_sum = np.sum(np.sin(angles) * weights)

    return np.arctan2(sin_sum, cos_sum)


def create_detection(target_class, detection_data, detection_indices):
    """Create a detection based on grid cell indices which belong to the same cluster.

    Confidence of the detection is the sum of coverage values and bbox coordinates are the
    weighted mean of the bbox coordinates in the grid cell indices.

    Args:
        target_class (str):
        detection_data: Values for bbox coordinates.
        detection_indices: Indices part of this detection.
    Returns:
        detection: Detection object that defines a detection.
    """
    cluster = {}
    for objective in detection_data:
        cluster[objective] = detection_data[objective][detection_indices]

    w = cluster['cov']
    n = len(w)

    # Sum of coverages and normalized coverages.
    aggregated_w = np.sum(w)
    w_norm = w / aggregated_w

    # Cluster mean.
    cluster_mean = {}
    for objective in detection_data:
        if objective == 'orientation':
            cluster_mean[objective] = mean_angle(cluster[objective], w_norm)
        elif objective == 'cov':
            cluster_mean[objective] = aggregated_w / n
        else:
            cluster_mean[objective] = np.sum((cluster[objective]*w_norm), axis=0)

    # Compute coefficient of variation of the box coords.
    bbox_area = _bbox_area_image(cluster_mean['bbox'])
    # Clamp to epsilon to avoid division by zero.
    epsilon = 0.001
    if bbox_area < epsilon:
        bbox_area = epsilon

    # Calculate weighted bounding box variance normalized by
    # bounding box size.
    bbox_variance = np.sum(w_norm.reshape((-1, 1)) * (cluster['bbox'] - cluster_mean['bbox']) ** 2,
                           axis=0)
    bbox_variance = np.sqrt(np.mean(bbox_variance) / bbox_area)

    detection = Detection(
        class_name=target_class,
        confidence=aggregated_w,
        bbox_variance=bbox_variance,
        num_raw_bboxes=n,
        **cluster_mean
    )

    return detection
