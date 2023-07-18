"""Tests for bbox clustering using nms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from google.protobuf.text_format import Merge as merge_text_proto
import numpy as np
import pytest

from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.utilities import (
    cluster_with_hybrid,
    cluster_with_nms,
    setup_dbscan_object
)
from nvidia_tao_tf1.cv.detectnet_v2.proto.inference_pb2 import BboxHandlerConfig


bbox_handler_config = """
kitti_dump: true
disable_overlay: false
overlay_linewidth: 2
classwise_bbox_handler_config{
    key: "person"
    value: {
        confidence_model: "aggregate_cov"
        output_map: "person"
        bbox_color{
            R: 0
            G: 255
            B: 0
        }
        clustering_config{
            clustering_algorithm: HYBRID
            nms_iou_threshold: 0.3
            nms_confidence_threshold: 0.2
            coverage_threshold: 0.005
            dbscan_confidence_threshold: 0.9
            dbscan_eps: 0.3
            dbscan_min_samples: 1
            minimum_bounding_box_height: 4
        }
    }
}
"""

TEST_CLASS = "person"
cluster_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def traverse_up(file_path, num_levels=3):
    """Traverse root up by num_levels.

    Args:
        file_path (str): Source path to the file.
        num_levels (int): Number of levels to traverse up.

    Returns:
        file_path (str): Updated path moved up by num_levels.
    """
    for _ in range(num_levels):
        file_path = os.path.dirname(file_path)
    return file_path


detectnet_root = traverse_up(os.path.realpath(__file__))
test_fixture_root = os.path.join(
    detectnet_root,
    "postprocessor/tests/test_fixtures"
)
labels_dbscan_candidates = os.path.join(
    test_fixture_root,
    "labels_dbscan_cluster_candidates.txt"
)
labels_nms_output = os.path.join(
    test_fixture_root,
    "labels_nms_output.txt"
)
labels_raw = os.path.join(
    test_fixture_root,
    "labels_raw.txt"
)


def read_kitti_labels(label_file):
    """Parse kitti label files.

    Args:
        label_path (str): Path to the kitti label string.

    Returns:
        label_data (dict): Dictionary of classwise boxes and covs.
    """
    label_list = []
    if not os.path.exists(label_file):
        raise ValueError("Labelfile : {} does not exist".format(label_file))
    with open(label_file, 'r') as lf:
        for row in csv.reader(lf, delimiter=' '):
            label_list.append(row)
    lf.closed
    return label_list


def generate_test_fixture(label_list):
    """Generate a test fixture from kitti labels.

    Args:
        label_list (list): List of parsed kitti labels.

    Returns:
        dict: bboxes and coverages formatted for the output.
    """
    bboxes = []
    coverages = []
    for obj in label_list:
        if obj[0].lower() == TEST_CLASS:
            bboxes.append([float(coord) for coord in obj[4:8]])
            coverages.append(float(obj[-1]))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    coverages = np.asarray(coverages, dtype=np.float32)
    return {"bboxes": bboxes, "coverages": coverages}


def load_bbox_handler_config(proto_string):
    """Read bbox handler prototxt."""
    bbox_handler_proto = BboxHandlerConfig()
    merge_text_proto(proto_string, bbox_handler_proto)
    return bbox_handler_proto


test_case_1 = {
    "raw_predictions": generate_test_fixture(read_kitti_labels(labels_raw)),
    "dbscan_candidates": generate_test_fixture(read_kitti_labels(labels_dbscan_candidates)),
    "nms_outputs": generate_test_fixture(read_kitti_labels(labels_nms_output)),
    "bbox_handler_spec": load_bbox_handler_config(bbox_handler_config)
}

test_data = [(test_case_1)]


@pytest.mark.parametrize(
    "test_fixt",
    test_data,
)
def test_dbscan_nms_hybrid(test_fixt):
    """Test hybrid clustering algorithm for detectnet inferences.

    Args:
        test_fixt (tuple): Tuple containing a dictionary of test cases.

    Returns:
        No explicit returns.
    """
    # Extract the text fixtures.
    b_config = test_fixt["bbox_handler_spec"]
    raw_predictions = test_fixt["raw_predictions"]
    dbscan_detections = test_fixt["dbscan_candidates"]
    classwise_bbox_handler_config = dict(b_config.classwise_bbox_handler_config)
    clustering_config = classwise_bbox_handler_config[TEST_CLASS].clustering_config
    confidence_model = classwise_bbox_handler_config[TEST_CLASS].confidence_model
    eps = clustering_config.dbscan_eps
    min_samples = clustering_config.dbscan_min_samples
    criterion = "IOU"

    # Setup dbscan clustering object.
    db = setup_dbscan_object(
        eps,
        min_samples,
        criterion
    )

    # Cluster bboxes using hybrid clustering.
    clustered_detections = cluster_with_hybrid(
        bboxes=raw_predictions["bboxes"],
        covs=raw_predictions["coverages"],
        criterion="IOU",
        db=db,
        confidence_model=confidence_model,
        cluster_weights=cluster_weights,
        min_height=clustering_config.minimum_bounding_box_height,
        nms_iou_threshold=clustering_config.nms_iou_threshold,
        confidence_threshold=clustering_config.dbscan_confidence_threshold,
        nms_confidence_threshold=clustering_config.nms_confidence_threshold
    )

    # Cluster dbscan candidates using NMS.
    nms_clustered_boxes_per_image = cluster_with_nms(
        dbscan_detections["bboxes"],
        dbscan_detections["coverages"],
        clustering_config.minimum_bounding_box_height,
        nms_iou_threshold=clustering_config.nms_iou_threshold,
        threshold=clustering_config.nms_confidence_threshold
    )

    # Check the number of bboxes output from the nms output
    assert len(clustered_detections) == len(test_fixt["nms_outputs"]["bboxes"])
    assert len(nms_clustered_boxes_per_image) == len(test_fixt["nms_outputs"]["bboxes"])
    output_bboxes = []
    for detection in clustered_detections:
        output_bboxes.append(detection.bbox)
    output_bboxes = np.asarray(output_bboxes).astype(np.float32)
    assert np.array_equal(output_bboxes, test_fixt["nms_outputs"]["bboxes"])
