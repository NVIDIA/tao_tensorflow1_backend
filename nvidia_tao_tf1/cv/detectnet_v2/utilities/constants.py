# Copyright (c) 2017 - 2019, NVIDIA CORPORATION.  All rights reserved.

"""Defining all the constants and magic numbers that the iva gridbox module uses."""

from collections import namedtuple

# Global Variables
# Setting output color book
color = {
    'car': 'green',
    'road_sign': 'cyan',
    'bicycle': 'yellow',
    'person': 'magenta',
    'heavy_truck': 'blue',
    'truck': 'red',
    'face': 'white'
}

# Setting output label color map for kitti dumps
output_map = {
    'car': 'automobile',
    'person': 'person',
    'bicycle': 'bicycle',
    'road_sign': 'road_sign',
    'face': 'face'
}

output_map_sec = {
    'car': 'automobile',
    'person': 'person',
    'bicycle': 'bicycle',
    'road_sign': 'road_sign',
    'face': 'face'
}

# Clustering parameters
scales = [(1.0, 'cc')]
offset = (0, 0)
train_img_size = (960, 544)
criterion = 'IOU'

DEBUG = False
EPSILON = 1e-05

# Global variable for accepted image extensions
valid_image_ext = ['.jpg', '.png', '.jpeg', '.ppm']

Detection = namedtuple('Detection', [
    # Bounding box of the detection in the LTRB format: [left, top, right, bottom]
    'bbox',
    # Confidence of detection
    'confidence',
    # Weighted variance of the bounding boxes in this cluster, normalized for the size of the box
    'cluster_cv',
    # Number of raw bounding boxes that went into this cluster
    'num_raw_boxes',
    # Sum of of the raw bounding boxes' coverage values in this cluster
    'sum_coverages',
    # Maximum coverage value among bboxes
    'max_cov_value',
    # Minimum converage value among bboxes
    'min_cov_value',
    # Candidate coverages.
    'candidate_covs',
    # Candidate bbox coordinates.
    'candidate_bboxes'
])
