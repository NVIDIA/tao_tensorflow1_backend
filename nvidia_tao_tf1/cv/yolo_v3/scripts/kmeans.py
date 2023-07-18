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

"""Kmeans algorithm to select Anchor shape. @Jeffery <zeyuz@nvidia.com>."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np

from PIL import Image
import nvidia_tao_tf1.cv.common.logging.logging as status_logging


def build_command_line_parser(parser=None):
    '''build parser.'''
    if parser is None:
        parser = argparse.ArgumentParser(prog='kmeans', description='Kmeans to select anchors.')

    parser.add_argument(
        '-l',
        '--label_folders',
        type=str,
        required=True,
        nargs='+',
        help='Paths to label files')
    parser.add_argument(
        '-i',
        '--image_folders',
        type=str,
        required=True,
        nargs='+',
        help='Paths to image files, must match order of label_folders')
    parser.add_argument(
        '-x',
        '--size_x',
        type=int,
        required=True,
        help='Network input width'
    )
    parser.add_argument(
        '-y',
        '--size_y',
        type=int,
        required=True,
        help='Network input height'
    )
    parser.add_argument(
        '-n',
        '--num_clusters',
        type=int,
        default=9,
        help='Number of clusters needed.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10000,
        help='maximum kmeans steps. Kmeans will stop even if not converged at max_steps'
    )
    parser.add_argument(
        '--min_x',
        type=int,
        default=0,
        help='ignore boxes with width (as in network input-size image) not larger than this value.'
    )
    parser.add_argument(
        '--min_y',
        type=int,
        default=0,
        help='ignore boxes with height (as in network input-size image) not larger than this value.'
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to the files where the logs are stored."
    )
    return parser


def parse_command_line(args):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def read_boxes(folders, img_folders, min_x, min_y):
    '''
    Read all boxes as two numpy arrays.

    Args:
        folders (list of strings): paths to kitti label txts.
        img_folders (list of strings): paths to kitti images.
        min_x (float): minimum x ratio
        min_y (float): minimum y ratio
    Returns:
        w (1-d array): widths of all boxes, 0-1 range
        h (1-d array): heights of all boxes, 0-1 range
    '''
    supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
    w = []
    h = []
    assert len(folders) == len(img_folders), "Labels and images folder must be 1-1 match"
    for idx, img_folder in enumerate(img_folders):
        for img_file in os.listdir(img_folder):
            fname, ext = os.path.splitext(img_file)
            if ext not in supported_img_format:
                continue

            label_file = os.path.join(folders[idx], fname+'.txt')
            if not os.path.isfile(label_file):
                print("Cannot find:", label_file)
                continue
            img_file = os.path.join(img_folder, img_file)
            img = Image.open(img_file)
            orig_w, orig_h = img.size

            lines = open(label_file, 'r').read().split('\n')
            for l in lines:
                l_sp = l.strip().split()
                if len(l_sp) < 15:
                    continue
                left = float(l_sp[4]) / orig_w
                top = float(l_sp[5]) / orig_h
                right = float(l_sp[6]) / orig_w
                bottom = float(l_sp[7]) / orig_h
                l_w = right - left
                l_h = bottom - top

                if l_w > min_x and l_h > min_y:
                    w.append(l_w)
                    h.append(l_h)

    return np.array(w), np.array(h)


def iou(w0, h0, w1, h1):
    '''
    Pairwise IOU.

    Args:
        w0, h0: Boxes group 0
        w1, h1: Boxes group 1
    Returns:
        iou (len(w0) rows and len(w1) cols): pairwise iou scores
    '''
    len0 = len(w0)
    len1 = len(w1)
    w0_m = w0.repeat(len1).reshape(len0, len1)
    h0_m = h0.repeat(len1).reshape(len0, len1)
    w1_m = np.tile(w1, len0).reshape(len0, len1)
    h1_m = np.tile(h1, len0).reshape(len0, len1)
    area0_m = w0_m * h0_m
    area1_m = w1_m * h1_m
    area_int_m = np.minimum(w0_m, w1_m) * np.minimum(h0_m, h1_m)

    return area_int_m / (area0_m + area1_m - area_int_m)


def kmeans(w, h, num_clusters, max_steps=1000):
    '''
    Calculate cluster centers.

    Args:
        w (1-d numpy array): 0-1 widths
        h (1-d numpy array): 0-1 heights
        num_clusters (int): num clusters needed
    Returns:
        cluster_centers (list of tuples): [(c_w, c_h)] sorted by area
    '''

    assert len(w) == len(h), "w and h should have same shape"
    assert num_clusters < len(w), "Must have more boxes than clusters"
    n_box = len(w)
    rand_id = np.random.choice(n_box, num_clusters, replace=False)
    clusters_w = w[rand_id]
    clusters_h = h[rand_id]

    # EM-algorithm
    cluster_assign = np.zeros((n_box,), int)

    for i in range(max_steps):
        # shape (n_box, num_cluster)
        if i % 10 == 0:
            print("Start optimization iteration:", i + 1)
        box_cluster_iou = iou(w, h, clusters_w, clusters_h)
        re_assign = np.argmax(box_cluster_iou, axis=1)
        if all(re_assign == cluster_assign):
            # converge
            break
        cluster_assign = re_assign
        for j in range(num_clusters):
            clusters_w[j] = np.median(w[cluster_assign == j])
            clusters_h[j] = np.median(h[cluster_assign == j])

    return sorted(zip(clusters_w, clusters_h), key=lambda x: x[0] * x[1])


def main(args=None):
    '''Main function.'''
    args = parse_command_line(args)
    # Set up status logging
    if args.results_dir:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        status_file = os.path.join(args.results_dir, "status.json")
        status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=True,
                verbosity=1,
                append=True
            )
        )
        s_logger = status_logging.get_status_logger()
        s_logger.write(
            status_level=status_logging.Status.STARTED,
            message="Starting k-means."
        )
    w, h = read_boxes(args.label_folders, args.image_folders,
                      float(args.min_x) / args.size_x, float(args.min_y) / args.size_y)
    results = kmeans(w, h, args.num_clusters, args.max_steps)
    print('Please use following anchor sizes in YOLO config:')
    anchors = []
    for x in results:
        print("(%0.2f, %0.2f)" % (x[0] * args.size_x, x[1] * args.size_y))
        anchors.append("(%0.2f, %0.2f)" % (x[0] * args.size_x, x[1] * args.size_y))
    if args.results_dir:
        s_logger.kpi.update({'k-means generated anchors': str(anchors)})
        s_logger.write(
            status_level=status_logging.Status.SUCCESS,
            message="K-means finished successfully."
        )


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="KMEANS was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
