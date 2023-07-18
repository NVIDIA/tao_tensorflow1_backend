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
"""FPE DataIO pipeline script which generates tfrecords."""

import argparse
import json
import os
from time import time
import cv2
import numpy as np
import tensorflow as tf
from yaml import load


# Color definition for stdout logs.
CRED = '\033[91m'
CGREEN = '\033[92m'
CYELLOW = '\033[93m'
CEND = '\033[0m'


def _bytes_feature(value):
    '''
    Returns a bytes_list from a string / byte.

    Args:
        value (str): String value.
    Returns:
        Bytes list.
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    '''
    Returns a float_list from a float / double.

    Args:
        value (float): Float value.
    Returns:
        Float list.
    '''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    '''
    Returns an int64_list from a bool / enum / int / uint.

    Args:
        value (int64): Int64 value.
    Returns:
        Float list.
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_array(value):
    '''
    Returns an int64_list from an array.

    Args:
        value (ndarray): Numpy nd array.
    Returns:
        int64 list.
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _dtype_feature(ndarray):
    '''
    Returns an float_list from an ndarray.

    Args:
        value (ndarray): Numpy nd array.
    Returns:
        Float list.
    '''
    assert isinstance(ndarray, np.ndarray)
    return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray))


def parse_json_contents(jsonFile, args):
    '''
    Function to read ground truth json file.

    Args:
        jsonFile (str): Path of json file.
        args (dict): User arguments.
    Returns:
        dataset (list): list of samples, sample{img_path, landmarks, occ}.
    '''
    num_keypoints = args['num_keypoints']
    json_data = json.loads(open(jsonFile, 'r').read())
    dataset = list()
    for img in json_data:
        sample = dict()
        sample['img_path'] = ''
        sample['landmarks'] = np.zeros((num_keypoints, 2))
        sample['occlusions'] = np.zeros((num_keypoints, 1))
        try:
            fname = str(img['filename'])

            if not os.path.exists(os.path.join(args['image_root_path'], fname)):
                print(CRED + 'Image does not exist: {}'.format(fname) + CEND)
                continue

            # Start collecting points information from the json file.
            x = [0] * num_keypoints
            y = [0] * num_keypoints
            # Occlusion tags
            tags = [0] * num_keypoints

            for chunk in img['annotations']:
                if 'fiducialpoints' not in chunk['class'].lower():
                    continue

                points_data = (point for point in chunk if ('class' not in point and
                                                            'version' not in point))
                for point in points_data:
                    number = int(
                        ''.join(c for c in str(point) if c.isdigit()))
                    if 'x' in str(point).lower() and number <= num_keypoints:
                        x[number - 1] = str(int(float(chunk[point])))
                    if 'y' in str(point).lower() and number <= num_keypoints:
                        y[number - 1] = str(int(float(chunk[point])))
                    if 'occ' in str(point).lower() and number <= num_keypoints and chunk[point]:
                        tags[number - 1] = 1

                sample = dict()
                sample['img_path'] = fname
                sample['landmarks'] = np.asarray([x, y]).T
                sample['occlusions'] = np.asarray(tags).T
                dataset.append(sample)
        except Exception as e:
            print(CRED + str(e) + CEND)
    return dataset


def get_bbox(x1, y1, x2, y2):
    '''
    Function to get normalized boundiung box.

    This module makes the bounding box sqaure by
    increasing the lower of the bounding width and height.
    Args:
        x1 (int): x_min value of bbox.
        y1 (int): y_min value of bbox.
        x2 (int): x_max value of bbox.
        y2 (int): y_max value of bbox.
    Returns:
        Normalized bounding box coordinates in form [x1, y1, x2, y2].
    '''
    x_start = int(np.floor(x1))
    x_end = int(np.ceil(x2))
    y_start = int(np.floor(y1))
    y_end = int(np.ceil(y2))

    width = np.ceil(x_end - x_start)
    height = np.ceil(y_end - y_start)

    if width < height:
        diff = height - width
        x_start -= (np.ceil(diff/2.0))
        x_end += (np.floor(diff/2.0))
    elif width > height:
        diff = width - height
        y_start -= (np.ceil(diff/2.0))
        y_end += (np.floor(diff/2.0))

    width = x_end - x_start
    height = y_end - y_start
    assert width == height
    rect_init_square = [int(x_start), int(y_start), int(width), int(height)]
    return rect_init_square


def enlarge_bbox(bbox, ratio=1.0):
    '''
    Module enlarges the bounding box by a scaling factor.

    Args:
        bbox (list): Bounding box coordinates of the form [x1, y1, x2, y2].
        ratio (float): Bounding box enlargement scale/ratio.
    Returns:
        Scaled bounding box coordinates.
    '''
    x_start, y_start, width, height = bbox
    x_end = x_start + width
    y_end = y_start + height
    assert width == height, 'width %s is not equal to height %s'\
        % (width, height)
    change = ratio - 1.0
    shift = int((change/2.0)*width)
    x_start_new = int(np.floor(x_start - shift))
    x_end_new = int(np.ceil(x_end + shift))
    y_start_new = int(np.floor(y_start - shift))
    y_end_new = int(np.ceil(y_end + shift))

    # Assertion for increase length.
    width = int(x_end_new - x_start_new)
    height = int(y_end_new - y_start_new)
    assert height == width
    rect_init_square = [x_start_new, y_start_new, width, height]
    return rect_init_square


def detect_bbox(kpts, img_size, dist_ratio, num_kpts=68):
    '''
    Utility to get the bounding box using only kpt information.

    This method gets the kpts and the original image size.
    Then, it then gets a square encompassing all key-points and
    later enlarges that by dist_ratio.
    Args:
        kpts: the kpts in either format of 1-dim of size #kpts * 2
            or 2-dim of shape [#kpts, 2].
        img_size: a 2-value tuple indicating the size of the original image
                with format (width_size, height_size)
        dist_ratio: the ratio by which the original key-points to be enlarged.
        num_kpts (int): Number of keypoints.
    Returns:
        bbox with values (x_start, y_start, width, height).
    '''
    x_min = np.nanmin(kpts[:, 0])
    x_max = np.nanmax(kpts[:, 0])
    y_min = np.nanmin(kpts[:, 1])
    y_max = np.nanmax(kpts[:, 1])

    bbox = get_bbox(x_min, y_min, x_max, y_max)
    # Enlarge the bbox by a ratio.
    rect = enlarge_bbox(bbox, dist_ratio)

    # Ensure enlarged bounding box within image bounds.
    if((bbox[0] < 0) or
       (bbox[1] < 0) or
       (bbox[2] + bbox[0] > img_size[0]) or
       (bbox[3] + bbox[1] > img_size[1])):
        return None

    return rect


def write_tfrecord(dataset, setid, args):
    '''
    Utility to dump tfrecords with all data.

    Args:
        dataset (list): list of samples, sample{img_path, landmarks, occ}.
        setid (str): Set name.
        args (dict): User provided arguments.
    Returns:
        None
    '''
    tfRecordPath = os.path.join(args['save_root_path'],
                                args['save_path'],
                                setid,
                                args['tfrecord_folder'])

    if not os.path.exists(tfRecordPath):
        os.makedirs(tfRecordPath)
    recordfile = os.path.join(tfRecordPath, args['tfrecord_name'])

    writer = tf.io.TFRecordWriter(recordfile)

    N = len(dataset)
    count = 0
    for i in range(N):

        img_name = dataset[i]['img_path']

        landmarks = dataset[i]['landmarks'].astype('float')
        landmarks_occ = dataset[i]['occlusions'].astype(int)

        image_path = os.path.join(args['image_root_path'], img_name)
        image = cv2.imread(image_path)
        if image is None:
            print(CRED + 'Bad image:{}'.format(image_path) + CEND)
            continue

        image_shape = image.shape

        bbox = detect_bbox(kpts=landmarks[:args['num_keypoints'], :],
                           img_size=(image_shape[1], image_shape[0]),
                           dist_ratio=args['bbox_enlarge_ratio'],
                           num_kpts=args['num_keypoints'])
        if bbox is None:
            continue

        feature_dict = {
            'train/image_frame_name' : _bytes_feature(img_name.encode()),
            'train/image_frame_width' : _int64_feature(image_shape[1]),
            'train/image_frame_height' : _int64_feature(image_shape[0]),
            'train/facebbx_x' : _int64_feature(bbox[0]),
            'train/facebbx_y' : _int64_feature(bbox[1]),
            'train/facebbx_w' : _int64_feature(bbox[2]),
            'train/facebbx_h' : _int64_feature(bbox[3]),
            'train/landmarks' : _dtype_feature(landmarks.reshape(-1)),
            'train/landmarks_occ' : _int64_feature_array(landmarks_occ.reshape(-1))
        }

        example = tf.train.Example(
            features=tf.train.Features(feature=feature_dict))
        writer.write(example.SerializeToString())
        count = count + 1
    print(CYELLOW + 'recordtype:{} count: {}'.format(recordfile, count) + CEND)
    writer.close()


def tfrecord_manager(args):
    '''
    Function to read json files for all sets and create tfrecords.

    Args:
        args (dict): User provided arguments.
            - "sets": Set IDs to extract as a list. Example- [set1, set2, set3].
            - "gt_path": Ground truth json path.
            - "save_path": Save path for TF Records.
            - "gt_root_path": Root path for ground truth jsons.
            - "save_root_path": Root path for saving tfrecords data.
            - "image_root_path": Root path for the images.
            - "tf_folder": TF record folder name.
            - "tfrecord_name": TF record file name.
            - "num_keypoints": Number of keypoints.
            - "bbox_enlarge_ratio": Scale to enlarge bounding box with.
    Returns:
        None
    '''
    for setid in args['sets']:
        now = time()
        set_gt_path = os.path.join(args['gt_root_path'], args['gt_path'], setid)
        jsonList = []
        for x in os.listdir(set_gt_path):
            if('.json' in x):
                jsonList.append(x)
        # collect data from all GT json files for the setid
        dataset = list()
        for jsonfile in jsonList:
            jsonPath = os.path.join(set_gt_path, jsonfile)
            jsondata = parse_json_contents(jsonPath, args)
            print(CGREEN + 'Json {} has image count: {}'.format(jsonPath, len(jsondata)) + CEND)
            dataset.extend(jsondata)
        write_tfrecord(dataset, setid, args)
        print(CGREEN + 'Set {} has total image count: {}'.format(setid, len(dataset)) + CEND)
        set_time = round(time() - now, 2)
        print(CGREEN + 'DataIO for {} done in {} sec.'.format(setid, str(set_time)) + CEND)


def main():
    '''Main function to parse use arguments and call tfrecord manager.'''

    parser = argparse.ArgumentParser(
            description="Generate TFRecords from json ground truth")
    parser.add_argument('-e', '--exp_spec',
                        type=str, required=True,
                        help='Config file with dataio inputs.')
    args, _ = parser.parse_known_args()

    config_path = args.exp_spec
    with open(config_path, 'r') as f:
        args = load(f)

    tfrecord_manager(args)


if __name__ == '__main__':
    main()
