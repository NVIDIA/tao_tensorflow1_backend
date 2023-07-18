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
"""Data loader and processing test cases."""
import glob
import hashlib
import os
import numpy as np
from PIL import Image

import tensorflow as tf
tf.enable_eager_execution()

from nvidia_tao_tf1.cv.efficientdet.dataloader import dataloader
from nvidia_tao_tf1.cv.efficientdet.models import anchors
from nvidia_tao_tf1.cv.efficientdet.object_detection import tf_example_decoder
from nvidia_tao_tf1.cv.efficientdet.utils import hparams_config


def int64_feature(value):
    """int64_feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    """int64_list_feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """bytes_feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    """bytes_list_feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    """float_feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    """float_list_feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def test_dataloader(tmpdir):
    tf.random.set_random_seed(42)
    # generate dummy tfrecord
    image_height = 512
    image_width = 512
    filename = "dummy_example.jpg"
    image_id = 1

    full_path = os.path.join(tmpdir, filename)
    # save dummy image to file
    dummy_array = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    Image.fromarray(dummy_array, 'RGB').save(full_path)
    with open(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
        # encoded_jpg_b = bytearray(encoded_jpg)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = [0.25]
    xmax = [0.5]
    ymin = [0.25]
    ymax = [0.5]
    is_crowd = [False]
    category_names = [b'void']
    category_ids = [0]
    area = [16384]

    feature_dict = {
        'image/height':
            int64_feature(image_height),
        'image/width':
            int64_feature(image_width),
        'image/filename':
            bytes_feature(filename.encode('utf8')),
        'image/source_id':
            bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            bytes_feature(key.encode('utf8')),
        'image/encoded':
            bytes_feature(encoded_jpg),
        'image/caption':
            bytes_list_feature([]),
        'image/format':
            bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            float_list_feature(xmin),
        'image/object/bbox/xmax':
            float_list_feature(xmax),
        'image/object/bbox/ymin':
            float_list_feature(ymin),
        'image/object/bbox/ymax':
            float_list_feature(ymax),
        'image/object/class/text':
            bytes_list_feature(category_names),
        'image/object/class/label':
            int64_list_feature(category_ids),
        'image/object/is_crowd':
            int64_list_feature(is_crowd),
        'image/object/area':
            float_list_feature(area),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    # dump tfrecords
    tfrecords_dir = tmpdir.mkdir("tfrecords")
    dummy_tfrecords = str(tfrecords_dir.join('/dummy-001'))
    writer = tf.python_io.TFRecordWriter(str(dummy_tfrecords))
    writer.write(example.SerializeToString())
    writer.close()

    params = hparams_config.get_detection_config('efficientdet-d0').as_dict()
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    params['image_size'])
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
    example_decoder = tf_example_decoder.TfExampleDecoder(
        regenerate_source_id=params['regenerate_source_id'])
    tfrecord_path = os.path.join(tfrecords_dir, "dummy*")

    dataset = tf.data.TFRecordDataset(glob.glob(tfrecord_path))
    value = next(iter(dataset))
    reader = dataloader.InputReader(
        tfrecord_path, is_training=True,
        use_fake_data=False,
        max_instances_per_image=100)
    result = reader._dataset_parser(value, example_decoder, anchor_labeler,
                                    params)
    print(result[1])
    print(result[2])
    assert np.allclose(result[0][0, 0, :], [-2.1651785, -2.0357141, -1.8124998])
    assert len(result) == 10
