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

"""Helper functions to generate data for tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image
from six.moves import range
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _bytes_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _float_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _int64_feature


def generate_dummy_images(directory, num_samples, height, width, num_channels):
    """Generate num_samples dummy images of given shape and store them to the given directory.

    Images will have the shape [H, W, 3] and the values are in [0, 255]. The images will be
    stored as 0.png, 1.png, etc.

    Args:
        directory: Directory where images are stored.
        num_samples: Number of images to generate.
        height, width: Image shape.
    """
    for i in range(num_samples):
        img_array = np.random.rand(height, width, 3) * 255
        im = Image.fromarray(img_array.astype('uint8')).convert('RGBA')
        if num_channels == 1:
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                bg_colour = (255, 255, 255)
                # Need to convert to RGBA if LA format due to a bug in PIL
                alpha = im.convert('RGBA').split()[-1]
                # Create a new background image of our matt color.
                # Must be RGBA because paste requires both images have the same format
                bg = Image.new("RGBA", im.size, bg_colour + (255,))
                bg.paste(im, mask=alpha)

            im = im.convert('L')
            # img = Image.fromarray(img_array.astype('uint8')).convert('L')
            # img = Image.fromarray(img_array)
            # img_conv = img.convert('L')
            # img = Image.fromarray(img_array.astype('uint8'))
        img_path = os.path.join(str(directory), '%d.png' % i)
        im.save(img_path)


def generate_dummy_labels(path, num_samples, height=0, width=0, labels=None):
    """Generate num_samples dummy labels and store them to a tfrecords file in the given path.

    Args:
        path: Path to the generated tfrecords file.
        num_samples: Labels will be generated for this many images.
        height, width: Optional image shape.
        labels: Optional, list of custom labels to write into the tfrecords file. The user is
            expected to provide a label for each sample. Each label is dictionary with the label
            name as the key and value as the corresponding tf.train.Feature.
    """
    if labels is None:
        labels = [{'target/object_class': _bytes_feature('car'),
                   'target/coordinates_x1': _float_feature(1.0),
                   'target/coordinates_y1': _float_feature(1.0),
                   'target/coordinates_x2': _float_feature(1.0),
                   'target/coordinates_y2': _float_feature(1.0),
                   'target/truncation': _float_feature(0.0),
                   'target/occlusion': _int64_feature(0),
                   'target/front': _float_feature(0.0),
                   'target/back': _float_feature(0.5),
                   'frame/id': _bytes_feature(str(i)),
                   'frame/height': _int64_feature(height),
                   'frame/width': _int64_feature(width)} for i in range(num_samples)]
    else:
        num_custom_labels = len(labels)
        assert num_custom_labels == num_samples, \
            "Expected %d custom labels, got %d." % (num_samples, num_custom_labels)

    writer = tf.python_io.TFRecordWriter(str(path))

    for label in labels:
        features = tf.train.Features(feature=label)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()


def generate_dummy_dataset(tmpdir, num_samples, height, width, labels=None, num_channels=3):
    """Construct a dummy dataset and return paths to it.

    Args:
        tmpdir (pytest tmpdir): Temporary folder under which the dummy dataset will be created.
        num_samples (int): Number of samples to use for the dataset.
        height (int): Height of the dummy images.
        width (int): Width for the dummy images.
        labels: Optional, list of custom labels to write into the tfrecords file. The user is
            expected to provide a label for each sample. Each label is a dictionary with the label
            name as the key and value as the corresponding tf.train.Feature.

    Returns:
        tfrecords_path (str): Path to generated tfrecords.
        image_directory_path (str): Path to generated images.
    """
    images_path = tmpdir.mkdir("images")
    labels_path = tmpdir.mkdir("labels")
    validation_tfrecords = str(labels_path.join('dummy-fold-000-of-002'))
    training_tfrecords = str(labels_path.join('dummy-fold-001-of-002'))

    generate_dummy_images(images_path, num_samples, height, width, num_channels)
    generate_dummy_labels(training_tfrecords, num_samples, height, width, labels=labels)
    generate_dummy_labels(validation_tfrecords, num_samples, height, width, labels=labels)

    tfrecords_path = str(labels_path.join('*'))
    image_directory_path = str(images_path)

    return tfrecords_path, image_directory_path
