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

"""
Utility functions for data loading.

These can be used independently of the dataloader classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import backend as K
import nvidia_tao_tf1.core
import six
from six.moves import zip
import tensorflow as tf


def get_data_root():
    """Get path to dataset base directory."""
    # In case DATA_BASE_PATH environment variable is not set, use ~/datasets
    FALLBACK_DIR = os.path.join(os.path.expanduser('~'), 'datasets')
    data_root = os.getenv('DATA_BASE_PATH', FALLBACK_DIR)

    return data_root


def get_absolute_data_path(path):
    """Get absolute path from path or paths relative to DATA_BASE_PATH."""
    dataset_path = get_data_root()
    # Note that os.path.join ignores dataset_path if tf_records_path or image_directory_path
    # are absolute.
    if isinstance(path, list):
        return [os.path.join(dataset_path, p) for p in path]
    return os.path.join(dataset_path, path)


class ImageTFRecordsIterator(nvidia_tao_tf1.core.processors.TFRecordsIterator):
    """
    Processor that sets up a TFRecordsDataset and yields (value, image_dir) tuples from the input.

    Extends nvidia_tao_tf1.core.processors.TFRecordsIterator.
    """

    def call(self):
        """call method.

        Returns:
            tuple (records, image_dirs) where
            records: a list or dense tensor (depending on the value of ``batch_as_list``) containing
                the next batch as yielded from the `TFRecordsDataset`. Each new call will pull a
                fresh batch of samples. The set of input records cannot be depleted, as the records
                will wrap around to the next epoch as required.
                If the iterator reaches end of dataset, reinitialize the iterator
            image_dirs: a list of tf.string Tensors specifying image directory path for each of the
                records.
        """
        records, image_dirs, source_weights = self.iterator.get_next()
        records = self.process_records(records)
        image_dirs = self.process_records(image_dirs)
        source_weights = self.process_records(source_weights)
        return (records, image_dirs, source_weights)


def _create_tf_dataset(file_list, img_dir_list, source_weights_list):
    """
    Helper function to create a TFRecordDataset to be passed to the iterator.

    Args:
        file_list: (list) List of lists containing the files for each data source.
        img_dir_list: (list) List of lists of image directory paths.

    Returns:
        datasets: (tf.data.TFRecordDataset) a concatenation of possibly multiple datasets.
    """
    # Prepare the dataset to be passed to the iterator.
    assert len(file_list) == len(img_dir_list),\
        "Lengths of lists don't match!" \
        "`file_list` and `imge_dir_list` should have the same lengths, but instead "\
        "are {} and {}.".format(len(file_list), len(img_dir_list))
    datasets = None
    for files, image_dir, data_weight in zip(file_list, img_dir_list, source_weights_list):
        dataset = tf.data.TFRecordDataset(files)

        if not image_dir.endswith('/'):
            image_dir += '/'

        def add_image_dir(serialized_example, img_dir=image_dir, source_weight=data_weight):
            return (serialized_example, tf.constant(img_dir),
                    tf.constant(source_weight, tf.float32))

        dataset = dataset.map(add_image_dir)

        if datasets is None:
            datasets = dataset
        else:
            datasets = datasets.concatenate(dataset)

    return datasets


def get_num_samples(data_sources, training):
    """Get number of samples in the training/validation dataset.

    Args:
        data_sources: (list) List of tuples (list_of_tfrecord_files, image_directory_path)
        training: (boolean)
    Returns:
        num_samples: Number of samples in the training/validation dataset.
    """
    # file_list is actually a list of lists
    for data_source in data_sources:
        if not isinstance(data_source, tuple):
            assert data_source.dataset_type == "tfrecord"
    dataset_types = {type(data_source) for data_source in data_sources}
    assert len(dataset_types) == 1, (
        "All the data source types should either of the same type. Here we got {}".
        format(dataset_types)
    )

    if tuple in dataset_types:
        file_list = [data_source[0] for data_source in data_sources]
    else:
        file_list = [data_source.dataset_files for data_source in data_sources]

    num_samples = sum([sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(filename))
                      for tfrecords_files in file_list for filename in tfrecords_files])

    return num_samples


def get_tfrecords_iterator(data_sources, batch_size, training, repeat):
    """Get TFRecordsIterator for given .tfrecords files.

    Args:
        data_sources: (list) List of tuples (list_of_tfrecord_files, image_directory_path)
        batch_size: (int)
        training: (boolean)
        repeat (boolean): Allow looping over the dataset multiple times.
    Returns:
        tfrecords_iterator:
        num_samples: Number of samples in the training/validation dataset.
    """
    # file_list is actually a list of lists
    for data_source in data_sources:
        if not isinstance(data_source, tuple):
            assert data_source.dataset_type == "tfrecord"

    dataset_types = {type(data_source) for data_source in data_sources}
    assert len(dataset_types) == 1, (
        "All the data source types should either of the same type. Here we got {}".
        format(dataset_types)
    )
    if tuple in dataset_types:
        file_list = [data_source[0] for data_source in data_sources]
        img_dir_list = [data_source[1] for data_source in data_sources]
        source_weight_list = [1.0 for data_source in data_sources]
    else:
        file_list = [data_source.dataset_files for data_source in data_sources]
        img_dir_list = [data_source.images_path for data_source in data_sources]
        source_weight_list = [data_source.source_weight for data_source in data_sources]

    # Shuffle samples if training.
    shuffle = training

    num_samples = get_num_samples(data_sources, training)

    # Samples in the buffer can be shuffled. However, memory consumption increases with larger
    # buffer size.
    shuffle_buffer_size = num_samples if shuffle else 0

    tfrecords_iterator = ImageTFRecordsIterator(file_list=None,  # Using our own dataset object.
                                                shuffle_buffer_size=shuffle_buffer_size,
                                                batch_size=batch_size,
                                                batch_as_list=True,
                                                sequence_length=0,
                                                repeat=repeat,
                                                shuffle=shuffle
                                                )

    # Pass in our own dataset, instead of using the default one from a file list.
    dataset = _create_tf_dataset(file_list, img_dir_list, source_weight_list)
    tfrecords_iterator.build(dataset=dataset)

    return tfrecords_iterator, num_samples


def process_image_for_dnn_input(image):
    """
    Process the image in such a way that a DNN expects.

    This is needed because augmentation ops expect HWC, but DNN models expect CHW.

    Args:
        image: A representation of the image in HWC order.

    Returns:
        transposed: The input image in CHW order.
    """
    # DNN input data type has to match the computation precision.
    # A more appropriate place for this would be in the read_image function but
    # currently our spatial augmentations require and return FP32 images, hence this needs
    # be done after augmentation.
    image = tf.cast(image, dtype=K.floatx())
    return tf.transpose(image, (2, 0, 1))


def extract_tfrecords_features(tfrecords_file):
    """Extract features in a tfrecords file for parsing a series of tfrecords files."""
    tfrecords_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecords_file)

    # Each record is assumed to contain all the features even if the features would contain no
    # values.
    record = next(tfrecords_iterator)

    extracted_features = dict()

    example = tf.train.Example()
    example.ParseFromString(record)

    features = example.features.feature

    for feature_name, feature in six.iteritems(features):
        if feature.HasField('int64_list'):
            feature_dtype = tf.int64
        elif feature.HasField('float_list'):
            feature_dtype = tf.float32
        elif feature.HasField('bytes_list'):
            feature_dtype = tf.string
        else:
            raise RuntimeError("Unknown tfrecords feature kind: %s" % feature)

        # The tf.train.Example does not contain information about the FixedLen/VarLen
        # distinction. Use VarLenFeature for FixedLenFeatures.
        extracted_features[feature_name] = tf.io.VarLenFeature(dtype=feature_dtype)

    return extracted_features


# TODO(@williamz): add a test for this sucker and find out all the places that use it.
def get_target_class_to_source_classes_mapping(source_to_target_class_mapping):
    """Generate the mapping from target class names to a list of source class names.

    Args:
        source_to_target_class_mapping (dict): from source to target class names.

    Returns:
        target_class_to_source_classes_mapping (dict): maps from a target class name (str) to
            a list of source classes (str).
    """
    # Convert unicode strings to python strings for class mapping
    _source_to_target_class_mapping = \
        {str(k): str(v) for k, v in source_to_target_class_mapping.items()}

    target_class_names = set(_source_to_target_class_mapping.values())

    # Initialize empty lists.
    target_class_to_source_classes_mapping = \
        {target_class_name: [] for target_class_name in target_class_names}

    # Now populate them.
    for source_class_name, target_class_name in six.iteritems(_source_to_target_class_mapping):
        target_class_to_source_classes_mapping[target_class_name].append(source_class_name)

    return target_class_to_source_classes_mapping


def get_file_list_from_directory(path):
    """Finds all files from path.

    Args:
        path (str): Path to directory, where files are searched.
    Returns:
        Sorted list of all files from path. The files are sorted so that this function could be
        used in parallel tasks and each task takes the files corresponding to its
        index (given by MAGLEV_WORKFLOW_STEP_INSTANCE_INDEX).
    """
    return sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


def read_image(image_path, image_file_encoding, num_input_channels=3,
               width=None, height=None, data_format="channels_last"):
    """
    Set up image tensor reading pipeline.

    Args:
        image_path: A representation to the path of the image (e.g. str or tf.string)
        image_file_encoding: (str) e.g. ".jpg", ".png", or ".fp16". See nvidia_tao_tf1.core.processors for
            more info
        num_input_channels: (int) Number of channels in the input image.
        width: Likewise, but for image width.
        height: A representation of the image height (e.g. int or tf.int32)
        data_format (str): One of "channels_first" or "channels_last". Defaults to the former.

    Returns:
        image: (tf.Tensor) Tensor containing loaded image, shape (H, W, C), scaled to [0, 1] range.
    """
    file_loader = nvidia_tao_tf1.core.processors.LoadFile()

    # JPG and PNG images are in [0, 255], while .fp16 images are in [0, 1] range. Normalize all
    # image types to [0, 1].
    normalize = 1.0 if image_file_encoding == 'fp16' else 255.
    image_decoder = nvidia_tao_tf1.core.processors.DecodeImage(
        encoding=image_file_encoding,
        channels=num_input_channels,
        normalize=normalize,
        data_format=data_format
    )
    data = file_loader(image_path)
    image = image_decoder(data)

    if image_file_encoding == 'fp16':
        assert width is not None and height is not None
        image_shape = tf.stack([num_input_channels, height, width])
        image_shape = tf.cast(image_shape, dtype=tf.int32)
        image = tf.reshape(image, image_shape)

        # CHW -> HWC if required. Note that this is only needed here because the `fp16` case
        # returns a 1-D vector, as opposed to a 3-D vector for JPEG / PNGs.
        if data_format == 'channels_last':
            image = tf.transpose(a=image, perm=[1, 2, 0])

    return image
