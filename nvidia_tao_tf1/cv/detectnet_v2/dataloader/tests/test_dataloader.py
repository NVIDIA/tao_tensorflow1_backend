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

"""Test data loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import os
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.utils import set_random_seed
from nvidia_tao_tf1.cv.detectnet_v2.common.graph import get_init_ops
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation.augmentation_config import (
    AugmentationConfig
)
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import build_data_source_lists
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.data_source_config import DataSourceConfig
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import DefaultDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import FRAME_ID_KEY
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.drivenet_dataloader import BW_POLY_COEFF1_60FC
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.drivenet_dataloader import DriveNetDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.tests.utilities.data_generation import (
    generate_dummy_dataset,
    generate_dummy_images,
    generate_dummy_labels
)
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import extract_tfrecords_features
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_num_samples
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_tfrecords_iterator
from nvidia_tao_tf1.cv.detectnet_v2.dataio.kitti_converter_lib import (
    _bytes_feature,
    _float_feature,
    _int64_feature
)
from nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config_pb2 import DatasetConfig
from nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config_pb2 import DataSource


@pytest.fixture(scope='module')
def test_paths(tmpdir_factory):
    """Test paths."""
    # Generate three folds with 1, 2, and 3 samples, respectively.
    tmpdir = tmpdir_factory.mktemp('test_data')
    folds = [str(tmpdir.join("data.tfrecords-fold-00%d-of-003" % i))
             for i in range(3)]

    for num_examples, path in enumerate(folds, 1):
        generate_dummy_labels(path=path, num_samples=num_examples)

    return folds


def _get_dataloader(dataloader_class, training_data_source_list,
                    validation_data_source_list, augmentation_config,
                    image_file_encoding):
    """Helper function for creating a dataloader class."""
    if dataloader_class == 'DefaultDataloader':
        dataloader = DefaultDataloader(
            training_data_source_list=training_data_source_list,
            image_file_encoding=image_file_encoding,
            validation_data_source_list=validation_data_source_list,
            augmentation_config=augmentation_config
        )
    elif dataloader_class == 'DriveNetDataloader':
        dataloader = DriveNetDataloader(
            training_data_source_list=training_data_source_list,
            image_file_encoding=image_file_encoding,
            validation_data_source_list=validation_data_source_list,
            augmentation_config=augmentation_config
        )
    return dataloader


def _get_dummy_augmentation_config(width=16, height=8, channel=3, min_side=0,
                                   max_side=0, crop_right=16, crop_bottom=8,
                                   scale_width=0.0, scale_height=0.0):
    """Construct a dummy augmentation config."""
    preprocessing = AugmentationConfig.Preprocessing(
        output_image_width=width, output_image_height=height,
        output_image_channel=channel, output_image_min=min_side,
        output_image_max=max_side, crop_left=0, crop_top=0,
        crop_right=crop_right, crop_bottom=crop_bottom,
        min_bbox_width=0., min_bbox_height=0.,
        scale_width=scale_width, scale_height=scale_height)
    spatial = AugmentationConfig.SpatialAugmentation(0., 0., 1.0, 1.0, 0., 0., 0., 0.)
    color = AugmentationConfig.ColorAugmentation(0.0, 0.0, 0.0, 0.0, 0.5,)
    return AugmentationConfig(preprocessing, spatial, color)


@pytest.mark.parametrize("dataloader_class,validation_fold,expected_num_samples",
                         [('DefaultDataloader', 0, (5, 1)),
                          ('DefaultDataloader', 1, (4, 2)),
                          ('DefaultDataloader', None, (6, 0)),
                          ('DriveNetDataloader', 0, (5, 1)),
                          ('DriveNetDataloader', 1, (4, 2)),
                          ('DriveNetDataloader', None, (6, 0))])
def test_dataloader_kfold_split(test_paths, dataloader_class,
                                validation_fold, expected_num_samples):
    """Test Dataloader returning correct values when doing kfold validation split."""
    dataset_proto = DatasetConfig()
    data_sources_proto = DataSource()

    data_sources_proto.tfrecords_path = os.path.join(
        os.path.dirname(test_paths[0]), "*")
    data_sources_proto.image_directory_path = os.path.dirname(test_paths[0])
    dataset_proto.data_sources.extend([data_sources_proto])
    if validation_fold is not None:
        dataset_proto.validation_fold = validation_fold

    training_data_source_list, validation_data_source_list, _ = \
        build_data_source_lists(dataset_proto)

    image_file_encoding = "fp16"
    dataloader = _get_dataloader(dataloader_class=dataloader_class,
                                 training_data_source_list=training_data_source_list,
                                 validation_data_source_list=validation_data_source_list,
                                 augmentation_config=_get_dummy_augmentation_config(),
                                 image_file_encoding=image_file_encoding)

    # Check that the data source lists have the expected number of elements.
    if validation_fold is not None:
        expected_num_training_sources, expected_num_validation_sources = 1, 1
    else:
        expected_num_training_sources, expected_num_validation_sources = 1, 0
    assert len(dataloader.training_data_sources) == expected_num_training_sources
    assert len(
        dataloader.validation_data_sources) == expected_num_validation_sources

    num_training_samples = dataloader.get_num_samples(True)
    num_validation_samples = dataloader.get_num_samples(False)

    expected_num_training_samples, expected_num_validation_samples = expected_num_samples
    assert num_training_samples == expected_num_training_samples
    assert num_validation_samples == expected_num_validation_samples


@pytest.mark.parametrize("dataloader_class,validation_data_used,expected_num_samples",
                         [('DefaultDataloader', True, (6, 1)),
                          ('DefaultDataloader', False, (6, 0)),
                          ('DriveNetDataloader', True, (6, 1)),
                          ('DriveNetDataloader', False, (6, 0))])
def test_dataloader_path_split(test_paths, dataloader_class,
                               validation_data_used, expected_num_samples):
    """Test Dataloader returning correct values when specifying path to validation files."""
    dataset_proto = DatasetConfig()

    for fold in test_paths:
        training_data_sources_proto = DataSource()
        training_data_sources_proto.tfrecords_path = fold
        training_data_sources_proto.image_directory_path = os.path.dirname(
            fold)
        dataset_proto.data_sources.extend([training_data_sources_proto])

    if validation_data_used:
        dataset_proto.validation_data_source.tfrecords_path = test_paths[0]
        dataset_proto.validation_data_source.image_directory_path = os.path.dirname(
            test_paths[0])

    training_data_source_list, validation_data_source_list, _ = \
        build_data_source_lists(dataset_proto)

    num_paths = len(test_paths)
    expected_num_training_sources = num_paths
    if validation_data_used:
        expected_num_validation_sources = 1
    else:
        expected_num_validation_sources = 0

    image_file_encoding = "fp16"
    dataloader = _get_dataloader(dataloader_class=dataloader_class,
                                 training_data_source_list=training_data_source_list,
                                 validation_data_source_list=validation_data_source_list,
                                 augmentation_config=_get_dummy_augmentation_config(),
                                 image_file_encoding=image_file_encoding)

    # Check that the data source lists have the expected number of elements.
    assert len(dataloader.training_data_sources) == expected_num_training_sources
    assert len(
        dataloader.validation_data_sources) == expected_num_validation_sources

    # Check that data tensor shape matches AugmentationConfig.
    assert dataloader.get_data_tensor_shape() == (3, 8, 16)

    num_training_samples = dataloader.get_num_samples(True)
    num_validation_samples = dataloader.get_num_samples(False)

    expected_num_training_samples, expected_num_validation_samples = expected_num_samples
    assert num_training_samples == expected_num_training_samples
    assert num_validation_samples == expected_num_validation_samples


def _create_dummy_example():
    example = dict()
    example['target/object_class'] = tf.constant(
        ['car', 'pedestrian', 'cyclist'])

    return example


features1 = {'bytes': _bytes_feature('test')}
expected_features1 = {'bytes': tf.io.VarLenFeature(dtype=tf.string)}

features2 = {'float': _float_feature(1.0)}
expected_features2 = {'float': tf.io.VarLenFeature(dtype=tf.float32)}

features3 = {'int64': _int64_feature(1)}
expected_features3 = {'int64': tf.io.VarLenFeature(dtype=tf.int64)}

features4 = {'bytes1': _bytes_feature('test1'), 'bytes2': _bytes_feature('test2'),
             'float1': _float_feature(1.0), 'float2': _float_feature(2.0),
             'int64_1': _int64_feature(1), 'int64_2': _int64_feature(2)}
expected_features4 = {'bytes1': tf.io.VarLenFeature(dtype=tf.string),
                      'bytes2': tf.io.VarLenFeature(dtype=tf.string),
                      'float1': tf.io.VarLenFeature(dtype=tf.float32),
                      'float2': tf.io.VarLenFeature(dtype=tf.float32),
                      'int64_1': tf.io.VarLenFeature(dtype=tf.int64),
                      'int64_2': tf.io.VarLenFeature(dtype=tf.int64)}

test_cases = [(features1, expected_features1), (features2, expected_features2),
              (features3, expected_features3), (features4, expected_features4)]


@pytest.mark.parametrize("features,expected_features", test_cases)
def test_extract_tfrecords_features(tmpdir_factory, features, expected_features):
    """Test that tfrecords features are extracted correctly from a sample file."""
    # Generate a sample tfrecords file with the given features.
    tffile = str(tmpdir_factory.mktemp('test_data').join("test.tfrecords"))
    generate_dummy_labels(path=tffile, num_samples=1, labels=[features])

    extracted_features = extract_tfrecords_features(tffile)

    assert extracted_features == expected_features


@pytest.mark.parametrize("dataloader_class", [('DefaultDataloader'),
                                              ('DriveNetDataloader')])
def test_get_dataset_tensors(tmpdir_factory, dataloader_class):
    """Test dataloader.get_dataset_tensors."""

    def get_frame_id(label):
        """Extract frame id from label.

        Args:
            label: Dataset label.
        Returns:
            Frame ID (str).
        """
        if isinstance(label, list):
            return label[0][FRAME_ID_KEY][0]

        return np.squeeze(label.frame_id).flatten()[0]

    result_dir = tmpdir_factory.mktemp('test_dataloader_dataset')

    # Generate a tfrecords file for the test.
    image_width = 16
    image_height = 12
    num_dataset_samples = 3  # Must be > 2.

    tfrecords_path, image_directory_path =\
        generate_dummy_dataset(tmpdir=result_dir, num_samples=num_dataset_samples,
                               width=image_width, height=image_height)

    dataset_proto = DatasetConfig()
    training_data_sources_proto = DataSource()
    training_data_sources_proto.tfrecords_path =\
        tfrecords_path.replace('*', 'dummy-fold-000-of-002')
    training_data_sources_proto.image_directory_path = image_directory_path
    dataset_proto.data_sources.extend([training_data_sources_proto])

    dataset_proto.validation_data_source.tfrecords_path =\
        tfrecords_path.replace('*', 'dummy-fold-001-of-002')
    dataset_proto.validation_data_source.image_directory_path = image_directory_path

    training_data_source_list, validation_data_source_list, _ = \
        build_data_source_lists(dataset_proto)

    # Instantiate a dataloader.
    augmentation_config = _get_dummy_augmentation_config(width=image_width,
                                                         height=image_height,
                                                         crop_right=image_width,
                                                         crop_bottom=image_height)
    dataloader = _get_dataloader(dataloader_class=dataloader_class,
                                 training_data_source_list=training_data_source_list,
                                 validation_data_source_list=validation_data_source_list,
                                 augmentation_config=augmentation_config,
                                 image_file_encoding="png")

    # Test training set iteration.
    training_images, training_labels, num_training_samples =\
        dataloader.get_dataset_tensors(1, True, False)

    assert num_training_samples == num_dataset_samples
    assert dataloader.get_num_samples(True) == num_dataset_samples

    expected_image_shape = (1, 3, image_height, image_width)
    # Note that get_data_tensor_shape does not include the batch dimension.
    assert dataloader.get_data_tensor_shape() == expected_image_shape[1:]

    with tf.compat.v1.Session() as session:
        session.run(get_init_ops())
        # Loop training set twice to test repeat.
        for _ in range(2):
            # Loop over training set once, storing the frame IDs.
            training_set = set()
            for _ in range(num_training_samples):
                image, label = session.run([training_images, training_labels])
                assert image.shape == expected_image_shape
                training_set.add(get_frame_id(label))

            # Assert that each frame id is present. Note that the order of the IDs is random.
            assert {str(i).encode()
                    for i in range(num_training_samples)} == training_set

    # Test validation set iteration.
    validation_images, validation_labels, num_validation_samples =\
        dataloader.get_dataset_tensors(1, False, False)

    assert num_validation_samples == num_dataset_samples
    assert dataloader.get_num_samples(False) == num_dataset_samples

    with tf.compat.v1.Session() as session:
        session.run(get_init_ops())

        # Loop over the validation set twice.
        for i in range(num_validation_samples):
            image, label = session.run([validation_images, validation_labels])
            assert image.shape == expected_image_shape
            assert get_frame_id(label) == str(
                i % num_validation_samples).encode()


X_COORDS = [20, 24, 24, 20, 12, 16, 16, 12, 16, 20, 20, 16]
Y_COORDS = [16, 16, 26, 26, 26, 26, 36, 36, 20, 20, 22, 22]
COORDINATE_FORMATS = [
    {'target/coordinates/x': _float_feature(*X_COORDS),
     'target/coordinates/y': _float_feature(*Y_COORDS),
     'target/coordinates/index': _int64_feature(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2)},
    {'target/coordinates_x1': _float_feature(20, 12, 16),
     'target/coordinates_x2': _float_feature(24, 16, 20),
     'target/coordinates_y1': _float_feature(16, 26, 20),
     'target/coordinates_y2': _float_feature(26, 36, 22)}
]


@pytest.fixture(params=COORDINATE_FORMATS)
def label_filters_test_data(request):
    image_width = 16
    image_height = 12

    input_labels = [{
        'frame/id': _bytes_feature("0"),
        'frame/height': _int64_feature(image_height),
        'frame/width': _int64_feature(image_width),
        'target/object_class': _bytes_feature("car", "car", "car")
    }]
    input_labels[0].update(request.param)
    # After applying stm, bbox1 should be clipped to crop region, bbox2 should be filtered out as
    # it is completely outside the crop region, bbox 3 is retained.
    expected_output = [
        {'frame/id': [b"0"], 'frame/height': [image_height], 'frame/width': [image_width],
         'target/object_class': [b"car", b"car"],
         'target/coordinates/x': [10, 12, 12, 10, 8, 10, 10, 8],
         'target/coordinates/y': [8, 8, 12, 12, 10, 10, 11, 11],
         'target/coordinates/index': [0, 0, 0, 0, 1, 1, 1, 1],
         'target/bbox_coordinates': [[10., 8., 12., 12.],
                                     [8., 10., 10., 11.]]}]
    fixture = {'image_width': image_width, 'image_height': image_height,
               'input_labels': input_labels, 'expected_output': expected_output}
    return fixture


# TODO(@williamz): How to adapt this test to DriveNetDataloader?
@pytest.mark.parametrize("dataloader_class", ['DefaultDataloader'])
def test_get_dataset_tensors_filters(tmpdir, label_filters_test_data, dataloader_class):
    """Test that get_dataset_tensors() applies label filters correctly."""
    # Generate a tfrecords file for the test.
    tfrecords_path = str(tmpdir.mkdir("labels"))
    image_directory_path = str(tmpdir.mkdir("images"))
    training_tfrecords_path = tfrecords_path + '/dummy-fold-000-of-002'

    image_width = label_filters_test_data['image_width']
    image_height = label_filters_test_data['image_height']
    generate_dummy_images(directory=image_directory_path,
                          num_samples=len(
                              label_filters_test_data['input_labels']),
                          height=image_height, width=image_width,
                          num_channels=3)
    generate_dummy_labels(path=training_tfrecords_path,
                          num_samples=len(
                              label_filters_test_data['input_labels']),
                          labels=label_filters_test_data['input_labels'])

    dataset_proto = DatasetConfig()
    training_data_sources_proto = DataSource()
    training_data_sources_proto.tfrecords_path = training_tfrecords_path
    training_data_sources_proto.image_directory_path = image_directory_path
    dataset_proto.data_sources.extend([training_data_sources_proto])

    training_data_source_list, validation_data_source_list, _ = \
        build_data_source_lists(dataset_proto)

    # Instantiate a dataloader.
    augmentation_config = _get_dummy_augmentation_config(width=image_width,
                                                         height=image_height,
                                                         crop_right=image_width//2,
                                                         crop_bottom=image_height//2,
                                                         scale_width=0.5,
                                                         scale_height=0.5)
    dataloader = _get_dataloader(dataloader_class=dataloader_class,
                                 training_data_source_list=training_data_source_list,
                                 validation_data_source_list=validation_data_source_list,
                                 augmentation_config=augmentation_config,
                                 image_file_encoding="png")

    _, training_labels, _ = dataloader.get_dataset_tensors(1, True, False)

    with tf.compat.v1.Session() as session:
        session.run(get_init_ops())
        label = session.run(training_labels)
        for key, value in label_filters_test_data['expected_output'][0].items():
            assert np.array_equal(value, label[0][key])


def test_get_tfrecords_iterator(test_paths):
    """Test that samples from all data sources are iterated over."""
    data_sources = []
    for i, test_path in enumerate(test_paths):
        data_sources.append(
            DataSourceConfig(dataset_type='tfrecord',
                             dataset_files=[test_path],
                             images_path=str(i),
                             export_format=None,
                             split_db_path=None,
                             split_tags=None))

    num_samples = get_num_samples(data_sources, training=False)

    tfrecords_iterator, num_samples2 = get_tfrecords_iterator(
        data_sources=data_sources,
        batch_size=1,
        training=False,
        repeat=True)

    assert num_samples == num_samples2

    # Initialize the iterator.
    with tf.compat.v1.Session() as session:
        session.run(get_init_ops())
        # Loop once through the entire dataset, and check that all sources have been iterated over.
        samples_per_source = Counter()
        for _ in range(num_samples):
            sample = session.run(tfrecords_iterator())
            img_dir = sample[1][0].decode()
            samples_per_source[img_dir] += 1

        # Given the test_paths fixture puts 1, 2 and 3 samples in each source, check that this is
        # correct.
        for i in range(len(test_paths)):
            assert samples_per_source[str(i) + '/'] == i + 1


crop_rect = [{'left': 5, 'right': 45, 'top': 5, 'bottom': 45},
             {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}]
coordinates = {
    'x': [0., 10., 10., 0., 20., 30., 30., 20., 40., 50., 50., 40.],
    'y': [0., 0., 10., 10., 20., 20., 30., 30., 40., 40., 50., 50.],
    'idx': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
}

truncation_type = [1, 0, 0]

expected_bboxes = [[[5., 20., 40.],
                    [10., 30., 45.],
                    [5., 20., 40.],
                    [10., 30., 45.]],
                   [[0., 20., 40.],
                    [10., 30., 50.],
                    [0., 20., 40.],
                    [10., 30., 50.]]]

expected_truncation_type = [[1, 0, 1], [1, 0, 0]]

test_cases = [(crop_rect[0], coordinates, truncation_type,
               expected_bboxes[0], expected_truncation_type[0]),
              (crop_rect[1], coordinates, truncation_type,
               expected_bboxes[1], expected_truncation_type[1])]


@pytest.mark.parametrize(
    "crop_rect,coordinates,truncation_type,expected_bboxes,expected_truncation_type",
    test_cases)
def test_update_example_after_crop(crop_rect, coordinates, truncation_type,
                                   expected_bboxes, expected_truncation_type):
    """Test expected updated bbox and truncation_type can be obtained."""
    # Create an example.
    example = _create_dummy_example()
    example['target/coordinates/x'] = tf.constant(coordinates['x'])
    example['target/coordinates/y'] = tf.constant(coordinates['y'])
    example['target/coordinates/index'] = tf.constant(coordinates['idx'])
    example['target/truncation_type'] = tf.constant(truncation_type)

    # Calculate preprocessed truncation.
    example = DefaultDataloader._update_example_after_crop(crop_rect['left'],
                                                           crop_rect['right'],
                                                           crop_rect['top'],
                                                           crop_rect['bottom'],
                                                           example)

    # Compute and compare results.
    with tf.compat.v1.Session() as session:
        session.run(get_init_ops())
        result_coords, result_truncation_type = \
            session.run([example['target/bbox_coordinates'],
                         example['target/truncation_type']])

        result_x1, result_y1, result_x2, result_y2 = np.split(
            result_coords, 4, axis=1)
        # Check the correctness of bbox coordinates.
        result_bboxes = [result_x1.tolist(), result_x2.tolist(),
                         result_y1.tolist(), result_y2.tolist()]
        for result, expected in zip(result_bboxes, expected_bboxes):
            for x, y in zip(result, expected):
                assert np.isclose(x, y)

        # Check the correctness of truncation_type.
        for x, y in zip(result_truncation_type, expected_truncation_type):
            assert x == y


def get_hflip_augmentation_config(hflip_probability):
    """Get an AugmentationConfig for testing horizontal flips."""
    assert hflip_probability in {0., 1.0}
    # Define some augmentation configurables.
    width, height = 12, 34
    preprocessing = AugmentationConfig.Preprocessing(
        output_image_width=width,
        output_image_height=height,
        output_image_channel=3,
        output_image_min=0,
        output_image_max=0,
        crop_left=0,
        crop_top=0,
        crop_right=width,
        crop_bottom=height,
        min_bbox_width=0.0,
        min_bbox_height=0.0,
        scale_width=0.0,
        scale_height=0.0)
    spatial_augmentation = AugmentationConfig.SpatialAugmentation(
        hflip_probability=hflip_probability,
        vflip_probability=0.0,
        zoom_min=0.9,
        zoom_max=1.2,
        translate_max_x=2.0,
        translate_max_y=3.0,
        rotate_rad_max=0.0,
        rotate_probability=1.0)

    return AugmentationConfig(preprocessing, spatial_augmentation, None)


@pytest.mark.parametrize(
    "hflip_probability,expected_front_marker,expected_back_marker",
    [(0.0, 0.0, 0.5), (1.0, 1.0, 0.5)]
)
@pytest.mark.parametrize("dataloader_class,num_preceding_frames",
                         [('DefaultDataloader', 0)])
def test_markers_are_augmented(tmpdir,
                               dataloader_class,
                               num_preceding_frames,
                               hflip_probability,
                               expected_front_marker,
                               expected_back_marker):
    """Test that get_dataset_tensors produces the expected front / back markers."""
    set_random_seed(123)
    num_samples, height, width = 1, 10, 20
    tfrecords_path, image_directory_path = \
        generate_dummy_dataset(tmpdir=tmpdir, num_samples=num_samples,
                               height=height, width=width)

    dataset_proto = DatasetConfig()
    training_data_sources_proto = DataSource()
    training_data_sources_proto.tfrecords_path = tfrecords_path
    training_data_sources_proto.image_directory_path = image_directory_path
    dataset_proto.data_sources.extend([training_data_sources_proto])

    training_data_source_list, validation_data_source_list, _ = \
        build_data_source_lists(dataset_proto)

    augmentation_config = get_hflip_augmentation_config(
        hflip_probability=hflip_probability)

    dataloader = _get_dataloader(
        dataloader_class=dataloader_class,
        training_data_source_list=training_data_source_list,
        image_file_encoding='png',
        validation_data_source_list=validation_data_source_list,
        augmentation_config=augmentation_config)

    labels_tensors = dataloader.get_dataset_tensors(
        batch_size=num_samples,
        training=True,
        enable_augmentation=True)[1]

    with tf.compat.v1.Session() as sess:
        sess.run(get_init_ops())
        labels = sess.run(labels_tensors)

    for frame_labels in labels:
        assert np.allclose(frame_labels['target/front'], expected_front_marker)
        assert np.allclose(frame_labels['target/back'], expected_back_marker)


# Augmentation matrices for depth augmentation test.
identity_stm = np.eye(3, dtype=np.float32)
zoom_out_stm = [[0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 1.0]]
zoom_in_stm = [[2.0, 0.0, 0.0],
               [0.0, 2.0, 0.0],
               [0.0, 0.0, 1.0]]
isq2 = np.sqrt(2.0) / 2.0
rotation_stm = [[isq2, -isq2, 0.0],
                [isq2, isq2, 0.0],
                [0.0, 0.0, 1.0]]


@pytest.mark.parametrize("label_depth,stm,bw_poly_coeff1,expected_depth",
                         [
                             # Regular depth.
                             ([12.34], identity_stm,
                                 BW_POLY_COEFF1_60FC, [12.34]),
                             # Zoom in 2x.
                             ([8.0], zoom_in_stm,
                                 BW_POLY_COEFF1_60FC, [4.0]),
                             # Zoom out 2x.
                             ([8.0], zoom_out_stm,
                                 BW_POLY_COEFF1_60FC, [16.0]),
                             # Rotation 45 degrees.
                             ([8.0], rotation_stm,
                                 BW_POLY_COEFF1_60FC, [8.0]),
                             # Different FOV camera.
                             ([8.0], identity_stm, 0.001, [14.6675553]),
                         ])
def test_depth_is_augmented(label_depth, stm, bw_poly_coeff1, expected_depth):
    """Test that deth is augmented properly."""
    set_random_seed(123)

    augmentation_config = _get_dummy_augmentation_config()

    dataloader = DefaultDataloader(
        training_data_source_list=[],
        image_file_encoding='png',
        validation_data_source_list=[],
        augmentation_config=augmentation_config)

    labels_tensors = {
        'target/world_bbox_z': tf.constant(label_depth),
        'frame/bw_poly_coeff1': tf.constant(bw_poly_coeff1)}

    labels_tensors = dataloader._translate_additional_labels(labels_tensors)
    labels_tensors = dataloader._apply_augmentations_to_additional_labels(
        additional_labels=labels_tensors,
        stm=tf.constant(stm))

    with tf.compat.v1.Session() as sess:
        sess.run(get_init_ops())
        labels = sess.run(labels_tensors)

    assert np.allclose(labels['target/world_bbox_z'], expected_depth)


class TestMultipleTFRecordsFormats(object):
    """
    Test that the Dataloader is able to handle both old and new TFRecords formats, when BOTH
    are supplied at the same time.
    """

    @pytest.fixture(scope='function')
    def dataloader(self, tmpdir):
        """Define a Dataloader instance for this test's purpose."""
        set_random_seed(123)
        num_samples, height, width = 1, 40, 40

        def get_common_labels(i):
            return {
                'target/object_class': _bytes_feature('car', 'cat', 'cart'),
                'target/truncation': _float_feature(0.0, 0.0, 0.0),
                'target/occlusion': _int64_feature(0, 0, 0),
                'target/front': _float_feature(0.0, 0.0, 0.0),
                'target/back': _float_feature(0.5, 0.5, 0.5),
                'frame/id': _bytes_feature(str(i)),
                'frame/height': _int64_feature(height),
                'frame/width': _int64_feature(width)}

        # Generate one tfrecord for each type of format.
        dataset_proto = DatasetConfig()
        for i, coordinate_format in enumerate(COORDINATE_FORMATS):
            labels = get_common_labels(i)
            labels.update(coordinate_format)
            tfrecords_path, image_directory_path = \
                generate_dummy_dataset(tmpdir=tmpdir.mkdir("dir%d" % i), num_samples=num_samples,
                                       height=height, width=width,
                                       labels=[labels])
            training_data_sources_proto = DataSource()
            training_data_sources_proto.tfrecords_path = tfrecords_path
            training_data_sources_proto.image_directory_path = image_directory_path
            dataset_proto.data_sources.extend([training_data_sources_proto])

        training_data_source_list, validation_data_source_list, _ = \
            build_data_source_lists(dataset_proto)

        augmentation_config = _get_dummy_augmentation_config(
            width=width, height=height, crop_right=width, crop_bottom=height)

        dataloader = DefaultDataloader(
            training_data_source_list=training_data_source_list,
            image_file_encoding='png',
            validation_data_source_list=validation_data_source_list,
            augmentation_config=augmentation_config)

        return dataloader

    def test_multiple_formats(self, dataloader):
        """Test that the labels coming out of the dataloader are as expected."""
        total_num_samples = dataloader.get_num_samples(training=True)

        labels_tensors = dataloader.get_dataset_tensors(
            batch_size=1, training=True, enable_augmentation=True)[1]

        frame_ids = set()
        with tf.compat.v1.Session() as sess:
            sess.run(get_init_ops())
            # 1 sample from each source.
            for _ in range(total_num_samples):
                batch_labels = sess.run(labels_tensors)
                for frame_labels in batch_labels:
                    frame_ids.add(frame_labels["frame/id"][0])
                    np.testing.assert_allclose(
                        frame_labels["target/coordinates/x"],
                        np.array(X_COORDS, dtype=np.float32))
                    np.testing.assert_allclose(
                        frame_labels["target/coordinates/y"],
                        np.array(Y_COORDS, dtype=np.float32))

        assert frame_ids == {b"0", b"1"}
