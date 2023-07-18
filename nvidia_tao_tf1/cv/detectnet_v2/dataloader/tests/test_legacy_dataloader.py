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
from six.moves import range
from six.moves import zip
import tensorflow as tf

from nvidia_tao_tf1.core.utils import set_random_seed
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import (
    _bytes_feature,
    _float_feature
)
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _int64_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.graph import get_init_ops
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation.augmentation_config import (
    AugmentationConfig
)
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.legacy_dataloader import FRAME_ID_KEY
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.legacy_dataloader import LegacyDataloader as \
    DefaultDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.tests.utilities.data_generation import (
    generate_dummy_dataset,
    generate_dummy_images,
    generate_dummy_labels
)
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import extract_tfrecords_features
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_num_samples
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_tfrecords_iterator


@pytest.fixture(scope='module')
def test_paths(tmpdir_factory):
    """Test paths."""
    # Generate three folds with 1, 2, and 3 samples, respectively.
    tmpdir = tmpdir_factory.mktemp('test_data')
    folds = [str(tmpdir.join("data.tfrecords-fold-00%d-of-003" % i)) for i in range(3)]

    for num_examples, path in enumerate(folds, 1):
        generate_dummy_labels(path, num_examples)

    return folds


@pytest.fixture
def mock_generate_tensors(mocker):
    """Skip the generation of input images and ground truth label tensors."""
    def make_mock(dataloader):
        mock = mocker.patch.object(dataloader, '_generate_images_and_ground_truth_labels')
        dummy_tensor = tf.constant([0.0])
        mock.return_value = dummy_tensor, dummy_tensor

        return mock

    return make_mock


def _get_dataloader(dataloader_class, training_data_source_list, target_class_mapping,
                    validation_fold, validation_data_source_list, augmentation_config):
    """Helper function for creating a dataloader class."""
    if dataloader_class == 'DefaultDataloader':
        dataloader = DefaultDataloader(
            training_data_source_list=training_data_source_list,
            target_class_mapping=target_class_mapping,
            image_file_encoding="fp16",
            validation_fold=validation_fold,
            validation_data_source_list=validation_data_source_list,
            augmentation_config=augmentation_config
        )
    # HOW TO CONFIGURE AN ASSERT FOR OTHER dataloades - temporal and point cloud
    return dataloader


@pytest.mark.parametrize("dataloader_class,validation_fold,expected_num_samples",
                         [('DefaultDataloader', 0, (5, 1)),
                          ('DefaultDataloader', 1, (4, 2)),
                          ('DefaultDataloader', None, (6, 0))])
def test_dataloader_kfold_split(mock_generate_tensors, test_paths, dataloader_class,
                                validation_fold, expected_num_samples):
    """Test Dataloader returning correct values when doing kfold validation split."""
    training_data_source_list = [(os.path.join(os.path.dirname(test_paths[0]), "*"), "")]

    target_class_mapping = dict()
    validation_data_source_list = None
    augmentation_config = None

    dataloader = _get_dataloader(dataloader_class, training_data_source_list, target_class_mapping,
                                 validation_fold, validation_data_source_list, augmentation_config)

    mock_generate_tensors(dataloader)

    # Check that the data source lists have the expected number of elements.
    if validation_fold is not None:
        expected_num_training_sources, expected_num_validation_sources = 1, 1
    else:
        expected_num_training_sources, expected_num_validation_sources = 1, 0
    assert len(dataloader.training_data_sources) == expected_num_training_sources
    assert len(dataloader.validation_data_sources) == expected_num_validation_sources

    _, _, num_training_samples = dataloader.get_dataset_tensors(10, True, False)
    _, _, num_validation_samples = dataloader.get_dataset_tensors(10, False, False)

    expected_num_training_samples, expected_num_validation_samples = expected_num_samples
    assert num_training_samples == expected_num_training_samples
    assert num_validation_samples == expected_num_validation_samples


@pytest.mark.parametrize("dataloader_class,validation_data_used,expected_num_samples,num_channels",
                         [('DefaultDataloader', True, (6, 1), 1),
                          ('DefaultDataloader', True, (6, 1), 3),
                          ('DefaultDataloader', False, (6, 0), 1),
                          ('DefaultDataloader', False, (6, 0), 3)])
def test_dataloader_path_split(mock_generate_tensors, test_paths, dataloader_class,
                               validation_data_used, expected_num_samples, num_channels):
    """Test Dataloader returning correct values when specifying path to validation files."""
    training_data_source_list = [(fold, "") for fold in test_paths]
    num_paths = len(test_paths)
    expected_num_training_sources = num_paths
    if validation_data_used:
        validation_data_source_list = [(test_paths[0], "")]
        expected_num_validation_sources = 1
    else:
        validation_data_source_list = None
        expected_num_validation_sources = 0

    preprocessing = AugmentationConfig.Preprocessing(
        output_image_width=16, output_image_height=8,
        output_image_channel=num_channels,
        output_image_min=0, output_image_max=0,
        crop_left=0, crop_top=0, crop_right=16, crop_bottom=8,
        min_bbox_width=0., min_bbox_height=0., scale_width=0., scale_height=0.)
    augmentation_config = AugmentationConfig(preprocessing, None, None)

    target_class_mapping = dict()
    validation_fold = None

    dataloader = _get_dataloader(dataloader_class, training_data_source_list, target_class_mapping,
                                 validation_fold, validation_data_source_list, augmentation_config)

    mock_generate_tensors(dataloader)

    # Check that the data source lists have the expected number of elements.
    assert len(dataloader.training_data_sources) == expected_num_training_sources
    assert len(dataloader.validation_data_sources) == expected_num_validation_sources

    # Check that data tensor shape matches AugmentationConfig.
    assert dataloader.get_data_tensor_shape() == \
        (augmentation_config.preprocessing.output_image_channel, 8, 16)
    num_training_samples = dataloader.get_num_samples(True)
    num_validation_samples = dataloader.get_num_samples(False)

    _, _, num_training_samples2 = dataloader.get_dataset_tensors(10, True, False)
    _, _, num_validation_samples2 = dataloader.get_dataset_tensors(10, False, False)

    # Check that get_num_samples and get_dataset_tensors return the same number of samples.
    assert num_training_samples == num_training_samples2
    assert num_validation_samples == num_validation_samples2

    expected_num_training_samples, expected_num_validation_samples = expected_num_samples
    assert num_training_samples == expected_num_training_samples
    assert num_validation_samples == expected_num_validation_samples


def _create_dummy_example():
    example = dict()
    example['target/object_class'] = tf.constant(['car', 'pedestrian', 'cyclist'])

    return example


@pytest.mark.parametrize("dataloader_class", [('DefaultDataloader')])
def test_target_class_mapping(test_paths, dataloader_class):
    """Target class mapping."""
    training_data_source_list = [(fold, "") for fold in test_paths]

    target_class_mapping = {b'car': b'car', b'cyclist': b'cyclist', b'pedestrian': b'cyclist'}

    validation_fold = None
    validation_data_source_list = None
    augmentation_config = None

    dataloader = _get_dataloader(dataloader_class, training_data_source_list, target_class_mapping,
                                 validation_fold, validation_data_source_list, augmentation_config)

    example = _create_dummy_example()
    example = dataloader._map_to_model_target_classes(example, target_class_mapping)

    with tf.Session() as session:
        session.run(get_init_ops())

        example = session.run(example)
        object_classes = example['target/object_class']
        # target_class_values = [f.decode() for f in target_class_mapping.values()
        assert set(object_classes) == set(target_class_mapping.values())


features1 = {'bytes': _bytes_feature('test')}
expected_features1 = {'bytes': tf.VarLenFeature(dtype=tf.string)}

features2 = {'float': _float_feature(1.0)}
expected_features2 = {'float': tf.VarLenFeature(dtype=tf.float32)}

features3 = {'int64': _int64_feature(1)}
expected_features3 = {'int64': tf.VarLenFeature(dtype=tf.int64)}

features4 = {'bytes1': _bytes_feature('test1'), 'bytes2': _bytes_feature('test2'),
             'float1': _float_feature(1.0), 'float2': _float_feature(2.0),
             'int64_1': _int64_feature(1), 'int64_2': _int64_feature(2)}
expected_features4 = {'bytes1': tf.VarLenFeature(dtype=tf.string),
                      'bytes2': tf.VarLenFeature(dtype=tf.string),
                      'float1': tf.VarLenFeature(dtype=tf.float32),
                      'float2': tf.VarLenFeature(dtype=tf.float32),
                      'int64_1': tf.VarLenFeature(dtype=tf.int64),
                      'int64_2': tf.VarLenFeature(dtype=tf.int64)}

test_cases = [(features1, expected_features1), (features2, expected_features2),
              (features3, expected_features3), (features4, expected_features4)]


@pytest.mark.parametrize("features,expected_features", test_cases)
def test_extract_tfrecords_features(tmpdir_factory, features, expected_features):
    """Test that tfrecords features are extracted correctly from a sample file."""
    # Generate a sample tfrecords file with the given features.
    tffile = str(tmpdir_factory.mktemp('test_data').join("test.tfrecords"))
    generate_dummy_labels(tffile, 1, labels=[features])

    extracted_features = extract_tfrecords_features(tffile)

    assert extracted_features == expected_features


@pytest.mark.parametrize("num_channels", [1, 3])
def test_get_dataset_tensors(tmpdir_factory, num_channels):
    """Test dataloader.get_dataset_tensors."""

    def get_frame_id(label):
        """Extract frame id from label.

        Args:
            label: Dataset label.
        Returns:
            Frame ID (str).
        """
        return label[0][FRAME_ID_KEY][0]

    result_dir = tmpdir_factory.mktemp('test_dataloader_dataset')

    # Generate a tfrecords file for the test.
    image_width = 16
    image_height = 12
    num_dataset_samples = 3  # Must be > 2.
    num_channels = num_channels
    tfrecords_path, image_directory_path = generate_dummy_dataset(result_dir, num_dataset_samples,
                                                                  image_width, image_height,
                                                                  labels=None,
                                                                  num_channels=num_channels)

    training_tfrecords_path = tfrecords_path.replace('*', 'dummy-fold-000-of-002')
    training_data_source_list = [(training_tfrecords_path, image_directory_path)]

    validation_tfrecords_path = tfrecords_path.replace('*', 'dummy-fold-001-of-002')
    validation_data_source_list = [(validation_tfrecords_path, image_directory_path)]

    # Instantiate a DefaultDataloader.
    preprocessing = AugmentationConfig.Preprocessing(
        output_image_width=image_width,
        output_image_height=image_height,
        output_image_channel=num_channels,
        output_image_min=0,
        output_image_max=0,
        crop_left=0, crop_top=0,
        crop_right=image_width, crop_bottom=image_height,
        min_bbox_width=0., min_bbox_height=0., scale_width=0., scale_height=0.)
    augmentation_config = AugmentationConfig(preprocessing=preprocessing,
                                             spatial_augmentation=None,
                                             color_augmentation=None)
    dataloader = DefaultDataloader(
        training_data_source_list=training_data_source_list,
        target_class_mapping=dict(),
        image_file_encoding="png",
        validation_fold=None,
        validation_data_source_list=validation_data_source_list,
        augmentation_config=augmentation_config
    )

    # Test training set iteration.
    training_images, training_labels, num_training_samples =\
        dataloader.get_dataset_tensors(1, True, False)

    assert num_training_samples == num_dataset_samples
    expected_image_shape = (1, augmentation_config.preprocessing.output_image_channel, image_height,
                            image_width)
    # Note that get_data_tensor_shape does not include the batch dimension.
    assert dataloader.get_data_tensor_shape() == expected_image_shape[1:]

    with tf.Session() as session:
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
            assert set(bytes(str(i), 'utf-8') for i in range(num_training_samples)) == training_set

    # Test validation set iteration.
    validation_images, validation_labels, num_validation_samples =\
        dataloader.get_dataset_tensors(1, False, False)

    assert num_validation_samples == num_dataset_samples

    with tf.Session() as session:
        session.run(get_init_ops())

        # Loop over the validation set twice.
        for i in range(num_validation_samples):
            image, label = session.run([validation_images, validation_labels])
            assert image.shape == expected_image_shape
            assert get_frame_id(label) == bytes(str(i % num_validation_samples), 'utf-8')


@pytest.fixture
def label_filters_test_data():
    image_width = 16
    image_height = 12
    input_labels = [
        {'frame/id': _bytes_feature("xx"), 'frame/height': _int64_feature(image_height),
         'frame/width': _int64_feature(image_width),
         'target/object_class': _bytes_feature("car", "car", "car"),
         'target/coordinates_x1': _float_feature(20, 12, 16),
         'target/coordinates_x2': _float_feature(24, 16, 20),
         'target/coordinates_y1': _float_feature(16, 26, 20),
         'target/coordinates_y2': _float_feature(26, 36, 22)}]
    # After applying stm, bbox1 should be clipped to crop region, bbox2 should be filtered out as
    # it is completely outside the crop region, bbox 3 is retained.
    expected_output = [
        {'frame/id': [b"xx"], 'frame/height': [image_height], 'frame/width': [image_width],
         'target/object_class': [b"car", b"car"], 'target/coordinates_x1': [10., 8.],
         'target/coordinates_x2':[12., 10.], 'target/coordinates_y1': [8., 10.],
         'target/coordinates_y2': [12., 11.],
         'target/bbox_coordinates': [[10., 8., 12., 12.],
                                     [8., 10., 10., 11.]]}]

    fixture = {'image_width': image_width, 'image_height': image_height,
               'input_labels': input_labels, 'expected_output': expected_output
               }
    return fixture


@pytest.mark.parametrize("num_channels", [1, 3])
def test_get_dataset_tensors_filters(tmpdir, label_filters_test_data, num_channels):
    """Test that get_dataset_tensors() applies label filters correctly."""
    # Generate a tfrecords file for the test.
    tfrecords_path = str(tmpdir.mkdir("labels"))
    image_directory_path = str(tmpdir.mkdir("images"))
    training_tfrecords_path = tfrecords_path + '/dummy-fold-000-of-002'
    num_channels = num_channels
    image_width = label_filters_test_data['image_width']
    image_height = label_filters_test_data['image_height']
    generate_dummy_images(image_directory_path, len(label_filters_test_data['input_labels']),
                          image_width, image_height, num_channels)
    generate_dummy_labels(training_tfrecords_path, len(label_filters_test_data['input_labels']),
                          labels=label_filters_test_data['input_labels'])

    training_data_source_list = [(training_tfrecords_path, image_directory_path)]

    # Instantiate a DefaultDataloader.
    preprocessing = AugmentationConfig.Preprocessing(
        output_image_width=image_width,
        output_image_height=image_height,
        output_image_channel=num_channels,
        output_image_min=0,
        output_image_max=0,
        crop_left=0, crop_top=0,
        crop_right=image_width//2, crop_bottom=image_height//2,
        min_bbox_width=0., min_bbox_height=0., scale_width=0.5, scale_height=0.5)
    augmentation_config = AugmentationConfig(preprocessing=preprocessing,
                                             spatial_augmentation=None,
                                             color_augmentation=None)
    dataloader = DefaultDataloader(
        training_data_source_list=training_data_source_list,
        target_class_mapping=dict(),
        image_file_encoding="png",
        validation_fold=None,
        validation_data_source_list=None,
        augmentation_config=augmentation_config
    )

    _, training_labels, _ = dataloader.get_dataset_tensors(1, True, False)

    with tf.Session() as session:
        session.run(get_init_ops())
        label = session.run([training_labels])
        for key, value in label_filters_test_data['expected_output'][0].items():
            assert np.array_equal(value, label[0][0][key])


def test_get_tfrecords_iterator(mock_generate_tensors, test_paths):
    """Test that samples from all data sources are iterated over."""
    data_sources = []
    for i, test_path in enumerate(test_paths):
        data_source = ([test_path], str(i))
        data_sources.append(data_source)

    num_samples = get_num_samples(data_sources, training=False)

    tfrecords_iterator, num_samples2 = get_tfrecords_iterator(
        data_sources=data_sources,
        batch_size=1,
        training=False,
        repeat=True)

    assert num_samples == num_samples2

    # Initialize the iterator.
    with tf.Session() as session:
        session.run(get_init_ops())
        # Loop once through the entire dataset, and check that all sources have been iterated over.
        samples_per_source = Counter()
        for _ in range(num_samples):
            sample = session.run(tfrecords_iterator())
            img_dir = sample[1][0]
            samples_per_source[img_dir] += 1

        # Given the test_paths fixture puts 1, 2 and 3 samples in each source, check that this is
        # correct.
        for i in range(len(test_paths)):
            assert samples_per_source[bytes(str(i), 'utf-8') + b'/'] == i + 1


crop_rect = [{'left': 5, 'right': 45, 'top': 5, 'bottom': 45},
             {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}]

bboxes = {'x1': [0., 20., 40.],
          'x2': [10., 30., 50.],
          'y1': [0., 20., 40.],
          'y2': [10., 30., 50.]}

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

test_cases = [(crop_rect[0], bboxes, truncation_type,
               expected_bboxes[0], expected_truncation_type[0]),
              (crop_rect[1], bboxes, truncation_type,
               expected_bboxes[1], expected_truncation_type[1])]


@pytest.mark.parametrize(
    "crop_rect,bboxes,truncation_type,expected_bboxes,expected_truncation_type",
    test_cases)
def test_update_example_after_crop(crop_rect, bboxes, truncation_type,
                                   expected_bboxes, expected_truncation_type):
    """Test expected updated bbox and truncation_type can be obtained."""
    # Create an example.
    example = _create_dummy_example()
    example['target/bbox_coordinates'] = tf.stack([bboxes['x1'], bboxes['y1'], bboxes['x2'],
                                                   bboxes['y2']], axis=1)
    example['target/truncation_type'] = tf.constant(truncation_type)

    # Calculate preprocessed truncation.
    example = DefaultDataloader._update_example_after_crop(crop_rect['left'],
                                                           crop_rect['right'],
                                                           crop_rect['top'],
                                                           crop_rect['bottom'],
                                                           example)

    # Compute and compare results.
    with tf.Session() as session:
        session.run(get_init_ops())
        result_coords, result_truncation_type = \
            session.run([example['target/bbox_coordinates'],
                         example['target/truncation_type']])

        result_x1, result_y1, result_x2, result_y2 = np.split(result_coords, 4, axis=1)
        # Check the correctness of bbox coordinates.
        result_bboxes = [result_x1.tolist(), result_x2.tolist(),
                         result_y1.tolist(), result_y2.tolist()]
        for result, expected in zip(result_bboxes, expected_bboxes):
            for x, y in zip(result, expected):
                assert np.isclose(x, y)

        # Check the correctness of truncation_type.
        for x, y in zip(result_truncation_type, expected_truncation_type):
            assert x == y


def get_hflip_augmentation_config(hflip_probability, num_channels):
    """Get an AugmentationConfig for testing horizontal flips."""
    assert hflip_probability in {0., 1.0}
    # Define some augmentation configurables.
    width, height, num_channels = 12, 34, num_channels
    preprocessing = AugmentationConfig.Preprocessing(
        output_image_width=width,
        output_image_height=height,
        output_image_channel=num_channels,
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
    "hflip_probability,expected_orientation,num_channels", [(0.0, np.pi / 4., 1),
                                                            (0.0, np.pi / 4., 3),
                                                            (1.0, -np.pi / 4., 1),
                                                            (1.0, -np.pi / 4., 3)]
)
def test_get_dataset_tensors_produces_additional_labels(tmpdir,
                                                        hflip_probability,
                                                        expected_orientation,
                                                        num_channels):
    """Test that get_dataset_tensors produces the expected additional tensors."""
    set_random_seed(123)
    num_samples, height, width, num_channels = 1, 10, 20, num_channels
    tfrecords_path, image_directory_path = \
        generate_dummy_dataset(tmpdir=tmpdir,
                               num_samples=num_samples,
                               height=height,
                               width=width,
                               labels=None,
                               num_channels=num_channels)
    training_data_source_list = [(tfrecords_path, image_directory_path)]
    target_class_mapping = dict()
    validation_data_source_list = None
    validation_fold = 1

    augmentation_config = get_hflip_augmentation_config(hflip_probability=hflip_probability,
                                                        num_channels=num_channels)
    # if not augmentation_config.preprocessing.input_mono:
    dataloader = DefaultDataloader(
        training_data_source_list=training_data_source_list,
        target_class_mapping=target_class_mapping,
        image_file_encoding='png',
        validation_fold=validation_fold,
        validation_data_source_list=validation_data_source_list,
        augmentation_config=augmentation_config)

    labels_tensors = dataloader.get_dataset_tensors(
        batch_size=num_samples,
        training=True,
        enable_augmentation=True)[1]

    with tf.Session() as sess:
        sess.run(get_init_ops())
        labels = sess.run(labels_tensors)

    for frame_labels in labels:
        assert np.allclose(frame_labels['target/orientation'], expected_orientation)
