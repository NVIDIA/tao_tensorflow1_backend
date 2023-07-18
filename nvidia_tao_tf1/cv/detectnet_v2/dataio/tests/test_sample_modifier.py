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

"""Tests for SampleModifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import six
from six.moves import range
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _bytes_feature
from nvidia_tao_tf1.cv.detectnet_v2.dataio.build_sample_modifier import build_sample_modifier
import nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_export_config_pb2 as\
    dataset_export_config_pb2


class TestSampleModifier:
    """Tests for SampleModifier."""

    @pytest.mark.parametrize(
        "objects_in_sample,source_to_target_class_mapping,filtered",
        [([b'automobile', b'van'], {b'automobile': b'car', b'van': b'car'}, True),
         # Even though 'cat' is not defined in the mapping, there should be no filtering happening,
         #  since 'dog' is.
         ([b'heavy_truck', b'dog', b'cat'], {b'heavy_truck': b'car', b'dog': b'animal'}, False),
         # In this case, only 'heavy_truck' is mapped, the frame should be filtered.
         ([b'heavy_truck', b'dog', b'cat'], {b'heavy_truck': b'car'}, True)
         ])
    def test_filter_samples(self, objects_in_sample, source_to_target_class_mapping, filtered):
        """Test filtering samples that contain objects only in one class."""
        sample_modifier_config = \
            dataset_export_config_pb2.DatasetExportConfig.SampleModifierConfig()
        # Assign class mapping.
        for source_class_name, target_class_name in six.iteritems(source_to_target_class_mapping):
            sample_modifier_config.source_to_target_class_mapping[source_class_name] = \
                target_class_name
        sample_modifier_config.filter_samples_containing_only.extend(['car'])

        sample_modifier = build_sample_modifier(sample_modifier_config=sample_modifier_config,
                                                validation_fold=0)

        example = tf.train.Example(features=tf.train.Features(feature={
            'target/object_class': _bytes_feature(*objects_in_sample),
        }))

        filtered_samples = sample_modifier._filter_sample(example)
        if filtered:
            assert filtered_samples is None
        else:
            assert filtered_samples == example

    @pytest.mark.parametrize(
        "objects_in_sample,minimum_target_class_imbalance,"
        "source_to_target_class_mapping,num_duplicates,num_expected_samples",
        # Number of canines / number of cars = 2.0 > 1.0 => Should be duplicated.
        [([b'automobile', b'dog', b'dog', b'cat'], 1.0, {b'automobile': b'car',
                                                         b'dog': b'canine'}, 1, 2),
         # Number of canine / number of cars = 1.0 => Should not be duplicated.
         ([b'automobile', b'dog'], 1.0, {b'automobile': b'car', b'dog': b'canine'}, 1, 1),
         # Number of canine / number of cars = 1.0 > 0.5 => Should be duplicated.
         ([b'automobile', b'dog'], 0.5, {b'automobile': b'car', b'dog': b'canine'}, 2, 3),
         # Number of canine / number of cars = 0.33 < 0.5 => Should not be duplicated.
         ([b'automobile', b'automobile', b'automobile', b'dog'], 0.5,
          {b'automobile': b'car', b'dog': b'canine'}, 1, 1)
         ])
    def test_duplicate_samples(self, objects_in_sample, minimum_target_class_imbalance,
                               source_to_target_class_mapping, num_duplicates,
                               num_expected_samples):
        """Test sample duplication.

        Test that samples that fulfill the condition
        number of rare class / number of dominant class > minimum_imbalance
        are duplicated.
        """
        sample_modifier_config = \
            dataset_export_config_pb2.DatasetExportConfig.SampleModifierConfig()
        # Assign class mapping.
        for source_class_name, target_class_name in six.iteritems(source_to_target_class_mapping):
            sample_modifier_config.source_to_target_class_mapping[source_class_name] = \
                target_class_name
        sample_modifier_config.dominant_target_classes.extend([b'car'])
        for target_class_name in set(source_to_target_class_mapping.values()):
            sample_modifier_config.minimum_target_class_imbalance[target_class_name] = \
                minimum_target_class_imbalance
        sample_modifier_config.minimum_target_class_imbalance[b'car'] = 1.0
        sample_modifier_config.num_duplicates = num_duplicates

        sample_modifier = build_sample_modifier(sample_modifier_config=sample_modifier_config,
                                                validation_fold=0)

        example = tf.train.Example(features=tf.train.Features(feature={
            'target/object_class': _bytes_feature(*objects_in_sample),
        }))

        duplicated_samples = sample_modifier._duplicate_sample(example)

        assert duplicated_samples == [example]*num_expected_samples

    @pytest.mark.parametrize("in_training_set", [True, False])
    def test_in_training_set(self, in_training_set):
        """Test that a sample is modified only if it belongs to the training set."""
        # Configure a SampleModifier and create a dummy sample that should be filtered if the
        # sample belongs to the training set.
        sample_modifier_config = \
            dataset_export_config_pb2.DatasetExportConfig.SampleModifierConfig()
        # Assign class mapping.
        sample_modifier_config.source_to_target_class_mapping[b'cvip'] = b'car'
        sample_modifier_config.filter_samples_containing_only.extend([b'car'])

        validation_fold = 0
        sample_modifier = build_sample_modifier(sample_modifier_config=sample_modifier_config,
                                                validation_fold=validation_fold)

        example = tf.train.Example(features=tf.train.Features(feature={
            'target/object_class': _bytes_feature(*['cvip', 'cvip']),
        }))

        validation_fold = validation_fold + 1 if in_training_set else validation_fold
        modified_samples = sample_modifier.modify_sample(example, validation_fold)

        expected = [] if in_training_set else [example]

        assert modified_samples == expected

    @pytest.mark.parametrize("objects_in_sample", [[b'car', b'person', b'person'],
                             [b'car', b'person']])
    def test_no_modifications(self, objects_in_sample):
        """Test that no modifications are done if the modification parameters are not set."""
        sample_modifier_config = \
            dataset_export_config_pb2.DatasetExportConfig.SampleModifierConfig()

        sample_modifier = build_sample_modifier(sample_modifier_config=sample_modifier_config,
                                                validation_fold=0)

        example = tf.train.Example(features=tf.train.Features(feature={
            'target/object_class': _bytes_feature(*objects_in_sample),
        }))

        modified_samples = sample_modifier.modify_sample(example, sample_modifier.validation_fold)

        assert modified_samples == [example]

    @pytest.mark.parametrize("objects_in_sample, folds, num_samples, max_training_samples,"
                             "validation_fold", [([b'car', b'person'], 5, 50, 25, 0),
                                                 ([b'car', b'person'], 4, 30, 20, None)])
    def test_max_num_training_samples(self, objects_in_sample, folds, num_samples,
                                      max_training_samples, validation_fold):
        """Test that no more than max_per_training_fold are retained in each training fold."""
        sample_modifier_config = \
            dataset_export_config_pb2.DatasetExportConfig.SampleModifierConfig()
        sample_modifier_config.max_training_samples = max_training_samples

        sample_modifier = build_sample_modifier(sample_modifier_config=sample_modifier_config,
                                                validation_fold=validation_fold,
                                                num_folds=folds)

        if validation_fold is None:
            expected_num_per_fold = max_training_samples // folds
        else:
            expected_num_per_fold = max_training_samples // (folds - 1)
            validation_fold = sample_modifier.validation_fold

        for fold in range(folds):
            for sample in range(num_samples // folds):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'target/object_class': _bytes_feature(*objects_in_sample),
                }))
                modified_samples = sample_modifier.modify_sample(example, fold)

                expect_retained = (fold == validation_fold) or (sample < expected_num_per_fold)
                expected_samples = [example] if expect_retained else []

                assert modified_samples == expected_samples
