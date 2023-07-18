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

"""Modify samples before writing them to .tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from six.moves import range


class SampleModifier(object):
    """Modify samples to be exported to .tfrecords.

    Currently, sample duplication and filtering are implemented.
    """

    def __init__(self, filter_samples_containing_only, dominant_target_classes,
                 minimum_target_class_imbalance, num_duplicates, max_training_samples,
                 source_to_target_class_mapping, validation_fold, num_folds=None):
        """Initialize a SampleModifier.

        Args:
            filter_samples_containing_only (list): List of strings indicating such target classes
                that will be filtered if the sample contains only that class.
            dominant_target_classes (list): List of strings indicating the dominant target classes:
                Target classes to be considered as dominant when determining whether to duplicate a
                sample.
            minimum_target_class_imbalance (dict): Target class - float pairs indicating the
                minimum imbalance determining when to duplicate. Basically if the class imbalance
                within the frame is larger than this, duplicate. E.g. if
                #bicycles / #dominant class objects > minimum_target_class_imbalance[bicycle],
                duplicate. Default value for a class is 1.0 if not given.
            num_duplicates (int): Number of duplicate samples to be added when the duplication
                condition above is fulfilled.
            max_training_samples (int): Maximum number of training samples. The number of training
                samples is capped to this number, i.e. any samples beyond it are filtered out.
            source_to_target_class_mapping (dict): Mapping from label/source classes to
                target classes.
            validation_fold (int): Validation fold number (0-based).
            num_folds (int): The total number of folds.
        """

        self.filter_samples_containing_only = filter_samples_containing_only
        self.dominant_target_classes = dominant_target_classes
        self.minimum_target_class_imbalance = minimum_target_class_imbalance
        self.num_duplicates = num_duplicates
        self.source_to_target_class_mapping = source_to_target_class_mapping
        self.validation_fold = validation_fold
        self.filter_samples_containing_only = [bytes(f_tmp, 'utf-8') for f_tmp in
                                               self.filter_samples_containing_only]
        self.dominant_target_classes = [bytes(f_tmp, 'utf-8') for
                                        f_tmp in self.dominant_target_classes]
        # Check that these two parameters have been defined in the mapping.

        assert set(self.dominant_target_classes) <= \
            set(self.source_to_target_class_mapping.values())

        assert set(self.filter_samples_containing_only) <= \
            set(self.source_to_target_class_mapping.values())

        if max_training_samples > 0:
            if num_folds is None:
                raise ValueError(("Number of folds must be specified if max_training_samples>0"))
            self.sample_counts = [0] * num_folds
            if validation_fold is not None:
                self.max_samples_per_training_fold = max_training_samples // (num_folds - 1)
            else:
                self.max_samples_per_training_fold = max_training_samples // num_folds
        else:
            self.max_samples_per_training_fold = 0

    def _is_in_training_set(self, fold):
        """Return True if the provided fold number is in the training set, otherwise False."""
        in_training_set = True

        if self.validation_fold is not None:
            if fold == self.validation_fold:
                in_training_set = False

        return in_training_set

    def modify_sample(self, example, fold):
        """Modify a sample if it belongs to the training set.

        If the validation set is not defined, then no changes are made to the sample.

        Args:
            example: tf.train.Example instance.
            fold (int): fold to add sample to.
        Return:
            examples: List of modified examples.
        """
        # Apply modifications only to the training set, i.e., exclude the validation examples.
        if self._is_in_training_set(fold):
            filtered_example = self._filter_sample(example)

            if filtered_example:
                examples = self._duplicate_sample(filtered_example)
            else:
                examples = []

            if self.max_samples_per_training_fold > 0:
                # Filter examples out if we have reached the max number of samples per fold.
                max_to_retain = self.max_samples_per_training_fold - self.sample_counts[fold]
                examples = examples[:max_to_retain]
                self.sample_counts[fold] += len(examples)
        else:
            examples = [example]

        return examples

    def _get_target_classes_in_sample(self, example):
        """Return target classes contained in the given sample.

        Args:
            example (tf.train.Example): The sample.

        Returns:
            target_classes_in_sample (list): List of strings
                indicating the target class names present in the sample.
        """
        source_classes_in_sample = \
            example.features.feature['target/object_class'].bytes_list.value
        src_mapping = [self.source_to_target_class_mapping.get(
            source_classes_in_sample_tmp) for source_classes_in_sample_tmp
            in source_classes_in_sample]
        target_classes_in_sample = [x for x in src_mapping if x is not None]

        return target_classes_in_sample

    def _filter_sample(self, example):
        """Filter samples based on the image contents and custom rules.

        Args:
            example: tf.train.Example.

        Return:
            filtered_example: None if the sample was filtered, otherwise return example as is.
        """
        filtered_example = example

        if self.filter_samples_containing_only:
            target_classes_in_sample = self._get_target_classes_in_sample(example)

            # Check whether the sample contains only objects in a single class.
            same_class = (len(set(target_classes_in_sample)) == 1 and
                          target_classes_in_sample[0] in self.filter_samples_containing_only)

            if same_class:
                filtered_example = None  # Filter the example.

        return filtered_example

    def _duplicate_sample(self, example):
        """Duplicate samples based on the image contents and custom rules.

        Args:
            example: tf.train.Example object.

        Return:
            duplicated_examples: A list of tf.train.Example objects. The list contains multiple
            copies of the sample if it is duplicated.
        """
        target_classes_in_sample = self._get_target_classes_in_sample(example)

        duplicated_examples = [example]
        if self.dominant_target_classes and target_classes_in_sample:

            # Count number of objects per target class in this sample.
            target_class_counts = Counter(target_classes_in_sample)

            # Ad-hoc rules for duplicating frames.If the imbalance
            # #rare class / #dominant class > minimum_imbalance[rare_class], then
            # duplicate.
            rare_target_classes = \
                [target_class_name for target_class_name in set(target_classes_in_sample)
                    if target_class_name not in self.dominant_target_classes]

            # Check if the minimum imbalance is exceeded for any class in this frame.
            minimum_imbalance_exceeded = \
                any([target_class_counts[rare_target_class] >
                     target_class_counts[dominant_target_class] *
                     self.minimum_target_class_imbalance.get(rare_target_class, 1.0)
                     for rare_target_class in rare_target_classes
                     for dominant_target_class in self.dominant_target_classes])

            if minimum_imbalance_exceeded:
                # Duplicate.
                for _ in range(self.num_duplicates):
                    duplicated_examples.append(example)

        return duplicated_examples
