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

"""Build sample modifier to write to tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
from nvidia_tao_tf1.cv.detectnet_v2.dataio.sample_modifier import SampleModifier


def build_sample_modifier(sample_modifier_config, validation_fold, num_folds=None):
    """Build a SampleModifier object.

    Args:
        sample_modifier_config(SampleModifierConfig): Configuration of sample modifier.
        validation_fold (int): Validation fold number (0-based). If samples are modified, then the
            modifications are applied only to the training set, while the validation set
            remains unchanged.
        num_folds (int): The total number of folds.

    Return:
        sample_modifier(SampleModifier): The created SampleModifier instance.
    """
    # Convert unicode strings to python strings for class mapping.
    source_to_target_class_mapping = \
        {bytes(str(source_class_name), 'utf-8'): bytes(str(target_class_name), 'utf-8')
            for source_class_name, target_class_name
            in six.iteritems(sample_modifier_config.source_to_target_class_mapping)}
    sample_modifier = SampleModifier(
        filter_samples_containing_only=sample_modifier_config.filter_samples_containing_only,
        dominant_target_classes=sample_modifier_config.dominant_target_classes,
        minimum_target_class_imbalance=sample_modifier_config.minimum_target_class_imbalance,
        num_duplicates=sample_modifier_config.num_duplicates,
        max_training_samples=sample_modifier_config.max_training_samples,
        source_to_target_class_mapping=source_to_target_class_mapping,
        validation_fold=validation_fold,
        num_folds=num_folds)

    return sample_modifier
