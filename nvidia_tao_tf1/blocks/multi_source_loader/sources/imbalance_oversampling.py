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

"""Oversampling strategy using ratios between rare and dominant classes.

This strategy is meant to mimic the one used during TFRecords generation for DriveNet via the
dlav.drivenet.dataio.sample_modifier.

The concepts to take note of here:
    * source class: This typically corresponds to the 'classifier' feature found in SQLite HumanLoop
      exports. In the particular use case of object detection, typical values include 'automobile',
      'cvip', 'heavy truck', ...
    * target class: This corresponds to what one may wish to map a source class to. e.g. one may
      map 'automobile' and 'cvip' to the same target class 'car', or 'rider' and 'person' to
      'person', etc.
    * dominant target classes / rare classes: The latter is taken implicitly as whatever is _not_
      a dominant target class. Dominant target classes have in practice been the more represented
      classes, such as 'car' or 'road sign', the implication being that 'bicycle' or 'person' were
      the rare target classes.

In a given frame, if the # (any rare target class) > some factor * # (any dominant target class),
then that frame is duplicated a specified amount of times. 'some factor' corresponds to the entry
minimum_target_class_imbalance[rare_target_class].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

from nvidia_tao_tf1.core.dataloader.dataset import OverSamplingStrategy
from nvidia_tao_tf1.core.coreobject import save_args


class ImbalanceOverSamplingStrategy(OverSamplingStrategy):
    """Oversampling strategy using ratios between rare and dominant classes."""

    @save_args
    def __init__(
        self,
        dominant_target_classes,
        minimum_target_class_imbalance,
        num_duplicates,
        source_to_target_class_mapping,
    ):
        """Constructor.

        Args:
            dominant_target_classes (list): List of strings indicating the dominant target classes:
                target classes to be considered as dominant when determining whether to duplicate a
                sample.
            minimum_target_class_imbalance (dict): Target class - float pairs indicating the
                minimum imbalance determining when to duplicate. Basically, if the class imbalance
                within the frame is larger than this, duplicate. E.g. if
                #bicycles / #dominant class objects > minimum_target_class_imbalance[bicycle],
                duplicate. Default value for a class is 1.0 if not given.
            num_duplicates (int): Number of duplicate samples to be added when the duplication
                condition above is fulfilled. If a sample is to be duplicated, it will appear
                (num_duplicates + 1) times in total.
            source_to_target_class_mapping (dict): Mapping from label/source classes to
                target classes.
        """
        self._dominant_target_classes = dominant_target_classes
        self._minimum_target_class_imbalance = minimum_target_class_imbalance
        self._num_duplicates = num_duplicates
        self._source_to_target_class_mapping = source_to_target_class_mapping

    def oversample(self, frame_groups, count_lookup):
        """Determines which frames to oversample.

        Args:
            frame_groups (list): List of list of tuples. Outer list is over frame groups,
                inner list contains a tuple of (frame id(int), unlabeled(bool)).
            count_lookup (dict): Maps from frame ID (int) to another dict, that maps from
                classifier (str) to occurrence (int).

        Returns:
            repeated_groups (list): Follows the same structure as `frame_groups`. It should
                contain the frames that are to be repeated.
        """
        repeated_groups = []
        for frame_group in frame_groups:
            num_duplicates = 1

            for frame_id, _ in frame_group:
                class_counts = Counter()
                for classifier, classifier_lookup in count_lookup[frame_id].items():
                    count = classifier_lookup["COUNT"]
                    if classifier in self._source_to_target_class_mapping:
                        class_counts[
                            self._source_to_target_class_mapping[classifier]
                        ] += count

                rare_target_classes = [
                    class_id
                    for class_id in class_counts
                    if class_id not in self._dominant_target_classes
                ]

                repeat = any(
                    class_counts[rare_target_class]
                    > class_counts[dominant_target_class]
                    * self._minimum_target_class_imbalance[rare_target_class]
                    for rare_target_class in rare_target_classes
                    for dominant_target_class in self._dominant_target_classes
                )

                if repeat:
                    num_duplicates += self._num_duplicates

            for _ in range(num_duplicates):
                repeated_groups.append(frame_group)

        return repeated_groups
