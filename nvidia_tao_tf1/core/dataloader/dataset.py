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
"""TAO Core dataset objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import collections
import logging

from nvidia_tao_tf1.core.coreobject import (
    AbstractTAOObject,
    TAOObject,
    save_args
)

logger = logging.getLogger(__name__)


# TODO(@williamz): Consider moving all this into the `DefaultOverSamplingStrategy` constructor.
class OverSamplingConfig(TAOObject):
    """Config object meant to accompany DefaultOverSamplingConfig.

    Configure oversampling for the data loader based on label classes ('classifier').

    There are two modes:
      OverSamplingConfig.LABEL_OCCURENCE: based on whether frame has or hasn't a label. If
        frame has label, then multiply based on the count given in 'class_id_base_counts.'
      OverSamplingConfig.INSTANCE_COUNT: scales repetition based on the number of instances of
        given label. Say there are 3 occurences of label A in the frame, and the base count
        for A is 6. Then the frame is repeated 6 / 3 = 2 times.
    """

    INSTANCE_COUNT = 0
    LABEL_OCCURENCE = 1

    @save_args
    def __init__(self, class_to_id_map, class_id_base_counts, mode):
        """Constructor.

        Note: the class names in class_to_id_map must be stripped from whitespace and lowercase:
        class_name.strip().lower().

        Args:
            class_to_id_map (dict[str, int]): Mapping from classifier name to class id number.
            class_id_base_counts(dict[int, float]): For each class id, the base count (see above.)
            mode (int): OverSamplingConfig.INSTANCE_COUNT|LABEL_OCCURENCE
        """
        if set(class_to_id_map.values()).difference(set(class_id_base_counts.keys())):
            raise ValueError(
                "Not all values in class_to_id_map have base counts, missing: {}".format(
                    set(class_to_id_map.values()).difference(
                        set(class_id_base_counts.keys())
                    )
                )
            )
        self.class_to_id_map = class_to_id_map
        self.class_id_base_counts = class_id_base_counts
        self.mode = mode

        for class_name in class_to_id_map:
            if class_name.strip().lower() != class_name:
                raise RuntimeError(
                    "class_to_id_map must have strip().lower()'ed' class names."
                )


# TODO(@williamz): This could technically be "abused" to implement filtering of frame IDs.
# Consider renaming if this is useful in practice.
class OverSamplingStrategy(AbstractTAOObject):
    """Interface for oversampling frames."""

    @abstractmethod
    def oversample(self, frame_groups, count_lookup):
        """Determines which frames to oversample.

        Note: Oversampling could lead to some frames being repeated a lot which can have an affect
        on random shuffling.

        Args:
            frame_groups (list): List of list of tuples. Outer list is over frame groups,
                inner list contains a tuple of (frame_id(int), unused).
            count_lookup (dict):  dict(frame_id -> dict(class_name -> dict(attribute_sets ->  cnt)))
                class_count[<frame id>][<classifier>][<attribute set>] -> class count for specific
                    attribute set in that frame.
                class_count[<frame id>][<classifier>]["COUNT"] -> count of all instances of
                    <classifier> in that frame.

        Returns:
            repeated_groups (list): Follows the same structure as `frame_groups`. It should
                contain the frames that are to be repeated.
        """
        raise NotImplementedError(
            "This method is not implemented in the base class."
        )


class DefaultOverSamplingStrategy(OverSamplingStrategy):
    """Default strategy for oversampling."""

    def __init__(self, oversampling_config):
        """Constructor.

        Args:
            oversampling_config (OverSamplingConfig).
        """
        self._oversampling_config = oversampling_config

    def oversample(self, frame_groups, count_lookup):
        """Determines which frames to oversample.

        Args:
            frame_groups (list): List of list of tuples. Outer list is over frame groups,
                inner list contains a tuple of (frame id(int), unlabeled(bool)).
            count_lookup (dict):  dict(frame_id -> dict(class_name -> dict(attribute_sets ->  cnt)))
                class_count[<frame id>][<classifier>][<attribute set>] -> class count for specific
                    attribute set in that frame.
                class_count[<frame id>][<classifier>]["COUNT"] -> count of all instances of
                    <classifier> in that frame.

        Returns:
            repeated_groups (list): Follows the same structure as `frame_groups`. It should
                contain the frames that are to be repeated.
        """
        repeated_groups = []
        for frame_group in frame_groups:
            class_counts = collections.Counter()
            class_to_id_map = self._oversampling_config.class_to_id_map

            for frame_id, _ in frame_group:
                if self._oversampling_config.mode == OverSamplingConfig.INSTANCE_COUNT:
                    classifier_map = count_lookup[frame_id]
                    for classifier in classifier_map.keys():
                        class_counts[class_to_id_map[classifier]] += float(
                            classifier_map[classifier]["COUNT"]
                        )
                elif (
                    self._oversampling_config.mode == OverSamplingConfig.LABEL_OCCURENCE
                ):
                    for classifier in count_lookup[frame_id]:
                        if classifier in class_to_id_map:
                            class_counts[class_to_id_map[classifier]] = 1

                else:
                    raise ValueError(
                        "Unknown oversampling mode: {}".format(
                            self._oversampling_config.mode
                        )
                    )

            repeats = 1
            for class_id, count in class_counts.items():
                repeats += (
                    self._oversampling_config.class_id_base_counts[class_id] / count
                )

            repeats = int(repeats + 0.5)
            for _ in range(repeats):
                repeated_groups.append(frame_group)

        return repeated_groups


class _DerivedField(object):
    """A synthetic field, whose value is computed based on other fields."""

    def __init__(self, value_type, shape=None):
        self.select = ""
        self.value_type = value_type
        self.frame_field = True
        self.label_type = None
        self.frame_id_key = False
        self.label_type_field = False
        self.select = "0"
        self.shape = shape
        self.prune = False

    def __repr__(self):
        return "DerivedField object: {}".format(self.__dict__)


def create_derived_field(value_type, shape):
    """Creates a field whose value is computed at read-time based on other fields."""
    return _DerivedField(value_type, shape)


class FeatureProcessor(AbstractTAOObject):
    """Class that filters and maps feature rows."""

    @abstractmethod
    def add_fields(self, example):
        """
        Add new fields to the example data structure (labels).

        Example:
            example.labels['BOX']['testfield_int'] = create_derived_field(tf.int32, shape=None)

        Args:
            example (namedtuple): data structure that the loader returns.
        """
        raise NotImplementedError()

    @abstractmethod
    def filter(self, example_col_idx, dtype, row):
        """
        Filter label rows.

        Args:
            example_col_idx (namedtuple): example data structure, where fields are integers
                                          that correspond to the index of the value in 'row'
            dtype (str): label type, such as 'BOX' or 'POLYGON'.
            row (list): flat list of values from the database for one label. Use example_col_idx
                        to find which element corresponds to which field in the 'example'.

        Return:
            True or False, depending on whether the row should be kept.
        """
        raise NotImplementedError()

    @abstractmethod
    def map(self, example_col_idx, dtype, row):
        """
        Modify or inject values into the feature row.

        Args:
            example_col_idx (namedtuple): example data structure, where fields are integers
                                          that correspond to the index of the value in 'row'
            dtype (str): label type, such as 'BOX' or 'POLYGON'.
            row (list): flat list of values from the database for one label. Use example_col_idx
                        to find which element corresponds to which field in the 'example'.

        Return:
            modified 'row'.
        """
        raise NotImplementedError()


class FrameProcessor(AbstractTAOObject):
    """
    Extension object to modify / add fields to frames.

    Note: frames cannot be filtered, this must be done when resolving the frame ids to load
    (see _get_all_frames() etc). This is because otherwise there is no efficient way to
    compute the dataset size.
    """

    @abstractmethod
    def add_fields(self, example):
        """
        Add new fields to the example data structure (labels).

        Example:
            example.labels['BOX']['testfield_int'] = create_derived_field(tf.int32, shape=None)

        Args:
            example (namedtuple): data structure that the loader returns.
        """
        raise NotImplementedError()

    @abstractmethod
    def map(self, example_col_idx, frame):
        """
        Modify or inject values into the frame.

        Args:
            example_col_idx (namedtuple): example data structure, where fields are integers
                                          that correspond to the index of the value in 'row'
            frame (list): flat list of values from the database for a frame. Use example_col_idx
                        to find which element corresponds to which field in the 'example'.

        Return:
            modified 'frame'.
        """
        raise NotImplementedError()


def _default_value(shape):
    if shape is None:
        return None
    if len(shape) == 1:
        return []
    if len(shape) == 2:
        return [[]]
    raise ValueError(
        "Currently only support 1 or 2 dimensional shapes, given: {}".format(shape)
    )
