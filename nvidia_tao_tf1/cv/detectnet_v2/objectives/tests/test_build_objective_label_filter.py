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

"""Test objective label filter builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf.text_format import Merge as merge_text_proto

import pytest

from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_dimensions_label_filter import (
    BboxDimensionsLabelFilter
)
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.source_class_label_filter import (
    SourceClassLabelFilter
)
from nvidia_tao_tf1.cv.detectnet_v2.objectives.build_objective_label_filter import (
    build_objective_label_filter
)
import nvidia_tao_tf1.cv.detectnet_v2.proto.objective_label_filter_pb2 as \
    objective_label_filter_pb2


# Some dummy learnable_objective_names.
_LEARNABLE_OBJECTIVE_NAMES = ['cov_norm']


class TestObjectiveLabelFilterBuilder(object):
    @pytest.fixture(scope="function")
    def objective_label_filter_proto(self):
        """Generate a proto to build an ObjectiveLabelFilter with."""
        objective_label_filter_proto = objective_label_filter_pb2.ObjectiveLabelFilter()
        prototxt = """
objective_label_filter_configs {
    target_class_names: "car"
    target_class_names: "person"
    label_filter: {
        bbox_dimensions_label_filter: {
            min_width: 10.0
            min_height: 10.0
            max_width: 400.0
            max_height: 400.0
        }
    }
}
objective_label_filter_configs {
    target_class_names: "car"
    objective_names: "depth"
    label_filter: {
        source_class_label_filter: {
            source_class_names: "automobile"
        }
    }
}
objective_label_filter_configs {
    target_class_names: "car"
    objective_names: "depth"
    label_filter: {
        source_class_label_filter: {
            source_class_names: "van"
        }
    }
}
"""
        merge_text_proto(prototxt, objective_label_filter_proto)

        return objective_label_filter_proto

    @pytest.fixture(scope='function')
    def target_class_to_source_classes_mapping(self):
        target_class_to_source_classes_mapping = {
            'person': ['pedestrian', 'person_group', 'rider'],
            'car': ['heavy_truck', 'automobile', 'unclassifiable_vehicle']
        }
        return target_class_to_source_classes_mapping

    def test_objective_label_filter_builder(self,
                                            objective_label_filter_proto,
                                            target_class_to_source_classes_mapping):
        """Test that the builder for ObjectiveLabelFilter instantiates the object correctly.

        Args:
            objective_label_filter_proto (proto.objective_label_filter_pb2.ObjectiveLabelFilter)
            target_class_to_source_classes_mapping (dict): Maps from target class name (str) to
                a list of source class names (str).
        """
        objective_label_filter = \
            build_objective_label_filter(
                objective_label_filter_proto=objective_label_filter_proto,
                target_class_to_source_classes_mapping=target_class_to_source_classes_mapping,
                learnable_objective_names=_LEARNABLE_OBJECTIVE_NAMES)

        # TODO(@williamz): ideally, would check that ObjectiveLabelFilter was called with certain
        #  args. However, it would be a little convoluted to do so settling for this approach.
        label_filter_lists = objective_label_filter._label_filter_lists
        # Check that the default mask_multiplier value is correctly set.
        assert objective_label_filter.mask_multiplier == 0.0

        # Check that correct target class names have corresponding entries.
        expected_target_class_names = {'car', 'person'}
        assert set(label_filter_lists.keys()) == expected_target_class_names

        # Same check for objective names.
        assert set(label_filter_lists['car'].keys()) == {'cov_norm', 'depth'}
        assert set(label_filter_lists['person'].keys()) == {'cov_norm'}

        # Check that there is only one label filter that applies to all objectives ('cov_norm').
        for target_class_name in expected_target_class_names:
            assert len(label_filter_lists[target_class_name]['cov_norm']) == 1
            # Check that it is of the correct type.
            assert isinstance(label_filter_lists[target_class_name]['cov_norm'][0],
                              BboxDimensionsLabelFilter)

        # SourceClassLabelFilter is only applied to 'depth' + 'car' combo.
        assert 'depth' not in label_filter_lists['person']
        # Even though it would be stupid to actually duplicate the filter like in the prototxt
        #  above, check that there are two filters for this combo.
        assert len(label_filter_lists['car']['depth']) == 2
        for sub_filter in label_filter_lists['car']['depth']:
            # Check that it is of the correct type.
            assert isinstance(sub_filter, SourceClassLabelFilter)
