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

"""Test loss mask filter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import BaseLabelFilter
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_dimensions_label_filter import (
    BboxDimensionsLabelFilter
)
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.source_class_label_filter import (
    SourceClassLabelFilter
)
from nvidia_tao_tf1.cv.detectnet_v2.objectives.objective_label_filter import ObjectiveLabelFilter
from nvidia_tao_tf1.cv.detectnet_v2.objectives.objective_label_filter_config import (
    ObjectiveLabelFilterConfig
)


# Some dummy learnable_objective_names.
_LEARNABLE_OBJECTIVE_NAMES = ['cov_norm']


class TestObjectiveLabelFilter:
    def test_objective_label_filter_init_assert(self):
        # Get init args.
        label_filter_configs = \
            [ObjectiveLabelFilterConfig(label_filter=BaseLabelFilter(),
                                        target_class_names=["car"])]
        with pytest.raises(AssertionError):
            # Since the class mapping is missing "car", it should fail.
            ObjectiveLabelFilter(label_filter_configs,
                                 dict(person=["pedestrian",
                                              "sentient_lifeform"]),
                                 _LEARNABLE_OBJECTIVE_NAMES)

        # This one should be fine.
        ObjectiveLabelFilter(label_filter_configs, dict(car=["panamera"]),
                             _LEARNABLE_OBJECTIVE_NAMES)

    @pytest.mark.parametrize(
        "model_label_filter_configs,target_class_to_source_classes_mapping,expected_structure",
        [
            # Case 1: Both filters apply to everything.
            ([ObjectiveLabelFilterConfig(BaseLabelFilter()),
              ObjectiveLabelFilterConfig(BboxDimensionsLabelFilter())],
             dict(person=['person', 'rider'], car=['automobile', 'truck']),
             {'person':
                 {'cov_norm': [BaseLabelFilter, BboxDimensionsLabelFilter]},
              'car':
                 {'cov_norm': [BaseLabelFilter, BboxDimensionsLabelFilter]}
              }
             ),  # ----- End case 1
            # Case 2: Each filter only applies to a single target class.
            ([ObjectiveLabelFilterConfig(SourceClassLabelFilter(), target_class_names=['car']),
              ObjectiveLabelFilterConfig(BboxDimensionsLabelFilter(),
                                         target_class_names=['person'])],
             dict(person=['person', 'rider'], car=['automobile', 'truck']),
             {'person':
                 {'cov_norm': [BboxDimensionsLabelFilter]},
              'car':
                 {'cov_norm': [SourceClassLabelFilter]}
              }
             ),  # ----- End case 2
            # Case 3: Each filter applies to a different target class and objective.
            ([ObjectiveLabelFilterConfig(SourceClassLabelFilter(),
                                         target_class_names=['truck'],
                                         objective_names=['bbox']),
              ObjectiveLabelFilterConfig(BboxDimensionsLabelFilter(),
                                         target_class_names=['car'],
                                         objective_names=['depth']),
              ObjectiveLabelFilterConfig(BaseLabelFilter(),
                                         target_class_names=['person'],
                                         objective_names=['orientation']),
              ObjectiveLabelFilterConfig(BboxDimensionsLabelFilter(),
                                         target_class_names=['truck'],
                                         objective_names=['bbox'])],
             dict(person=['pedestrian'], car=[
                  'automobile', 'van'], truck=['otto', 'pacar']),
             {'person':
                 {'orientation': [BaseLabelFilter]},
              'car':
                 {'depth': [BboxDimensionsLabelFilter]},
              'truck':
                 {'bbox': [SourceClassLabelFilter, BboxDimensionsLabelFilter]}
              }  # ----- End case 3
             )
        ]
    )
    def test_get_label_filter_lists(self,
                                    model_label_filter_configs,
                                    target_class_to_source_classes_mapping,
                                    expected_structure):
        """Test that the ObjectiveLabelFilter builds an inner hierarchy that is the expected one."""
        # Get the ObjectiveLabelFilter.
        objective_label_filter = ObjectiveLabelFilter(model_label_filter_configs,
                                                      target_class_to_source_classes_mapping,
                                                      _LEARNABLE_OBJECTIVE_NAMES)
        # Check that the correct 'hierarchy' was built internally.
        filter_lists = objective_label_filter._label_filter_lists
        assert set(filter_lists.keys()) == set(expected_structure.keys())
        # Now inner keys.
        for target_class_name in expected_structure:
            assert set(filter_lists[target_class_name].keys()) == \
                set(expected_structure[target_class_name].keys())
            for objective_name in expected_structure[target_class_name]:
                # Check that the LabelFilter objects are of the correct instance.
                # Note that order matters.
                assert all(map(lambda x: isinstance(*x),
                               zip(filter_lists[target_class_name][objective_name],
                                   expected_structure[target_class_name][objective_name])))

    @pytest.mark.parametrize(
        "model_label_filter_configs,batch_labels,target_class_to_source_classes_mapping,"
        "expected_output",
        [
            # Case 1: No kwargs for ObjectiveLabelFilterConfig --> should be no-ops.
            ([ObjectiveLabelFilterConfig(BboxDimensionsLabelFilter()),
              ObjectiveLabelFilterConfig(SourceClassLabelFilter())],
             [{'target/object_class': ['automobile', 'pedestrian']},  # 1st frame.
              {'target/object_class': ['pedestrian']}],  # Second frame.
             # The following line indicates that the output dict should only have this class.
             {'car': ['automobile']},
             # Since we supplied no objective_names, it should be for 'cov_norm'.
             {'car': {'cov_norm': [{'target/object_class': ['automobile', 'pedestrian']},  # frame1.
                                   {'target/object_class': ['pedestrian']}]}  # Second frame.
              }),
            # -------- End case 1.
            # Case 2: Only keep 'person' labels.
            ([ObjectiveLabelFilterConfig(SourceClassLabelFilter(source_class_names=['pedestrian']),
                                         target_class_names=['person'])],
             [{'target/object_class': ['automobile', 'pedestrian']},
              {'target/object_class': ['pedestrian']}],
             # The following line indicates that the output dict should only have this class.
             {'car': ['automobile'], 'person': ['pedestrian']},
             # Since we supplied no objective_names, it should be for 'cov_norm'.
             {'person': {'cov_norm': [{'target/object_class': ['pedestrian']},
                                      {'target/object_class': ['pedestrian']}]}
              }),
            # -------- End case 2.
            # Case 3:
            ([ObjectiveLabelFilterConfig(BboxDimensionsLabelFilter(min_width=10.0),
                                         target_class_names=['person']),
              ObjectiveLabelFilterConfig(SourceClassLabelFilter(source_class_names=['automobile']),
                                         target_class_names=['car'],
                                         objective_names=['depth'])],
             [{'target/object_class': ['automobile', 'pedestrian'],
               'target/coordinates_x1': np.array([20.0, 30.0], dtype=np.float32),
               # 'person' should be gone for 'person' because of width.
               'target/coordinates_x2': np.array([31.0, 39.9], dtype=np.float32),
               'target/coordinates_y1': np.array([23.0, 24.0], dtype=np.float32),
               'target/coordinates_y2': np.array([23.1, 24.1], dtype=np.float32),
               'target/bbox_coordinates': \
               np.array([[20.0, 23.0, 29.0, 23.1], [30.0, 24.0, 39.9, 24.1]], dtype=np.float32)},
              {'target/object_class': ['pedestrian'],
               # This one is above the min_width so should be kept.
               'target/coordinates_x1': np.array([10.0], dtype=np.float32),
               'target/coordinates_x2': np.array([20.1], dtype=np.float32),
               'target/coordinates_y1': np.array([0.0], dtype=np.float32),
               'target/coordinates_y2': np.array([123.0], dtype=np.float32),
               'target/bbox_coordinates': np.array([[10.0, 0.0, 20.1, 123.0]],
                                                   dtype=np.float32)
               }],
             # The following line indicates that the output dict should only have this class.
             {'car': ['automobile'], 'person': ['pedestrian']},
             # Since we supplied no objective_names, it should be for 'cov_norm'.
             {'person':
                {'cov_norm':
                 [{'target/object_class': np.array([]).astype(str),
                   'target/coordinates_x1': np.array([], dtype=np.float32),
                   'target/coordinates_x2': np.array([], dtype=np.float32),
                   'target/coordinates_y1': np.array([], dtype=np.float32),
                   'target/coordinates_y2': np.array([], dtype=np.float32),
                   'target/bbox_coordinates': np.empty([0, 4], dtype=np.float32)
                   },  # End first frame.
                  {'target/object_class': np.array(['pedestrian']),
                   'target/coordinates_x1': np.array([10.0], dtype=np.float32),
                   'target/coordinates_x2': np.array([20.1], dtype=np.float32),
                   'target/coordinates_y1': np.array([0.0], dtype=np.float32),
                   'target/coordinates_y2': np.array([123.0], dtype=np.float32),
                   'target/bbox_coordinates': np.array([[10.0, 0.0, 20.1, 123.0]],
                                                       dtype=np.float32)}]  # End 2nd frame.
                 },  # End ['person']['cov_norm'].
              'car':
                {'depth':
                 [{'target/object_class': np.array(['automobile']),
                   'target/coordinates_x1': np.array([20.0], dtype=np.float32),
                   'target/coordinates_x2': np.array([31.0], dtype=np.float32),
                   'target/coordinates_y1': np.array([23.0], dtype=np.float32),
                   'target/coordinates_y2': np.array([23.1], dtype=np.float32),
                   'target/bbox_coordinates': np.array([[20.0, 23.0, 29.0, 23.1]],
                                                       dtype=np.float32)
                   },  # End first frame.
                  {'target/object_class': np.array([]).astype(str),
                   'target/coordinates_x1': np.array([], dtype=np.float32),
                   'target/coordinates_x2': np.array([], dtype=np.float32),
                   'target/coordinates_y1': np.array([], dtype=np.float32),
                   'target/coordinates_y2': np.array([], dtype=np.float32),
                   'target/bbox_coordinates': np.empty([0, 4], dtype=np.float32),
                   }],  # End 2nd frame.
                 }  # End 'depth'.
              }  # End 'car', end <expected_output>.
             ),  # -------- End case 3.
        ]
    )
    def test_apply_filters(
            self,
            model_label_filter_configs,
            batch_labels,
            target_class_to_source_classes_mapping,
            expected_output):
        # First, get the ObjectiveLabelFilter.
        objective_label_filter = ObjectiveLabelFilter(model_label_filter_configs,
                                                      target_class_to_source_classes_mapping,
                                                      _LEARNABLE_OBJECTIVE_NAMES)
        _filtered_labels = objective_label_filter.apply_filters(batch_labels)
        with tf.compat.v1.Session() as sess:
            filtered_labels = sess.run(_filtered_labels)

        # Check the filtering matches our expectations.
        assert set(filtered_labels.keys()) == set(expected_output.keys())
        for target_class_name in filtered_labels:
            assert set(filtered_labels[target_class_name].keys()) == \
                set(expected_output[target_class_name].keys())
            for objective_name in filtered_labels[target_class_name]:
                # Check all the frames from the batch are present.
                assert len(filtered_labels[target_class_name][objective_name]) == \
                    len(expected_output[target_class_name][objective_name])
                # Now check all the individual frames match our expectations.
                for i in range(len(filtered_labels[target_class_name][objective_name])):
                    filtered_frame = filtered_labels[target_class_name][objective_name][i]
                    expected_frame = expected_output[target_class_name][objective_name][i]
                    assert set(filtered_frame.keys()) == set(
                        expected_frame.keys())
                    # Check all features are filtered correctly (both order and value).
                    for feature_name in filtered_frame:
                        feature_frame = filtered_frame[feature_name]
                        if feature_frame.dtype.str.startswith("|O"):
                            feature_frame = feature_frame.astype(str)
                        assert np.array_equal(feature_frame,
                                              expected_frame[feature_name])
