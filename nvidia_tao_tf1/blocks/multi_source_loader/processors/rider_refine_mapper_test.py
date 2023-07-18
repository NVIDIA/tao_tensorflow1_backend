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

"""Test for RiderRefineMapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import modulus
from nvidia_tao_tf1.blocks.multi_source_loader.processors.rider_refine_mapper import (
    RiderRefineMapper,
)


class TestRiderRefineMapper(object):
    """Test for RiderRefineMapper."""

    @pytest.mark.parametrize(
        "object_class, attributes, expected_object_class",
        [
            # Below are attribute types with label_name = "rider" in training and test datasets.
            ("rider", ["other", "other_occlusion", "vehicle"], "vehicle_rider"),
            ("rider", ["other", "vehicle"], "vehicle_rider"),
            ("rider", ["other_occlusion", "vehicle"], "vehicle_rider"),
            ("rider", ["vehicle"], "vehicle_rider"),
            ("rider", ["bicycle", "other_occlusion"], "bicycle_rider"),
            ("rider", ["bicycle"], "bicycle_rider"),
            ("rider", ["other", "bicycle", "other_occlusion"], "bicycle_rider"),
            ("rider", ["other", "bicycle"], "bicycle_rider"),
            ("rider", ["motorcycle", "other"], "motorcycle_rider"),
            ("rider", ["motorcycle", "other_occlusion"], "motorcycle_rider"),
            ("rider", ["motorcycle", "other", "other_occlusion"], "motorcycle_rider"),
            ("rider", ["motorcycle"], "motorcycle_rider"),
            ("rider", ["bicycle", "vehicle"], "unknown_rider"),
            ("rider", ["motorcycle", "bicycle"], "unknown_rider"),
            ("rider", ["motorcycle", "vehicle"], "unknown_rider"),
            ("rider", ["other", "other_occlusion"], "unknown_rider"),
            ("rider", ["other"], "unknown_rider"),
            ("rider", ["other_occlusion"], "unknown_rider"),
            ("rider", [], "unknown_rider"),
            # Check the correctness of other source class.
            ("heavy truck", [], "heavy truck"),
        ],
    )
    def test_map(self, object_class, attributes, expected_object_class):
        """Test the correctness of map method in RiderRefineMapper."""
        dummy_row = [[], u"automobile", u"BOX"]

        dummy_label_indices = {"BOX": {"attributes": 0, "classifier": 1}, "dtype": 2}

        dummy_row[dummy_label_indices["BOX"]["classifier"]] = object_class
        dummy_row[dummy_label_indices["BOX"]["attributes"]] = attributes

        example = modulus.types.types.Example(
            instances=None, labels=dummy_label_indices
        )
        mapper = RiderRefineMapper()

        keep = mapper.filter(example, "BOX", dummy_row)
        assert keep

        mapped_row = mapper.map(example, "BOX", dummy_row)

        assert (
            mapped_row[dummy_label_indices["BOX"]["classifier"]]
            == expected_object_class
        )
