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

"""Test for DriveNetLegacyMapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from parameterized import parameterized
import tensorflow as tf
import modulus
from nvidia_tao_tf1.blocks.multi_source_loader.processors.drivenet_legacy_mapper import (
    DriveNetLegacyMapper,
)


class TestDriveNetLegacyMapper(tf.test.TestCase):
    """Test for DriveNetLegacyMapper."""

    @parameterized.expand(
        [
            ("some class", "bottomWidth", "leftRight", "some_class", 3, 2),
            ("some other class", "bottom", "bottomRight", "some_other_class", 1, 3),
            ("no class", "width", "other bogus value", "no_class", 2, 0),
            ("school in july", "bogus value", "right", "school_in_july", 0, 2),
            ("heavy truck", "unknown", "bottomLeftRight", "heavy_truck", 0, 3),
            ("big SUV", "full", "bottom", "big_SUV", 0, 1),
            ("mini cooper", "Unknown", "bottomLeft", "mini_cooper", 0, 3),
            ("crossover", "?", "left", "crossover", 0, 2),
            ("luxury sedan", "not sure", "full", "luxury_sedan", 0, 0),
            ("sports car", "bottom", "unknown", "sports_car", 1, 0),
        ]
    )
    def test_process(
        self,
        object_class,
        occlusion,
        truncation_type,
        expected_object_class,
        expected_occlusion,
        expected_truncation_type,
    ):

        row = [
            u"cyclops-c",
            0,
            46835,
            0,
            0,
            u"cyclops-c",
            48,
            "",
            u"004f2c28-6bf8-5596-9216-963a141e7775",
            0,
            0,
            u"120FOV",
            u"none",
            u"video_1_front_center_120FOV_cyclops",
            1008,
            1920,
            0,
            [],
            0.26423147320747375,
            u"automobile",
            1,
            0,
            0,
            0,
            u"width",
            u"full",
            [[50.5, 688.83], [164.28, 766.8]],
            None,
            None,  # These 2 are the fields added by "add_fields".
            u"BOX",
        ]

        label_indices = {
            "BOX": {
                "is_cvip": 21,
                "occlusion": 24,
                "vertices": 26,
                "back": 18,
                "non_facing": 22,
                "front": 20,
                "attributes": 17,
                "classifier": 19,
                "num_vertices": 23,
                "truncation": 25,
                "mapped_occlusion": 27,
                "mapped_truncation": 28,
            },
            "dtype": 29,
        }

        row[label_indices["BOX"]["classifier"]] = object_class
        row[label_indices["BOX"]["truncation"]] = truncation_type
        row[label_indices["BOX"]["occlusion"]] = occlusion

        example = modulus.types.types.Example(instances=None, labels=label_indices)
        mapper = DriveNetLegacyMapper()

        keep = mapper.filter(example, "BOX", row)
        assert keep

        mapped_row = mapper.map(example, "BOX", row)

        assert mapped_row[label_indices["BOX"]["classifier"]] == expected_object_class
        assert (
            mapped_row[label_indices["BOX"]["mapped_truncation"]]
            == expected_truncation_type
        )
        assert (
            mapped_row[label_indices["BOX"]["mapped_occlusion"]] == expected_occlusion
        )
