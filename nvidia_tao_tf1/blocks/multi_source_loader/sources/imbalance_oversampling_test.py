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

"""Tests for ImbalanceOverSamplingStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from nvidia_tao_tf1.blocks.multi_source_loader.sources.imbalance_oversampling import (
    ImbalanceOverSamplingStrategy,
)
from modulus.dataloader.humanloop import example_template
from modulus.dataloader.humanloop_sqlite_dataset import HumanloopSqliteDataset
from modulus.dataloader.testdata.db_test_builder import DbTestBuilder


class TestOverSamplingStrategy(object):
    """Tests for ImbalanceOverSamplingStrategy."""

    @pytest.mark.parametrize(
        "count_lookup, minimum_target_class_imbalance, source_to_target_class_mapping,"
        "num_duplicates, num_expected_duplicates",
        # Number of canines / number of cars = 2.0 > 1.0 => Should be duplicated.
        [
            (
                {"automobile": {"COUNT": 1}, "dog": {"COUNT": 2}},
                1.0,
                {"automobile": "car", "dog": "canine"},
                1,
                2,
            ),
            # Number of canine / number of cars = 1.0 => Should not be duplicated.
            (
                {"automobile": {"COUNT": 1}, "dog": {"COUNT": 1}},
                1.0,
                {"automobile": "car", "dog": "canine"},
                1,
                1,
            ),
            # Number of canine / number of cars = 1.0 > 0.5 => Should be duplicated.
            (
                {"automobile": {"COUNT": 1}, "dog": {"COUNT": 1}},
                0.5,
                {"automobile": "car", "dog": "canine"},
                2,
                3,
            ),
            # Number of canine / number of cars = 0.33 < 0.5 => Should not be duplicated.
            (
                {"automobile": {"COUNT": 3}, "dog": {"COUNT": 1}},
                0.5,
                {"automobile": "car", "dog": "canine"},
                1,
                1,
            ),
        ],
    )
    def test_oversample(
        self,
        count_lookup,
        minimum_target_class_imbalance,
        source_to_target_class_mapping,
        num_duplicates,
        num_expected_duplicates,
    ):
        """Test the oversample method."""
        dominant_target_classes = ["car"]

        minimum_target_class_imbalance = {
            target_class_name: minimum_target_class_imbalance
            for target_class_name in set(source_to_target_class_mapping.values())
        }

        oversampling_strategy = ImbalanceOverSamplingStrategy(
            dominant_target_classes=dominant_target_classes,
            minimum_target_class_imbalance=minimum_target_class_imbalance,
            num_duplicates=num_duplicates,
            source_to_target_class_mapping=source_to_target_class_mapping,
        )

        # 0-th and 2nd frames should appear once, while 1st frame should be duplicated.
        count_lookup = {0: {}, 1: count_lookup, 2: {}}
        frame_groups = [[(0, False)], [(1, False)], [(2, False)]]

        repeated_groups = oversampling_strategy.oversample(
            frame_groups=frame_groups, count_lookup=count_lookup
        )

        expected_groups = (
            [[(0, False)]] + [[(1, False)]] * num_expected_duplicates + [[(2, False)]]
        )

        assert expected_groups == repeated_groups

    @pytest.fixture
    def sqlite_path(self):
        db_test_builder = DbTestBuilder()

        sqlite_path = db_test_builder.get_db("test_dataset", 8)

        yield sqlite_path

        db_test_builder.cleanup()

    def test_interface(self, sqlite_path):
        """Smoke test that the interface is compatible with the HumanloopSqliteDataset."""
        dominant_target_classes = ["person"]
        source_to_target_class_mapping = {
            "automobile": "car",
            "heavy_truck": "car",
            "heavy truck": "car",
            "rider": "person",
        }
        num_duplicates = 3
        minimum_target_class_imbalance = {"car": 0.1}

        oversampling_strategy = ImbalanceOverSamplingStrategy(
            dominant_target_classes=dominant_target_classes,
            minimum_target_class_imbalance=minimum_target_class_imbalance,
            num_duplicates=num_duplicates,
            source_to_target_class_mapping=source_to_target_class_mapping,
        )

        regular_dataset = HumanloopSqliteDataset(
            filename=sqlite_path, export_format="jpeg", example=example_template()
        )
        oversampled_dataset = HumanloopSqliteDataset(
            filename=sqlite_path,
            export_format="jpeg",
            example=example_template(),
            oversampling_strategy=oversampling_strategy,
        )

        regular_dataset.set_batch(1, slice_batch=True)
        oversampled_dataset.set_batch(1, slice_batch=True)

        regular_frames = regular_dataset.num_unsharded_frames()
        oversampled_frames = oversampled_dataset.num_unsharded_frames()

        assert oversampled_frames > regular_frames
