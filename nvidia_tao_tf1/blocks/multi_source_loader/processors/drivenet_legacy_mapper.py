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

"""DriveNet legacy mapper for occlusion and truncation labels.

The raison d'etre for this Processor is for DriveNet consumers of ``Bbox2DLabel`` instances to
maintain their interpretation of the occlusion and truncation fields that were inherited from
lossy TFRecords conversions. It also removes the need to adapt the class mapping part of a
DriveNet spec (see below why).

This allows other (potentially new) consumers of the SqliteDataSource requesting 'BOX' type labels
to consume ``Bbox2DLabel`` instances _without_ these lossy mappings applied (i.e. pretty much
what comes out of HumanLoop).

The lossy mappings that are applied concern:
    * truncation --> truncation_type.
    * occlusion

For historical reasons as well, class names had whitespaces replaced with underscores. This is also
taken care of here.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import modulus.dataloader.humanloop_sqlite_dataset as hl_sql


class DriveNetLegacyMapper(hl_sql.FeatureProcessor):
    """DriveNet legacy mapper for occlusion and truncation labels."""

    # 0 = fully visible (or unknown), 1 = bottom occluded,
    # 2 = width occluded, 3 = bottom and width occluded.
    # Note: this follows the json converter convention of mapping 'unknown' to fully visible.
    OCCLUSION_MAPPING = {
        "unknown": 0,
        "full": 0,
        "bottom": 1,
        "width": 2,
        "bottomWidth": 3,
    }

    # 0 = not truncated (or unknown), 1 = left/right/left&right truncated,
    # 2 = bottom truncated, 3 = bottom & (left/right/left&right) truncated.
    TRUNCATION_MAPPING = {
        "unknown": 0,
        "full": 0,
        "bottom": 1,
        "left": 2,
        "right": 2,
        "leftRight": 2,
        "bottomLeft": 3,
        "bottomRight": 3,
        "bottomLeftRight": 3,
    }

    def add_fields(self, example):
        """Replace fields with int32 versions."""
        example.labels["BOX"]["mapped_occlusion"] = hl_sql.create_derived_field(
            tf.int32, shape=None
        )
        example.labels["BOX"]["mapped_truncation"] = hl_sql.create_derived_field(
            tf.int32, shape=None
        )

    def filter(self, example_col_idx, dtype, row):
        """No filtering."""
        return True

    def map(self, example_col_idx, dtype, row):
        """Do the label mappings."""
        label_idx = example_col_idx.labels
        if dtype == "BOX":
            occlusion = row[label_idx["BOX"]["occlusion"]]
            occlusion = self.OCCLUSION_MAPPING.get(
                occlusion, self.OCCLUSION_MAPPING["unknown"]
            )
            row[label_idx["BOX"]["mapped_occlusion"]] = occlusion

            truncation = row[label_idx["BOX"]["truncation"]]
            truncation = self.TRUNCATION_MAPPING.get(
                truncation, self.TRUNCATION_MAPPING["unknown"]
            )
            row[label_idx["BOX"]["mapped_truncation"]] = truncation

            classifier = row[label_idx["BOX"]["classifier"]]
            classifier = classifier.replace(" ", "_")
            row[label_idx["BOX"]["classifier"]] = classifier
        return row
