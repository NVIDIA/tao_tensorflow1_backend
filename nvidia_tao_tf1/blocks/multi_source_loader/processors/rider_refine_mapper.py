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

"""Mapper for converting 'rider' into the refined subclasses for subclassification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import modulus.dataloader.humanloop_sqlite_dataset as hl_sql


class RiderRefineMapper(hl_sql.FeatureProcessor):
    """Mapper for converting 'rider' to the refined subclasses."""

    CONCERNED_ATTRIBUTES = {"vehicle", "bicycle", "motorcycle"}
    UNKNOWN_RIDER_TYPE = "unknown_rider"
    RIDER_ATTRIBUTES_MAPPING = {
        "vehicle": "vehicle_rider",
        "bicycle": "bicycle_rider",
        "motorcycle": "motorcycle_rider",
    }

    def add_fields(self, example):
        """No fields should be added."""
        pass

    def filter(self, example_col_idx, dtype, row):
        """No filtering."""
        return True

    def map(self, example_col_idx, dtype, row):
        """Convert 'rider' to the refined subclasses."""
        label_idx = example_col_idx.labels
        if dtype == "BOX":
            # Only keep the concerned attributes for rider refine mapper.
            attrs = set(row[label_idx["BOX"]["attributes"]]) & self.CONCERNED_ATTRIBUTES

            classifier = row[label_idx["BOX"]["classifier"]]
            if classifier == "rider":
                if len(attrs) == 1:
                    # Refined rider should only have one unique concerned attribute type.
                    rider_attr = attrs.pop()
                    new_classifier = self.RIDER_ATTRIBUTES_MAPPING[rider_attr]
                else:
                    # Assign rider as UNKNOWN_RIDER_TYPE if there are no concerned attribute type
                    # or multiple conflicted attributes.
                    new_classifier = self.UNKNOWN_RIDER_TYPE
                row[label_idx["BOX"]["classifier"]] = new_classifier
        return row
