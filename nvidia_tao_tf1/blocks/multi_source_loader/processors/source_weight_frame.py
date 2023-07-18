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

"""Source weight frame processors for assigning source weight for each frame.

By assigning different source weight values to frames from different data source.
We managed to treat data sources differently according to our need. Finally those
source weight values will be used in the loss function calculation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import modulus.dataloader.humanloop_sqlite_dataset as hl_sql
from nvidia_tao_tf1.core.coreobject import save_args


class SourceWeightSQLFrameProcessor(hl_sql.FrameProcessor):
    """Adds additional field 'source_weight' to each frame from specific datasource."""

    @save_args
    def __init__(self, source_weight=1.0):
        """
        Init methods.

        Args:
            source_weight (float): Value by which to weight the loss for samples
                coming from this DataSource.
        """
        self.source_weight = source_weight

    def add_fields(self, example):
        """
        Add new fields to the example data structure (labels).

        Example:
            example.labels['BOX']['testfield_int'] = create_derived_field(tf.int32, shape=None)

        Args:
            example (namedtuple): data structure that the loader returns.
        """
        example.instances["source_weight"] = hl_sql.create_derived_field(
            tf.float32, shape=None
        )

    def map(self, example_col_idx, frame):
        """
        Modify or inject values into the frame.

        Args:
            example_col_idx (namedtuple): example data structure, where fields are integers
                                          that correspond to the index of the value in 'row'
            dtype (str): label type, such as 'BOX' or 'POLYGON'.
            frame (list): flat list of values from the database for a frame. Use example_col_idx
                        to find which element corresponds to which field in the 'example'.

        Return:
            modified 'frame'.
        """
        if "source_weight" in example_col_idx.instances:
            col_idx = example_col_idx.instances["source_weight"]
            frame[col_idx] = self.source_weight
        return frame
