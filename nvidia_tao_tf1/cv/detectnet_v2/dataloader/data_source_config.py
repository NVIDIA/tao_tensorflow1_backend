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

"""Data source config class for DriveNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DataSourceConfig(object):
    """Hold all data source related parameters."""

    def __init__(self,
                 dataset_type,
                 dataset_files,
                 images_path,
                 export_format,
                 split_db_path,
                 split_tags,
                 source_weight=1.0,
                 minimum_target_class_imbalance=None,
                 num_duplicates=0,
                 skip_empty_frames=False,
                 ignored_classifiers_for_skip=None,
                 additional_conditions=None):
        """Constructor.

        Args:
            dataset_type (string): Currently only 'tfrecord' and 'sqlite' are supported.
            dataset_files (list): A list of absolute paths to dataset files. In case of
                tfrecords, a list of absolute paths to .tfrecord files.
            images_path (string): Absolute path to images directory.
            export_format (string): (SQL only) Image format name.
            split_db_path (string): (SQL only) Path to split database.
            split_tags (list of strings): (SQL only) A list of split tags (eg. ['train'] or
                ['val0', 'val1']).
            source_weight (float): Value by which to weight the loss for samples
                coming from this DataSource.
            minimum_target_class_imbalance (map<string, float>): Minimum ratio
                (#dominant_class_instances/#target_class_instances) criteria for duplication
                of frames. The string is the non-dominant class name and the float is the
                ratio for duplication.
            num_duplicates (int): Number of duplicates of frames to be added, if the frame
                satifies the minimum_target_class_imbalance.
            skip_empty_frames (bool): Whether to ignore empty frames (i.e frames without relevant
                features. By default, False, i.e all frames are returned.
            ignored_classifiers_for_skip (set): Names of classifiers to ignore when
                considering if frame is empty. I.e if frame only has these classes, it is still
                regarded as empty.
            additional_conditions (list): List of additional sql conditions for a 'where' clause.
                It's only for SqliteDataSource, and other data sources will ignore it.
        """
        self.dataset_type = dataset_type
        self.dataset_files = dataset_files
        self.images_path = images_path
        self.export_format = export_format
        self.split_db_path = split_db_path
        self.split_tags = split_tags
        if source_weight < 0.0:
            raise ValueError("source_weight cannot be negative value")
        elif source_weight == 0.0:
            # Assume it was meant to be 1.0.
            self.source_weight = 1.0
        else:
            self.source_weight = source_weight
        self.minimum_target_class_imbalance = minimum_target_class_imbalance
        self.num_duplicates = num_duplicates
        self.skip_empty_frames = skip_empty_frames
        self.ignored_classifiers_for_skip = ignored_classifiers_for_skip
        self.additional_conditions = additional_conditions
