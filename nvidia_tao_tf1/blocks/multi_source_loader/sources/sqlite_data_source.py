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
"""Adapter for sqlite based HumanLoop datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sqlite3

import six
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.processors.source_weight_frame import (
    SourceWeightSQLFrameProcessor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.data_source import (
    DataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.sqlite_bbox_label_parser import (
    BboxFeatureProcessor,
    SqliteBboxLabelParser,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Coordinates2DWithCounts,
    FEATURE_CAMERA,
    FEATURE_SESSION,
    Images2DReference,
    LABEL_MAP,
    LABEL_OBJECT,
    Polygon2DLabel,
    SequenceExample,
    Session,
)
import modulus.dataloader.humanloop as hl
import modulus.dataloader.humanloop_sqlite_dataset as hl_sql
from nvidia_tao_tf1.core.coreobject import save_args
from modulus.types import Canvas2D

logger = logging.getLogger(__name__)


class SqliteDataSource(DataSource):
    """SqliteDataSource for ingesting sqlite files conforming to the HumanLoop schema."""

    FORMAT_RGB_HALF_DWSOFTISP = "rgb_half_dwsoftisp_v0.52b"
    FORMAT_JPEG = "jpeg"
    DEFAULT_SUPPORTED_LABELS = ["POLYGON"]

    @save_args
    def __init__(
        self,
        sqlite_path,
        image_dir,
        export_format,
        split_db_filename=None,
        split_tags=None,
        supported_labels=None,
        subset_size=None,
        skip_empty_frames=None,
        ignored_classifiers_for_skip=None,
        feature_processors=None,
        oversampling_strategy=None,
        frame_processors=None,
        include_unlabeled_frames=None,
        additional_conditions=None,
        **kwargs,
    ):
        """
        Construct a SqliteDataSource.

        Args:
            sqlite_path (str): Path to sqlite file.
            image_dir (str): Path to directory where images referenced by examples are stored.
            export_format (str): Folder from where to load the images depending on their format.
            split_db_filename (str): Optional split database defining training/validation sets.
            split_tags (list of str): A list of split tags to choose a subset of sets (eg.
                ['val0', 'val1'] to iterate over two validation folds).
            supported_labels (list<str>): List of supported labels to be loaded by this source
            subset_size (int): Number of frames to use. If None, use all.
            skip_empty_frames (bool). Whether to ignore empty frames (i.e frames withou relevant
                features. By default, False, i.e all frames are returned.
            ignored_classifiers_for_skip (set or list or None): Names of classifiers to ignore
                when considering if frame is empty. I.e if frame only has these classes,
                it is still regarded as empty.
            oversampling_strategy (hl_sql.OverSamplingStrategy): OverSamplingStrategy child class
                by which to duplicate certain frames.
            frame_processors (list): list of func hook, which will be triggered in frame parsing.
            include_unlabeled_frames (bool): Whether to include unlabeled frames. Default False.
            additional_conditions (list): list of additional sql conditions for a 'where' clause.
                Each element in the list can only reference one of the following tables: 'features',
                'frames', 'sequences'.
        """
        super(SqliteDataSource, self).__init__(**kwargs)
        if not os.path.isfile(sqlite_path):
            raise ValueError(
                "No dataset sqlite file found at path: '{}'".format(sqlite_path)
            )
        if not os.path.isdir(image_dir):
            raise ValueError(
                "No dataset image directory found at path: '{}'".format(image_dir)
            )
        if split_db_filename is not None and not os.path.isfile(split_db_filename):
            raise ValueError(
                "No dataset split file found at path: '{}'".format(split_db_filename)
            )
        if subset_size is not None and subset_size < 0:
            raise ValueError("subset_size can not be negative")

        if additional_conditions is not None:
            self._check_additional_conditions(additional_conditions)
        self.sqlite_path = sqlite_path
        self.image_dir = image_dir
        self.export_format = export_format
        self.split_db_filename = split_db_filename
        if split_tags is not None:
            split_tags = [
                t.decode() if isinstance(t, six.binary_type) else t for t in split_tags
            ]
        if split_db_filename and split_tags:
            split_db_tags = set(self.query_split_tags(split_db_filename))
            missing_tags = set(split_tags) - split_db_tags
            if missing_tags:
                raise ValueError(
                    "Split tags {} not in split db '{}' with tags {}.".format(
                        list(missing_tags), split_db_filename, list(split_db_tags)
                    )
                )

        self.split_tags = split_tags
        self._include_unlabeled_frames = include_unlabeled_frames or False
        self.supported_labels = supported_labels or self.DEFAULT_SUPPORTED_LABELS
        self._num_shards = 1
        self._shard_id = 0
        self._pseudo_sharding = False
        self._sequence_length = None
        self._shuffle = False
        self._shuffle_buffer_size = 10000
        self._dataset = None
        self._subset_size = subset_size
        self._skip_empty_frames = skip_empty_frames or False
        self._ignored_classifiers_for_skip = set()
        if ignored_classifiers_for_skip is not None:
            self._ignored_classifiers_for_skip.update(ignored_classifiers_for_skip)
        self._feature_processors = feature_processors
        self._oversampling_strategy = oversampling_strategy
        self._frame_processors = frame_processors
        self.num_samples = None
        self._additional_conditions = additional_conditions
        self.set_image_properties(*self.get_image_properties())

    @staticmethod
    def query_split_tags(split_db_filename):
        """Query Sqlite split db for a list of split tags.

        Args:
            split_db_filename (str): Path to split db.

        Returns:
            A list of split tag strings.
        """
        connection = sqlite3.connect(split_db_filename)
        tags_query = "SELECT DISTINCT tag FROM split"
        tags = connection.cursor().execute(tags_query).fetchall()
        return [tag[0] for tag in tags]

    def get_image_properties(self):
        """Returns the maximum width and height of images for this source."""
        split_join = ""
        initial_statement = ""
        if self.split_db_filename:
            initial_statement = "ATTACH DATABASE '{filename}' AS {db_name}".format(
                filename=self.split_db_filename, db_name=hl_sql.SPLIT_DB_NAME
            )

            tags = ", ".join(["'{}'".format(tag) for tag in self.split_tags])
            split_join = (
                "\tJOIN {db_name}.{table} ON {db_name}.{table}.id = sequences.session_uuid\n"
                "\t\tAND {db_name}.{table}.tag IN ({tags})\n"
            )
            split_join = split_join.format(
                db_name=hl_sql.SPLIT_DB_NAME, table="split", tags=tags
            )

        additional_sequence_conditions = ""
        if self._additional_conditions is not None:
            conditions = list(
                condition
                for condition in self._additional_conditions
                if "sequences" in condition
            )
            if len(conditions) > 0:
                additional_sequence_conditions = " WHERE " + " AND ".join(conditions)

        connection = sqlite3.connect(self.sqlite_path)
        cursor = connection.cursor()
        cursor.execute(initial_statement)

        max_image_size_query = (
            "SELECT MAX(width), MAX(height) FROM sequences"
            + split_join
            + additional_sequence_conditions
        )
        max_image_width, max_image_height = cursor.execute(
            max_image_size_query
        ).fetchone()
        if max_image_height is None or max_image_width is None:
            max_image_height = 0
            max_image_width = 0

        export_params_query = (
            "SELECT region_width, region_height, width, height, extension FROM"
            " formats WHERE name = ?"
        )
        row = cursor.execute(export_params_query, (self.export_format,)).fetchall()
        if not row:
            raise ValueError(
                "export_format {} not found in database {}.".format(
                    self.export_format, self.sqlite_path
                )
            )
        region_width, region_height, width_factor, height_factor, extension = row[0]

        region_width = region_width or 1.0
        region_height = region_height or 1.0
        width_factor = width_factor or 1.0
        height_factor = height_factor or 1.0

        max_image_width = int(round(max_image_width * region_width * width_factor))
        max_image_height = int(round(max_image_height * region_height * height_factor))

        self.extension = extension

        return max_image_width, max_image_height

    def set_image_properties(self, max_image_width, max_image_height):
        """Overrides the maximum image width and height of this source."""
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height

    def set_shard(self, num_shards, shard_id, pseudo_sharding=False):
        """
        Sets the sharding configuration of this source.

        Args:
           num_shards (int):  The number of shards.
           shard_id (int):    Shard id from 0 to num_shards - 1.
           pseudo_sharding (bool) if True, then data is not actually sharded, but different shuffle
                              seeds are used to differentiate shard batches.
        """
        self._num_shards = num_shards
        self._shard_id = shard_id
        self._pseudo_sharding = pseudo_sharding

    def supports_sharding(self):
        """Whether this source can do its own sharding."""
        return True

    def supports_shuffling(self):
        """Whether this source can do its own shuffling."""
        return True

    def set_shuffle(self, buffer_size):
        """Enables shuffling on this data source."""
        self._shuffle = True
        self._shuffle_buffer_size = buffer_size

    def set_sequence_length(self, sequence_length):
        """Sets the sequence length (number of frames in sequence)."""
        self._sequence_length = sequence_length

    def supports_temporal_batching(self):
        """Whether this source does its own temporal batching."""
        return True

    def initialize(self):
        """Called by data loaders after all configuration is done."""
        example = hl.example_template(label_names=self.supported_labels)

        # Prune fields we don't need. This has significant impact on
        # the tensorflow overhead of processing data.
        example.instances["sequence"]["dataset_name"].prune = True
        example.instances["sequence"]["id"].prune = True
        example.instances["sequence"]["recording_date"].prune = True
        example.instances["source"]["camera_lens"].prune = True
        example.instances["source"]["camera_location"].prune = True
        example.instances["source"]["width"].prune = True
        example.instances["source"]["height"].prune = True

        if self._skip_empty_frames:
            self._ignored_classifiers_for_skip.add("do_not_care")

        # It would be meaningless to specify classifiers for skipping but have skipping disabled,
        # so it would be likely a user error.
        if self._ignored_classifiers_for_skip and not self._skip_empty_frames:
            raise ValueError(
                "Cant specify ignored_classifiers_for_skip w/ skip_empty_frames=False"
            )

        feature_processors = []
        if "BOX" in self.supported_labels and (
            self._feature_processors is None
            or not any(
                isinstance(f, BboxFeatureProcessor) for f in self._feature_processors
            )
        ):
            feature_processors.append(BboxFeatureProcessor())

        if self._feature_processors:
            feature_processors.extend(self._feature_processors)

        frame_processors = []
        if "BOX" in self.supported_labels and (
            self._frame_processors is None
            or not any(
                isinstance(f, SourceWeightSQLFrameProcessor)
                for f in self._frame_processors
            )
        ):
            frame_processors.append(SourceWeightSQLFrameProcessor())

        if self._frame_processors:
            frame_processors.extend(self._frame_processors)

        self._dataset = hl_sql.HumanloopSqliteDataset(
            filename=self.sqlite_path,
            export_format=self.export_format,
            split_db_filename=self.split_db_filename,
            split_tags=self.split_tags,
            example=example,
            order_by=hl_sql.ORDER_BY_SESSION_CAMERA_FRAME,
            export_path=self.image_dir,
            skip_empty_frames=self._skip_empty_frames,
            feature_processors=feature_processors,
            frame_processors=frame_processors,
            subset_size=self._subset_size,
            ignored_classifiers_for_skip=self._ignored_classifiers_for_skip,
            oversampling_strategy=self._oversampling_strategy,
            include_unlabeled_frames=self._include_unlabeled_frames,
            additional_conditions=self._additional_conditions,
        )

        self._dataset.set_sequence_length(self._sequence_length)
        self._dataset.set_shard(
            self._num_shards, self._shard_id, pseudo_sharding=self._pseudo_sharding
        )
        if self._shuffle:
            self._dataset.set_shuffle(buffer_size=self._shuffle_buffer_size)
        self.num_samples = self._dataset.num_unsharded_sequences()

    def call(self):
        """Return a tf.data.Dataset for this data source."""

        # Note: currently batch is sliced to frames and batching is done externally.
        # TODO (vkallioniemi): do batching here.
        # NOTE: 512 is used to control how many frames the loader loads at once. Should be
        # replaced by the actual batch size when batching is moved to the source. JIRA ML-1553
        if not self._dataset:
            self.initialize()
        self._dataset.set_batch(512, slice_batch=True)

        return self._dataset().map(
            _ExampleAdapter(
                supported_labels=self.supported_labels,
                max_image_width=self.max_image_width,
                max_image_height=self.max_image_height,
                sequence_length=self._sequence_length,
                include_unlabeled_frames=self._include_unlabeled_frames,
            )
        )

    def __len__(self):
        """Return the number of examples in the underlying sql query."""
        assert self._dataset, "Need to call initialize before calling len()"
        return self.num_samples

    @staticmethod
    def _check_additional_conditions(additional_conditions):
        # Conditions are limited to the tables joined in the feature query in
        # modulus.dataloader.humanloop_sqlite_dataset.HumanloopSqliteDataset
        # see moduluspy/modulus/dataloader/humanloop_sqlite_dataset.py#306
        #
        # Limit to a single table per condition is necessary because additional conditions need to
        # be used in get_image_properties so we need to be able to extract out conditions on
        # sequences.
        tables = ["features", "frames", "sequences"]
        for condition in additional_conditions:
            if list(table in condition for table in tables).count(True) != 1:
                raise (ValueError("Each condition can only reference one table."))


# TODO(vkallioniemi): Restructure this class to be either to be part of the SqliteDataSource or
# move to a separate file.
class _ExampleAdapter(object):
    """Extract examples from HumanLoop datasets."""

    def __init__(
        self,
        supported_labels,
        max_image_width,
        max_image_height,
        sequence_length,
        include_unlabeled_frames,
    ):
        """
        Constructs an _ExampleAdapter.

        Args:
            supported_labels (list<str>): List of supported labels
            max_image_width (int): Maximum image width to which (smaller) images will be padded to.
            max_image_width (int): Maximum image height to which (smaller) images will be padded to.
            include_unlabeled_frames(bool) Whether or not unlabeled frames will be included and need
                to be tracked.
        """
        self.supported_labels = supported_labels
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self._sequence_length = sequence_length
        self._bbox_label_parser = None
        if "BOX" in self.supported_labels:
            self._bbox_label_parser = SqliteBboxLabelParser()

    def __call__(self, data):
        """
        Transforms an sqlite specific Example to a normalized/canonical form.

        Args:
            data (Example): Example structure with instances and labels matching the schema used
                for the HumanLoop dataloader.

        Returns:
            (SequenceExample): Example structure with instances and labels matching the
                       normalized/canonical schema used by all DataSources.
        """
        return SequenceExample(
            instances={
                FEATURE_CAMERA: self._extract_frame_paths(data.instances),
                FEATURE_SESSION: self._extract_session(data.instances),
            },
            labels=self._extract_labels(data),
        )

    def _extract_session(self, instances):
        return Session(
            uuid=instances["sequence"]["session_uuid"],
            camera_name=instances["source"]["camera_name"],
            frame_number=instances["number"],
        )

    def _extract_frame_paths(self, instances):
        temporal_dim = [self._sequence_length] if self._sequence_length else []
        canvas_shape = Canvas2D(
            height=tf.ones(temporal_dim + [self.max_image_height]),
            width=tf.ones(temporal_dim + [self.max_image_width]),
        )
        input_height = instances["shape"].height
        input_width = instances["shape"].width

        return Images2DReference(
            path=instances["path"],
            extension=instances["export"]["extension"],
            canvas_shape=canvas_shape,
            input_height=input_height,
            input_width=input_width,
        )

    def _extract_map_labels(self, labels, supported_labels):
        """Extract map labels."""
        all_vertices = []
        all_classes = []
        all_attributes = []

        for label_name in supported_labels:
            label = labels[label_name]

            vertices = label["vertices"]
            classes = label["classifier"]
            attributes = label["attributes"]
            all_vertices.append(vertices)
            all_classes.append(classes)
            all_attributes.append(attributes)

        # TODO(weich) Tensosrflow 2.x will deprecate expand_nonconcat_dim, but Tensorflow 1.14
        # supports both expand_nonconcat_dims and expand_nonconcat_dim, and defaulting
        # expand_nonconcat_dim to be false.
        # So if we set  expand_nonconcat_dims=True, tensorflow will complain.
        # We will switch expand_nonconcat_dim to expand_nonconcat_dims upon upgrading to
        # Tensorflow 2.x
        if len(all_vertices) > 1:
            axis = 1 if self._sequence_length else 0
            vertices = tf.sparse.concat(
                axis=axis,
                sp_inputs=all_vertices,
                name="sqlite_cat_vertices",
                expand_nonconcat_dim=True,
            )
            classes = tf.sparse.concat(
                axis=axis,
                sp_inputs=all_classes,
                name="sqlite_cat_classes",
                expand_nonconcat_dim=True,
            )
            attributes = tf.sparse.concat(
                axis=axis,
                sp_inputs=all_attributes,
                name="sqlite_cat_attributes",
                expand_nonconcat_dim=True,
            )

        # Expand classes to be of the same rank as attributes. This allows for multiple classes
        # per object.
        classes = tf.sparse.expand_dims(classes, axis=-1)

        return Polygon2DLabel(
            vertices=Coordinates2DWithCounts(
                coordinates=vertices,
                canvas_shape=Canvas2D(
                    height=tf.constant(self.max_image_height, tf.int32),
                    width=tf.constant(self.max_image_width, tf.int32),
                ),
                vertices_count=label["num_vertices"],
            ),
            classes=classes,
            attributes=attributes,
        )

    def _extract_bbox_labels(self, labels):
        return self._bbox_label_parser(labels)

    def _extract_labels(self, example):
        labels = example.labels
        output_labels = dict()

        expected_labels = set(self.supported_labels)

        if "BOX" in expected_labels:
            output_labels[LABEL_OBJECT] = self._extract_bbox_labels(example)
            expected_labels.remove("BOX")

        map_labels = {"POLYGON", "POLYLINE"}.intersection(expected_labels)
        if map_labels:
            output_labels[LABEL_MAP] = self._extract_map_labels(labels, map_labels)
            expected_labels = expected_labels.difference(map_labels)

        # TODO(@williamz): Add unit tests that would have caught polyline support getting dropped.
        if expected_labels:
            raise ValueError(
                "This datasource was configured to support labels that there are "
                "currently no handlers for. Invalid label types: {}.".format(
                    expected_labels
                )
            )

        return output_labels
