# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.

"""Parser for 'BOX' type labels as provided via modulus's SqlDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import Bbox2DLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Coordinates2DWithCounts,
)
import modulus.dataloader.humanloop_sqlite_dataset as hl_sql
from modulus.types import Canvas2D


class BboxFeatureProcessor(hl_sql.FeatureProcessor):
    """Feature Processor used with humanloop_sqlite_dataset to augment BOX labels."""

    def __init__(self, min_width=1.0, min_height=1.0):
        """Constructor: min_width and min_height refer to the minimum bounding box size."""
        self._min_width = min_width
        self._min_height = min_height

    def add_fields(self, example):
        """Add auxiliary fields to the 'BOX' label."""
        example.labels["BOX"]["is_cvip"] = hl_sql.create_derived_field(
            tf.int32, shape=None
        )
        example.labels["BOX"]["non_facing"] = hl_sql.create_derived_field(
            tf.int32, shape=None
        )

    def filter(self, example_col_idx, dtype, row):
        """Filter label rows by removing bounding boxes that are too small."""
        if dtype == "BOX":
            vertices = row[example_col_idx.labels["BOX"]["vertices"]]
            if len(vertices) != 2:
                return False
            width = abs(vertices[1][0] - vertices[0][0])
            height = abs(vertices[1][1] - vertices[0][1])
            return width >= self._min_width and height >= self._min_height
        return True

    def map(self, example_col_idx, dtype, row):
        """
        Populate non_facing and is_cvip attributes for BOX label.

        Args:
            example_col_idx (namedtuple): example data structure, where fields are integers
                                          that correspond to the index of the value in 'row'
            dtype (str): label type, such as 'BOX' or 'POLYGON'.
            row (list): flat list of values from the database for one label. Use example_col_idx
                        to find which element corresponds to which field in the 'example'.

        Return:
            modified 'frame'.
        """
        label_idx = example_col_idx.labels
        if dtype == "BOX":
            attrs = row[label_idx["BOX"]["attributes"]]
            row[label_idx["BOX"]["non_facing"]] = 1 if "non facing" in attrs else 0
            row[label_idx["BOX"]["is_cvip"]] = (
                1 if ("cvip" in attrs or "cipo" in attrs) else 0
            )
        return row


class SqliteBboxLabelParser(object):
    """Parser for converting Modulus Examples into DriveNet compatible Dlav/common Examples."""

    def __init__(self):
        """Construct a parser for translating HumanLoop sqlite exports to Bbox2DLabel."""
        pass

    def __call__(self, data):
        """Convert labels to DriveNet format.

        Args:
            data (modulus.types.Example): Modulus Example namedtuple.

        Returns:
            bbox_label (Bbox2DLabel).
        """
        bbox_label_kwargs = {field_name: [] for field_name in Bbox2DLabel._fields}
        bbox_label_kwargs["frame_id"] = [data.instances["uri"]]
        width = data.instances["shape"].width
        height = data.instances["shape"].height

        box = data.labels["BOX"]
        # Parse bbox coordinates.
        bbox_label_kwargs["vertices"] = Coordinates2DWithCounts(
            coordinates=box["vertices"],
            canvas_shape=Canvas2D(height=height, width=width),
            vertices_count=box["num_vertices"],
        )
        # Parse other fields guaranteed to exist for 'BOX'.
        bbox_label_kwargs["object_class"] = box["classifier"]
        bbox_label_kwargs["back"] = box["back"]
        bbox_label_kwargs["front"] = box["front"]
        # The end-to-end behavior of the DataLoader is quite 'unfortunate', for this case.
        # The fields in the `Example.labels` produced by the SQLite are declared 'statically' by
        # some predefined fields in the HumanloopSqliteDataset, and by each FrameProcessor's
        # add_fields() method. As such, their types are also declared, and used in the
        # rest of the TensorFlow graph. Therefore, one cannot replace an existing field such as
        # 'occlusion' with a data type different than what it might be used for by dependent ops.
        # E.g. for legacy reasons, DriveNet would like to consume 'occlusion' as an int (!),
        # but the field is a string in SQLite (e.g. 'leftBottom'). If DriveNetLegacyMapper were
        # to replace the pre-existing 'occlusion' field with one of dtype tf.int32, then all values
        # will default to 0. If one were to replace it with a tf.string entry, then, other parts of
        # the graph that assumed int values would raise errors (since the part that maps the string
        # values to ints happens at 'runtime' and is not known to the TF graph).
        occlusion_key = (
            "occlusion" if "mapped_occlusion" not in box else "mapped_occlusion"
        )
        bbox_label_kwargs["occlusion"] = box[occlusion_key]
        truncation_key = (
            "truncation" if "mapped_truncation" not in box else "mapped_truncation"
        )
        bbox_label_kwargs["truncation_type"] = box[truncation_key]
        bbox_label_kwargs["is_cvip"] = box["is_cvip"]
        bbox_label_kwargs["non_facing"] = box["non_facing"]

        # Parse the optional source_weight for each frame, which may not exist.
        if "source_weight" in data.instances:
            bbox_label_kwargs["source_weight"] = [data.instances["source_weight"]]

        bbox_label = Bbox2DLabel(**bbox_label_kwargs)
        return bbox_label
