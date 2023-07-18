# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

"""Processor to convert 'BOX' type labels to 'POLYGON' type labels."""

import numpy as np

from nvidia_tao_tf1.core.dataloader.dataset import FeatureProcessor


class BboxToPolygon(FeatureProcessor):
    """Feature processor to convert 'BOX' type labels to 'POLYGON' labels.

    It also splits labels into several different labels based on the size of the object (needed
    for AhbNet).
    """

    def __init__(self, size_buckets=None, labels_to_split=None, image_coordinates=None):
        """Constructor for the bbox to polygon label converter.

        It also splits labels into several different labels based on the size of the object,
        if labels_to_split is specified.

        Args:
            size_buckets (dict): Dictionary with the buckets (min, max) for each object size.
            For example: {"tiny": {"min": 0, "max": 10},
                          "small": {"min": 10, "max": 20},
                          "median: {"min": 20, "max": 30},
                          "large": {"min": 30},}
            labels_to_split (list[str]): List with the class keys that need to be split
                based on the size of the object.
            image_coordinates (dict): Dictionary with the x_min, y_min, x_max and y_max of
                the image. If the min values are not specified, they are default to 0.0.
        """
        self._size_buckets = size_buckets if size_buckets else {}
        self._labels_to_split = labels_to_split
        if labels_to_split and not image_coordinates:
            raise ValueError(
                "When labels_to_split is specified, the image coordinates have to be specified too."
            )
        if image_coordinates:
            if not ("x_max" in image_coordinates and "y_max" in image_coordinates):
                raise ValueError(
                    "The x_max and y_max of the image need to be specified."
                )
            if "x_min" not in image_coordinates:
                image_coordinates["x_min"] = 0.0
            if "y_min" not in image_coordinates:
                image_coordinates["y_min"] = 0.0
        self._image_coordinates = image_coordinates

    def add_fields(self, example):
        """Add new fields to the example data structure (labels).

        Args:
            example (namedtuple): Data structure that the loader returns.
        """
        pass

    def filter(self, example_col_idx, dtype, feat_row):
        """Filter label rows.

        Args:
            example_col_idx (namedtuple): Example data structure, where fields are integers
                                          that correspond to the index of the value in 'row'
            dtype (str): Label type, such as 'BOX' or 'POLYGON'.
            row (list): Flat list of values from the database for one label. Use example_col_idx
                        to find which element corresponds to which field in the 'example'.

        Returns:
            True or False, depending on wheter the row should be kept.
        """
        return True

    def map(self, example_col_idx, dtype, feat_row):
        """Modify or inject values into the feature row.

        Args:
            example_col_idx (namedtuple): Example data structure, where fields are integers that
               correspond to the index of the value in 'row'.
            dtype (str): Label type, such as 'POLYGON'.
            feat_row (list): Flat list of values from the database for one label.
                Use example_col_idx.

        Returns: modified 'row'.
        """
        if dtype == "POLYGON":
            label = example_col_idx.labels[dtype]
            verts = feat_row[label["vertices"]]
            if len(verts) > 0:
                if self._labels_to_split:
                    all_x = [v[0] for v in verts]
                    all_y = [v[1] for v in verts]
                    image_x_min = self._image_coordinates["x_min"]
                    image_y_min = self._image_coordinates["y_min"]
                    image_x_max = self._image_coordinates["x_max"]
                    image_y_max = self._image_coordinates["y_max"]
                    all_x = np.clip(all_x, a_min=image_x_min, a_max=image_x_max)
                    all_y = np.clip(all_y, a_min=image_y_min, a_max=image_y_max)
                    if len(all_x) > 0 and len(all_y) > 0:
                        max_x = np.max(all_x)
                        min_x = np.min(all_x)
                        diff_x = np.abs(max_x - min_x)
                        max_y = np.max(all_y)
                        min_y = np.min(all_y)
                        diff_y = np.abs(max_y - min_y)

                        if feat_row[label["classifier"]] in self._labels_to_split:
                            # The cars larger than any size bucket will be tagged as
                            # feat_row[label["classifier"]]
                            for size_bucket in self._size_buckets:
                                max_value = self._size_buckets[size_bucket]["max"]
                                min_value = self._size_buckets[size_bucket]["min"]
                                if (min_value < diff_y <= max_value) or (
                                    min_value < diff_x <= max_value
                                ):
                                    size_bucket = "_" + size_bucket
                                    feat_row[label["classifier"]] += size_bucket
                                    break
            if len(verts) == 2:  # BBOX then we modify
                x0, y0 = verts[0][0], verts[0][1]
                x1, y1 = verts[1][0], verts[1][1]
                new_verts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                feat_row[label["vertices"]] = new_verts
                feat_row[label["num_vertices"]] = 4
        return feat_row
