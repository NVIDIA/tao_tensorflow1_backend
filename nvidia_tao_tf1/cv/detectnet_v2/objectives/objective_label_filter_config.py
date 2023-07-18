# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Define a lightweight class for configuring ObjectiveLabelFilter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ObjectiveLabelFilterConfig(object):
    """Lightweight class with the information necessary to instantiate a ObjectiveLabelFilter."""

    def __init__(self,
                 label_filter,
                 objective_names=None,
                 target_class_names=None):
        """Constructor.

        Args:
            label_filter: LabelFilter instance.
            objective_names (list of str): List of objective names to which this label filter config
                should apply. If None, indicates the config should be for all objectives.
            target_class_names (list of str): List of target class names to which this label filter
                config should apply. If None, indicates the config should be for all target classes.
        """
        self.label_filter = label_filter
        self.objective_names = set(
            objective_names) if objective_names is not None else None
        self.target_class_names = \
            set(target_class_names) if target_class_names is not None else None
