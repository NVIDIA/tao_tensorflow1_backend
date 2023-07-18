# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Objective label filter class that handles the necessary label filtering logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import Bbox2DLabel
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import (
    filter_labels,
    get_chained_filters_indices
)
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.source_class_label_filter import (
    SourceClassLabelFilter
)


class ObjectiveLabelFilter(object):
    """Holds the necessary <LabelFilter>s to apply to ground truths.

    Unlike the LabelFilter classes, which have been stripped of as much model-specific information
    as possible, this class holds such information in a 'hierarchy'.
    It is for now comprised of two levels: [target_class_name][objective_name], although in the
    future it is quite likely an additional [head_name] level will be pre-pended to it.
    """

    def __init__(self, objective_label_filter_configs, target_class_to_source_classes_mapping,
                 learnable_objective_names, mask_multiplier=1.0, preserve_ground_truth=False):
        """Constructor.

        Args:
            objective_label_filter_configs (list of ObjectiveLabelFilterConfig).
            target_class_to_source_classes_mapping (dict): maps from target class name to a list of
                source class names.
            learnable_objective_names (list of str): List of learnable objective names. These are
                the objective names a LabelFilter will be applied to if the
                ObjectiveLabelFilterConfig containing it has objective_names set to <NoneType>.
            mask_multiplier (float): Indicates the weight to be assigned to the labels resulting
                from this set of filters. Default value of 1.0 amounts to a no-op.
            preserve_ground_truth (bool): When True, the objective label filter will NOT multiply
                areas which already have nonzero coverage (the definition of a dont-care region).
                Default False implies coverage will not affect objective filtering.
        """
        self.objective_label_filter_configs = objective_label_filter_configs
        self.target_class_to_source_classes_mapping = target_class_to_source_classes_mapping
        self.learnable_objective_names = set(learnable_objective_names)
        self.mask_multiplier = mask_multiplier
        self.preserve_ground_truth = preserve_ground_truth
        # Do some sanity checks.
        for label_filter_config in self.objective_label_filter_configs:
            if label_filter_config.target_class_names is not None:
                assert set(label_filter_config.target_class_names) <= \
                    set(target_class_to_source_classes_mapping.keys()), \
                    "The filter is configured to act on at least one target class that does not " \
                    "appear in target_class_to_source_classes_mapping."
        # The following will hold the 'hierarchy' as defined in the class docstring above.
        self._label_filter_lists = self._get_label_filter_lists()

    def _get_label_filter_lists(self):
        """Set up the defined hierarchy and populates it with the necessary LabelFilters.

        Returns:
            label_filter_lists (dict): maps from [target_class_name][objective_name] to list of
                LabelFilter objects.
        """
        label_filter_lists = dict()
        # Get the "atomic" label filters.
        for config in self.objective_label_filter_configs:
            # Determine which objective(s) this particular label filter will be used for.
            objective_names = self.learnable_objective_names if config.objective_names is None \
                else config.objective_names
            # Determine which target class(es) this particular label filter will be used for.
            if config.target_class_names is None:
                # This means the filter should apply to all classes.
                target_class_names = list(
                    self.target_class_to_source_classes_mapping.keys())
            else:
                target_class_names = config.target_class_names

            # Finally, instantiate the LabelFilters.
            for target_class_name in target_class_names:
                if target_class_name not in label_filter_lists:
                    # Initialize to empty dict.
                    label_filter_lists[target_class_name] = dict()

                for objective_name in objective_names:
                    if objective_name not in label_filter_lists[target_class_name]:
                        # Initialize to empty list.
                        label_filter_lists[target_class_name][objective_name] = list(
                        )
                    # Add the appropriate LabelFilter.
                    label_filter_lists[target_class_name][objective_name].\
                        append(config.label_filter)

        return label_filter_lists

    def _apply_filters_to_labels(self, labels, label_filters,
                                 source_class_label_filter):
        """Helper method to apply filters to a single frame's labels.

        For a high-level description of some of the logic implemented here, please refer to
        doc/loss_masks.md.

        Args:
            frame_labels (dict of Tensors): Contains the labels for a single frame.
            label_filters (list): Each element is an instance of BaseLabelFilter to apply to
                <frame_labels>.
            source_class_label_filter (SourceClassLabelFilter): This will be used in conjunction
                with those filters in <label_filters> that are not of type SourceClassLabelFilter.

        Returns:
            filtered_labels (dict of Tensors): Same format as <frame_labels>, but with
                <label_filters> applied to them.
        """
        # Initialize indices to False.
        if isinstance(labels, dict):
            filtered_indices = \
                tf.zeros_like(labels['target/object_class'], dtype=tf.bool)
        elif isinstance(labels, Bbox2DLabel):
            filtered_indices = \
                tf.zeros_like(labels.object_class.values, dtype=tf.bool)
        else:
            raise ValueError("Unsupported type.")

        # First, get the filters in filter_list that are also SourceClassLabelFilter.
        source_class_label_filters = \
            [l for l in label_filters if isinstance(l, SourceClassLabelFilter)]
        other_label_filters = \
            [l for l in label_filters if not isinstance(
                l, SourceClassLabelFilter)]

        # Find those labels mapped to target_class_name, and satisfying any of the
        # other_filters. The assumption here is that, if a user specifies a filter that is not of
        # type SourceClassLabelFilter, then implicitly they would like it to be applied to only
        # those source classes mapped to a given target class. e.g. If one would specify that
        # targets whose bbox dimensions were in a given range should be selected for the target
        # class 'car', then only those objects that are actually (mapped to) 'car' will have this
        # filter applied on them, hence the logical-and.
        if len(other_label_filters) > 0:
            filtered_indices = \
                tf.logical_and(get_chained_filters_indices(other_label_filters, labels, 'or'),
                               source_class_label_filter.is_criterion_satisfied(labels))

        # Apply the user-specified source class label filters, if necessary. Here, the indices
        # satisfying any said source class label filter will be logical-or-ed with the result
        # of the previous step. We do not want to logical-and the user-specified source class label
        # filters with the one that maps to a given target class, because the assumption is that
        # if the user specifies such filters, it is that they only want those.
        # Note that the source classes for a source class label filter need not be present in the
        # mapping for a given target class for this to work.
        if len(source_class_label_filters) > 0:
            source_class_filtered_indices = \
                get_chained_filters_indices(
                    source_class_label_filters, labels, 'or')
            filtered_indices = \
                tf.logical_or(filtered_indices, source_class_filtered_indices)

        filtered_labels = filter_labels(labels, filtered_indices)

        return filtered_labels

    def apply_filters(self, batch_labels):
        """Method that users will call to actually do the filtering.

        Args:
            batch_labels (list of dict of Tensors): contains the labels for a batch of frames.
                Each element in the list corresponds to a single frame's labels, and is a dict
                containing various label features.

        Returns:
            filtered_labels_dict (nested dict): for now, has two levels:
                [target_class_name][objective_name]. The leaf values are the corresponding filtered
                ground truth labels in tf.Tensor form for a batch of frames.
        """
        filtered_labels_dict = dict()
        for target_class_name, target_class_filters in six.iteritems(self._label_filter_lists):
            filtered_labels_dict[target_class_name] = dict()
            # Get a filter that will filter labels whose source class names are mapped to
            # this target_class_name.
            source_class_names = self.target_class_to_source_classes_mapping[target_class_name]
            source_class_label_filter = \
                SourceClassLabelFilter(source_class_names=source_class_names)
            for objective_name, filter_list in six.iteritems(target_class_filters):
                # Initialize the list of filtered labels for this combination of
                # [target_class_name][objective_name]. Each element will correspond to one frame's
                # labels.
                if isinstance(batch_labels, list):
                    filtered_labels = []
                    for frame_labels in batch_labels:
                        filtered_labels.append(self._apply_filters_to_labels(
                            frame_labels, filter_list, source_class_label_filter))
                elif isinstance(batch_labels, Bbox2DLabel):
                    filtered_labels = \
                        self._apply_filters_to_labels(batch_labels,
                                                      filter_list,
                                                      source_class_label_filter)
                else:
                    raise ValueError("Unsupported type.")

                filtered_labels_dict[target_class_name][objective_name] = filtered_labels

        return filtered_labels_dict
