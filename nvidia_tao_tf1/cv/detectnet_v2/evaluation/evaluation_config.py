# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""EvaluationConfig class that holds evaluation parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _build_evaluation_box_config(evaluation_box_config_proto, target_classes):
    """Build EvaluationBoxConfig from proto.

    Args:
        evaluation_box_config_proto: evaluation_config.evaluation_box_config message.
    Returns:
        A dict of EvalutionBoxConfig instances indexed by target class names.
    """
    evaluation_box_configs = {}
    for key in target_classes:
        # Check if key is present in the evaluation config.
        if key not in evaluation_box_config_proto.keys():
            raise ValueError("Evaluation box config is missing for {}".format(key))
        config = evaluation_box_config_proto[key]
        evaluation_box_configs[key] = EvaluationBoxConfig(config.minimum_height,
                                                          config.maximum_height,
                                                          config.minimum_width,
                                                          config.maximum_width)
    return evaluation_box_configs


def build_evaluation_config(evaluation_config_proto, target_classes):
    """Build EvaluationConfig from proto.

    Args:
        evaluation_config_proto: evaluation_config message.
    Returns:
        EvaluationConfig object.
    """
    # Get validation_period_during_training.
    validation_period_during_training = evaluation_config_proto.validation_period_during_training

    # Get first_validation_epoch.
    first_validation_epoch = evaluation_config_proto.first_validation_epoch

    # Create minimum_detection_ground_truth_overlap dict from evaluation_config_proto.
    minimum_detection_ground_truth_overlaps = {}
    for key in target_classes:
        if key not in evaluation_config_proto.minimum_detection_ground_truth_overlap.keys():
            raise ValueError("Cannot find a min overlap threshold for {}".format(key))
        minimum_detection_ground_truth_overlaps[key] = evaluation_config_proto.\
            minimum_detection_ground_truth_overlap[key]

    # Build EvaluationBoxConfig from evaluation_config_proto.
    evaluation_box_configs = \
        _build_evaluation_box_config(
            evaluation_config_proto.evaluation_box_config,
            target_classes)

    average_precision_mode = evaluation_config_proto.average_precision_mode
    # Build EvaluationConfig object.
    evaluation_config = EvaluationConfig(validation_period_during_training,
                                         first_validation_epoch,
                                         minimum_detection_ground_truth_overlaps,
                                         evaluation_box_configs,
                                         average_precision_mode)
    return evaluation_config


class EvaluationBoxConfig(object):
    """Holds parameters for EvaluationBoxConfig."""

    def __init__(self, minimum_height, maximum_height, minimum_width,
                 maximum_width):
        """Constructor.

        Evaluation boc configs are used to filter detections based on object height, width.

        Args:
            minimum_height (int): Ground truths with height below this value are ignored.
            maximum_height (int): Ground truths with height above this value are ignored.
            minimum_width (int): Ground truths with width below this value are ignored.
            maximum_width (int): Ground truths with width above this value are ignored.
        """
        self.minimum_height = minimum_height
        self.maximum_height = maximum_height
        self.minimum_width = minimum_width
        self.maximum_width = maximum_width


class EvaluationConfig(object):
    """Holds parameters for EvaluationConfig."""

    def __init__(self, validation_period_during_training, first_validation_epoch,
                 minimum_detection_ground_truth_overlap, evaluation_box_configs,
                 average_precision_mode):
        """Constructor.

        EvaluationConfig is a class definition for specifying parameters to
        evaluate DriveNet detections against ground truth labels.

        Allows the user to specify:
            - Minimum overlap between a detection and ground truth to count as a true positive.
            - Evaluation boxs that each specify e.g. minimum and maximum object height.
            - Weights for computing weighted metrics (e.g. weighted AP).

        Args:
            validation_period_during_training (int): The frequency for model validation during
                training (in epochs).
            first_validation_epoch (int): The first validation epoch. After this, validation is done
                on epochs first_validation_epoch + i*validation_period_during_training.
            minimum_detection_ground_truth_overlap (dict): Minimum overlap of a ground truth and a
                detection bbox to consider the detection to be a true positive.
            evaluation_box_config (dict): dict in which keys are class names
                values are EvaluationBoxConfig objects containing
                parameters such as minimum and maximum bbox height.
        """
        self.validation_period_during_training = validation_period_during_training
        self.first_validation_epoch = first_validation_epoch
        self.minimum_detection_ground_truth_overlap = minimum_detection_ground_truth_overlap
        self.evaluation_box_configs = evaluation_box_configs
        self.average_precision_mode = average_precision_mode
