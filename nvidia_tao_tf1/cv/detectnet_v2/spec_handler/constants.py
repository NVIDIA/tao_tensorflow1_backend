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

"""File containing constants for the spec handling."""

from nvidia_tao_tf1.cv.common.spec_validator import ValueChecker

TRAINVAL_OPTIONAL_CHECK_DICT = {
    "max_objective_weight": ValueChecker(">=", 0.0),
    "min_objective_weight": ValueChecker(">=", 0.0),
    "checkpoint_interval": ValueChecker(">", 0.0),
    "num_images": ValueChecker(">", 0),
    "scales": ValueChecker("!=", ""),
    "steps": ValueChecker("!=", ""),
    "offsets": ValueChecker("!=", "")
}

TRAINVAL_VALUE_CHECK_DICT = {
    # model config parameters.
    "arch": [ValueChecker("!=", ""),
             ValueChecker("in", ["resnet",
                                 "vgg",
                                 "darknet",
                                 "mobilenet_v1",
                                 "mobilenet_v2",
                                 "squeezenet",
                                 "googlenet",
                                 "efficientnet_b0"])],
    "num_layers": [ValueChecker(">=", 0)],
    "scale": [ValueChecker(">", 0)],
    "offset": [ValueChecker(">=", 0)],
    # bbox rasterizer parameters.
    "cov_center_x": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "cov_center_y": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "cov_radius_x": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "cov_radius_y": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "bbox_min_radius": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "deadzone_radius": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    # Training config.
    "batch_size_per_gpu": [ValueChecker(">", 0)],
    "num_epochs": [ValueChecker(">", 0)],
    "min_learning_rate": [ValueChecker(">", 0)],
    "max_learning_rate": [ValueChecker(">", 0)],
    "soft_start": [ValueChecker(">=", 0), ValueChecker("<", 1.0)],
    "annealing": [ValueChecker(">=", 0), ValueChecker("<", 1.0)],
    # evaluation config parameters.
    "validation_period_during_training": [ValueChecker(">", 0)],
    "first_validation_epoch": [ValueChecker(">", 0)],
    "minimum_height": [ValueChecker(">=", 0)],
    "minimum_width": [ValueChecker(">=", 0)],
    "maximum_height": [ValueChecker(">", 0)],
    "maximum_width": [ValueChecker(">", 0)],
    "batch_size": [ValueChecker(">", 0)],
    # regularizer
    "weight": [ValueChecker(">=", 0.0)],
    # Postprocessing config.
    "coverage_threshold": [ValueChecker(">=", 0.), ValueChecker("<=", 1.0)],
    "minimum_bounding_box_height": [ValueChecker(">=", 0.)],
    "confidence_threshold": [ValueChecker(">=", 0),
                             ValueChecker("<=", 1.0)],
    "nms_iou_threshold": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "nms_confidence_threshold": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "dbscan_eps": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "dbscan_min_samples": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "neighborhood_size": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "dbscan_confidence_threshold": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    # augmentation_config
    "min_bbox_width": [ValueChecker(">=", 0.0)],
    "min_bbox_height": [ValueChecker(">=", 0.0)],
    "output_image_width": [ValueChecker(">", 0), ValueChecker("%", 16)],
    "output_image_height": [ValueChecker(">", 0), ValueChecker("%", 16)],
    "output_channel": [ValueChecker("in", [1, 3])],
    # spatial augmentation config
    "hflip_probability": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "vflip_probability": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "zoom_min": [ValueChecker(">=", 0)],
    "zoom_max": [ValueChecker(">=", 0)],
    "translate_max_x": [ValueChecker(">=", 0)],
    "translate_max_y": [ValueChecker(">=", 0)],
    # color augmentation parameters
    "color_shift_stddev": [ValueChecker(">=", 0.0), ValueChecker("<=", 1.0)],
    "hue_rotation_max": [ValueChecker(">=", 0)],
    "saturation_shift_max": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "contrast_scale_max": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "contrast_center": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "tfrecords_path": [ValueChecker("!=", "")],
    "image_directory_path": [ValueChecker("!=", "")],
    # cost scaling config,
    "initial_exponent": [ValueChecker(">", 0.0)],
    "increment": [ValueChecker(">", 0.0)],
    "decrement": [ValueChecker(">", 0.0)],
    # optimizer config
    "epsilon": [ValueChecker(">", 0.0)],
    "beta1": [ValueChecker(">", 0.0)],
    "beta2": [ValueChecker(">", 0.0)],
    # Cost function config.
    "name": [ValueChecker("!=", "")],
    "class_weight": [ValueChecker(">=", 0.0)],
    "coverage_foreground_weight": [ValueChecker(">=", 0.0)],
    "initial_weight": [ValueChecker(">=", 0.0)],
    "weight_target": [ValueChecker(">=", 0.0)],
}

TRAINVAL_EXP_REQUIRED_MSG = ["model_config", "training_config", "evaluation_config",
                             "cost_function_config", "augmentation_config",
                             "bbox_rasterizer_config", "postprocessing_config",
                             "dataset_config"]
EVALUATION_EXP_REQUIRED_MSG = ["model_config", "training_config", "evaluation_config",
                               "augmentation_config", "postprocessing_config", "dataset_config",
                               "cost_function_config", "bbox_rasterizer_config"]
INFERENCE_EXP_REQUIRED_MSG = ["inferencer_config", "bbox_handler_config"]

INFERENCE_REQUIRED_MSG_DICT = {
    "inferencer_config": [
        "model_config_type", "batch_size",
        "image_height", "image_width", "image_channels",
        "target_classes"
    ],
    "tlt_config": ["model"],
    "calibrator_config": [
        "calibration_cache"
    ],
    "bbox_handler_config": [
        "classwise_bbox_handler_config",
        "confidence_model",
        "output_map",
        "bbox_color",
        "clustering_config"
    ]
}

TRAINVAL_REQUIRED_MSG_DICT = {
    # Required parameter for augmentation config.
    "augmentation_config": ["preprocessing"],
    # Required parameter for bbox rasterizer config.
    "bbox_rasterizer_config": [
        "target_class_config",
        "dead_zone_radius"
    ],
    # Required parameters for the target_class_config.
    "target_class_config": [
        "cov_center_x", "cov_center_y",
        "cov_radius_x", "cov_radius_y",
        "bbox_min_radius"
    ],
    # Required parameter of the training_config.
    "training_config": [
        "num_epochs",
        "learning_rate",
        "regularizer",
        "optimizer",
        "cost_scaling"
    ],
    "optimizer": ["adam"],
    "adam": ["epsilon", "beta", "gamma"],
    "cost_scaling": ["initial_exponent", "increment, decrement"],
    # Required parameters for the evaluation config.
    "evaluation_config": ["minimum_detection_ground_truth_overlap",
                          "evaluation_box_config"],
    # Required parameters for the cost_function_config.
    "cost_function_config": ["target_classes"],
    # Required parameters for the cost_function_config, target_classes
    "target_classes": [
        "name",
        "class_weights",
        "coverage_foreground_weight",
        "objectives"
    ],
    "objectives": [
        "name"
    ],
    "postprocessing_config": ["target_class_config"],
    "clustering_config": [
        "coverage_threshold"
    ],
    "soft_start_annealing_schedule": [
        "min_learning_rate",
        "max_learning_rate",
        "soft_start",
        "annealing"
    ],
    "early_stopping_annealing_schedule": [
        "min_learning_rate",
        "max_learning_rate",
        "soft_start_epochs",
        "annealing_epochs",
        "patience_steps"
    ],
    "dataset_config": ["data_sources", "target_class_mapping"],
    "data_sources": ["tfrecords_path", "image_directory_path"],
    # model_config
    "model_config": ["objective_set", "arch"],
    "objective_set": ["bbox", "cov"],
    "bbox": ["scale", "offset"],
}

INFERENCE_VALUE_CHECK_DICT = {
    # inferencer config
    "batch_size": [ValueChecker(">", 0)],
    "image_height": [ValueChecker(">", 0)],
    "image_width": [ValueChecker(">", 0)],
    "image_channels": [ValueChecker("in", [1, 3])],
    # calibrator_config
    "calibration_cache": [ValueChecker("!=", "")],
    "coverage_threshold": [ValueChecker(">=", 0.), ValueChecker("<=", 1.0)],
    "minimum_bounding_box_height": [ValueChecker(">=", 0.)],
}

INFERENCE_OPTIONAL_CHECK_DICT = {
    "calibration_tensorfile": [ValueChecker("!=", "")],
    "n_batches": [ValueChecker(">", 0)],
    "etlt_model": [ValueChecker("!=", "")],
    "caffemodel": [ValueChecker("!=", "")],
    "prototxt": [ValueChecker("!=", "")],
    "uff_model": [ValueChecker("!=", "")],
    "trt_engine": [ValueChecker("!=", "")],
    "nms_iou_threshold": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "nms_confidence_threshold": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "dbscan_eps": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "dbscan_min_samples": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "neighborhood_size": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
    "dbscan_confidence_threshold": [ValueChecker(">=", 0), ValueChecker("<=", 1.0)],
}
