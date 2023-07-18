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
    # model config optional parameters.
    "n_layers": [ValueChecker(">", 0)],
    "freeze_blocks": [ValueChecker(">=", 0)],
}

TRAINVAL_VALUE_CHECK_DICT = {
    # model config parameters.
    "arch": [ValueChecker("!=", ""),
             ValueChecker("in", ["resnet",
                                 "vgg",
                                 "alexnet",
                                 "googlenet",
                                 "mobilenet_v1",
                                 "mobilenet_v2",
                                 "squeezenet",
                                 "darknet",
                                 "efficientnet_b0",
                                 "efficientnet_b1",
                                 "efficientnet_b2",
                                 "efficientnet_b3",
                                 "efficientnet_b4",
                                 "efficientnet_b5",
                                 "efficientnet_b6",
                                 "efficientnet_b7",
                                 "cspdarknet",
                                 "cspdarknet_tiny"])],
    "input_image_size": [ValueChecker("!=", "")],
    "activation_type": [ValueChecker("!=", "")],
    # training config
    "n_workers": [ValueChecker(">", 0)],
    "label_smoothing": [ValueChecker(">=", 0.0)],
    "mixup_alpha": [ValueChecker(">=", 0.0)],
    "train_dataset_path": [ValueChecker("!=", "")],
    "val_dataset_path": [ValueChecker("!=", "")],
    "n_epochs": [ValueChecker(">", 0)],
    "batch_size_per_gpu": [ValueChecker(">", 0)],
    "preprocess_mode": [ValueChecker("in", ["tf", "caffe", "torch"])],
    # evaluation config required parameters.
    "eval_dataset_path": [ValueChecker("!=", "")],
    # Learning rate scheduler config.
    "learning_rate": [ValueChecker(">", 0.0)],
    # optimizer config.
    "momentum": [ValueChecker(">=", 0)],
    "epsilon": [ValueChecker(">=", 0)],
    "rho": [ValueChecker(">=", 0)],
    "beta_1": [ValueChecker(">=", 0)],
    "beta_2": [ValueChecker(">=", 0)],
    "decay": [ValueChecker(">=", 0)],
    "dropout": [ValueChecker(">=", 0.0)],
    "step_size": [ValueChecker(">=", 0)],
    "gamma": [ValueChecker(">=", 0.0)],
    "soft_start": [ValueChecker(">=", 0)],
    "annealing_divider": [ValueChecker(">=", 0)],
    "annealing_points": [ValueChecker(">=", 0)],
    "min_lr_ratio": [ValueChecker(">=", 0.0)],
    "lr": [ValueChecker(">", 0.0)],
    # regularizer config
    "type": [ValueChecker("!=", ""), ValueChecker("in", ["L1", "L2", "None"])],
    "scope": [ValueChecker("!=", "")],
    "weight_decay": [ValueChecker(">", 0.0)],
}

TRAINVAL_EXP_REQUIRED_MSG = ["model_config", "train_config"]
VALIDATION_EXP_REQUIRED_MSG = TRAINVAL_EXP_REQUIRED_MSG + ["eval_config"]

TRAINVAL_REQUIRED_MSG_DICT = {
    "model_config": ["arch", "input_image_size"],
    "eval_config": ["top_k", "eval_dataset_path",
                    "model_path", "batch_size"],
    "train_config": [
        "train_dataset_path", "val_dataset_path", "optimizer",
        "batch_size_per_gpu", "n_epochs", "reg_config", "lr_config"
    ],
    "reg_config": ["type", "scope", "weight_decay"]
}
