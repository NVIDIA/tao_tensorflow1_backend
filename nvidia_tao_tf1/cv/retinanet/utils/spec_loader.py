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

"""Load an experiment spec file to run RetinaNet training, evaluation, pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from google.protobuf.text_format import Merge as merge_text_proto
from nvidia_tao_tf1.core.utils.path_utils import expand_path
import nvidia_tao_tf1.cv.retinanet.proto.experiment_pb2 as experiment_pb2

logger = logging.getLogger(__name__)


def load_proto(spec_path, proto_buffer, default_spec_path=None, merge_from_default=True):
    """Load spec from file and merge with given proto_buffer instance.

    Args:
        spec_path (str): location of a file containing the custom spec proto.
        proto_buffer(pb2): protocal buffer instance to be loaded.
        default_spec_path(str): location of default spec to use if merge_from_default is True.
        merge_from_default (bool): disable default spec, if False, spec_path must be set.

    Returns:
        proto_buffer(pb2): protocol buffer instance updated with spec.
    """
    def _load_from_file(filename, pb2):
        with open(expand_path(filename), "r") as f:
            merge_text_proto(f.read(), pb2)

    # Setting this flag false prevents concatenating repeated-fields
    if merge_from_default:
        assert default_spec_path, \
               "default spec path has to be defined if merge_from_default is enabled"
        # Load the default spec
        _load_from_file(default_spec_path, proto_buffer)
    else:
        assert spec_path, "spec_path has to be defined, if merge_from_default is disabled"

    # Merge a custom proto on top of the default spec, if given
    if spec_path:
        logger.info("Merging specification from %s", spec_path)
        _load_from_file(spec_path, proto_buffer)

    return proto_buffer


def load_experiment_spec(spec_path=None, merge_from_default=False):
    """Load experiment spec from a .txt file and return an experiment_pb2.Experiment object.

    Args:
        spec_path (str): location of a file containing the custom experiment spec proto.
        dataset_export_spec_paths (list of str): paths to the dataset export specs.
        merge_from_default (bool): disable default spec, if False, spec_path must be set.

    Returns:
        experiment_spec: protocol buffer instance of type experiment_pb2.Experiment.
    """
    experiment_spec = experiment_pb2.Experiment()
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path = os.path.join(file_path, 'experiment_specs/default_spec.txt')
    experiment_spec = load_proto(spec_path, experiment_spec, default_spec_path,
                                 merge_from_default)

    # dataset_config
    assert len(experiment_spec.dataset_config.target_class_mapping.values()) > 0, \
        "Please specify target_class_mapping"
    assert len(experiment_spec.dataset_config.data_sources) > 0, "Please specify data sources"
    assert len(experiment_spec.dataset_config.validation_data_sources) > 0, \
        "Please specify validation data sources"

    # augmentation check is in SSD augmentation (data_augmentation_chain_original_ssd.py)
    assert experiment_spec.augmentation_config.output_channel in [1, 3], \
        "output_channel must be either 1 or 3."
    img_mean = experiment_spec.augmentation_config.image_mean
    if experiment_spec.augmentation_config.output_channel == 3:
        if img_mean:
            assert all(c in img_mean for c in ['r', 'g', 'b']) , (
                "'r', 'g', 'b' should all be present in image_mean "
                "for images with 3 channels."
            )
    else:
        if img_mean:
            assert 'l' in img_mean, (
                "'l' should be present in image_mean for images "
                "with 1 channel."
            )
    # training config
    assert experiment_spec.training_config.batch_size_per_gpu > 0, "batch size must be positive"
    assert experiment_spec.training_config.num_epochs > 0, \
        "number of training epochs (num_epochs) must be positive"
    assert (experiment_spec.training_config.checkpoint_interval or 1) > 0, \
        "checkpoint_interval must be positive"

    # eval config
    assert experiment_spec.eval_config.batch_size > 0, "batch size must be positive"
    assert 0.0 < experiment_spec.eval_config.matching_iou_threshold <= 1.0, \
        "matching_iou_threshold must be within (0, 1]"

    # nms config
    assert 0.0 < experiment_spec.nms_config.clustering_iou_threshold <= 1.0, \
        "clustering_iou_threshold must be within (0, 1]"

    # retinanet config
    assert len(eval(experiment_spec.retinanet_config.scales)) == 6, \
        "FPN should have 6 scales for configuration."
    assert len(eval(experiment_spec.retinanet_config.variances)) == 4, \
        "4 values must be specified for variance."
    assert 0 < experiment_spec.retinanet_config.focal_loss_alpha < 1, \
        "focal_loss_alpha must be within (0, 1)."
    assert 0 < experiment_spec.retinanet_config.focal_loss_gamma, \
        "focal_loss_gamma must be greater than 0."
    assert 0 < experiment_spec.retinanet_config.n_kernels, \
        "n_kernels must be greater than 0."
    assert 1 < experiment_spec.retinanet_config.feature_size, \
        "feature_size must be greater than 1."
    assert 0 < experiment_spec.retinanet_config.n_anchor_levels, \
        "n_anchor_levels must be greater than 0."

    # Validate early_stopping config
    if experiment_spec.training_config.HasField("early_stopping"):
        es = experiment_spec.training_config.early_stopping
        if es.monitor not in ["loss", "validation_loss", "val_loss"]:
            raise ValueError(
                "Only `loss` and `validation_loss` and `val_loss` are supported monitors"
                f", got {es.monitor}"
            )
        if es.min_delta < 0.:
            raise ValueError(
                f"`min_delta` should be non-negative, got {es.min_delta}"
            )
        if es.patience == 0:
            raise ValueError(
                f"`patience` should be positive, got {es.patience}"
            )

    return experiment_spec
