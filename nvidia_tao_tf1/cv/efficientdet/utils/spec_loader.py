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

"""Load an experiment spec file to run EfficientDet training, evaluation, pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from google.protobuf.text_format import Merge as merge_text_proto
import six

import nvidia_tao_tf1.cv.efficientdet.proto.experiment_pb2 as experiment_pb2
from nvidia_tao_tf1.cv.efficientdet.utils import utils

logger = logging.getLogger(__name__)


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


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
        with open(filename, "r") as f:
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
    default_spec_path = os.path.join(file_path, 'experiment_specs/default.txt')
    experiment_spec = load_proto(spec_path, experiment_spec, default_spec_path,
                                 merge_from_default)
    spec_checker(experiment_spec)
    return experiment_spec


def generate_params_from_spec(config, spec, mode):
    """Generate parameters from experient spec."""
    if spec.model_config.aspect_ratios:
        aspect_ratios = eval_str(spec.model_config.aspect_ratios)
        if not isinstance(aspect_ratios, list):
            raise SyntaxError("aspect_ratios should be a list of tuples.")
    else:
        aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    if spec.model_config.max_level != 7 or spec.model_config.min_level != 3:
        print("WARNING: min_level and max_level are forced to 3 and 7 respectively "
              "in the current version.")
    return dict(
        config.as_dict(),
        # model_config
        name=spec.model_config.model_name,
        aspect_ratios=aspect_ratios,
        anchor_scale=spec.model_config.anchor_scale or 4,
        min_level=3,
        max_level=7,
        num_scales=spec.model_config.num_scales or 3,
        freeze_bn=spec.model_config.freeze_bn,
        freeze_blocks=eval_str(spec.model_config.freeze_blocks)
        if spec.model_config.freeze_blocks else None,
        # data config
        val_json_file=spec.dataset_config.validation_json_file,
        testdev_dir=spec.dataset_config.testdev_dir,
        num_classes=spec.dataset_config.num_classes,
        max_instances_per_image=spec.dataset_config.max_instances_per_image or 100,
        skip_crowd_during_training=spec.dataset_config.skip_crowd_during_training,
        # Parse image size in case it is in string format. (H, W)
        image_size=utils.parse_image_size(spec.dataset_config.image_size),
        # augmentation config
        input_rand_hflip=spec.augmentation_config.rand_hflip,
        train_scale_min=spec.augmentation_config.random_crop_min_scale or 0.1,
        train_scale_max=spec.augmentation_config.random_crop_max_scale or 2.0,
        # train eval config
        momentum=spec.training_config.momentum or 0.9,
        iterations_per_loop=spec.training_config.iterations_per_loop,
        num_examples_per_epoch=spec.training_config.num_examples_per_epoch,
        checkpoint=spec.training_config.checkpoint,
        ckpt=None,
        mode=mode,
        checkpoint_period=spec.training_config.checkpoint_period,
        train_batch_size=spec.training_config.train_batch_size,
        eval_batch_size=spec.eval_config.eval_batch_size,
        eval_samples=spec.eval_config.eval_samples,
        stop_at_epoch=spec.training_config.stop_at_epoch,
        profile_skip_steps=spec.training_config.profile_skip_steps,
        learning_rate=spec.training_config.learning_rate,
        tf_random_seed=spec.training_config.tf_random_seed or 42,
        pruned_model_path=spec.training_config.pruned_model_path,
        moving_average_decay=spec.training_config.moving_average_decay,
        lr_warmup_epoch=spec.training_config.lr_warmup_epoch or 5,
        lr_warmup_init=spec.training_config.lr_warmup_init or 0.00001,
        amp=spec.training_config.amp,
        data_format='channels_last',
        l2_weight_decay=spec.training_config.l2_weight_decay,
        l1_weight_decay=spec.training_config.l1_weight_decay,
        clip_gradients_norm=spec.training_config.clip_gradients_norm or 5.0,
        skip_checkpoint_variables=spec.training_config.skip_checkpoint_variables,
        num_epochs=spec.training_config.num_epochs,
        eval_epoch_cycle=spec.eval_config.eval_epoch_cycle,
        logging_frequency=spec.training_config.logging_frequency or 10
    )


def spec_checker(experiment_spec):
    """Check if parameters in the spec file are valid.

    Args:
        experiment_spec (proto): experiment spec proto.
    """
    # training config
    assert experiment_spec.training_config.train_batch_size > 0, \
        "batch size must be positive."
    assert experiment_spec.training_config.checkpoint_period > 0, \
        "checkpoint interval must be positive."
    assert experiment_spec.training_config.num_examples_per_epoch > 0, \
        "Number of samples must be positive."
    assert experiment_spec.training_config.num_epochs >= \
        experiment_spec.eval_config.eval_epoch_cycle, \
        "num_epochs must be positive and no less than eval_epoch_cycle."
    assert 0 <= experiment_spec.training_config.moving_average_decay < 1, \
        "Moving average decay must be within [0, 1)."
    assert 0 < experiment_spec.training_config.lr_warmup_init < 1, \
        "The initial learning rate during warmup must be within (0, 1)."
    assert experiment_spec.training_config.learning_rate > 0, \
        "learning_rate must be positive."

    # model config
    assert experiment_spec.model_config.model_name, \
        "model_name must be specified. Choose from ['efficientdet-d0', ..., 'efficientdet-d5']."

    # eval config
    assert experiment_spec.eval_config.eval_batch_size > 0, "batch size must be positive"
    assert experiment_spec.eval_config.eval_epoch_cycle > 0, \
        "Evaluation cycle (every N epochs) must be positive."
    assert 0 < experiment_spec.eval_config.eval_samples, \
        "Number of evaluation samples must be positive."

    # dataset config
    assert experiment_spec.dataset_config.training_file_pattern, \
        "training_file_pattern must be specified."
    assert experiment_spec.dataset_config.validation_file_pattern, \
        "validation_file_pattern must be specified."
    assert experiment_spec.dataset_config.validation_json_file, \
        "validation_json_file must be specified."
    assert 1 < experiment_spec.dataset_config.num_classes, \
        "num_classes is number of categories + 1 (background). It must be greater than 1."
    assert experiment_spec.dataset_config.image_size, \
        "image size must be specified in 'hh,ww' format."
