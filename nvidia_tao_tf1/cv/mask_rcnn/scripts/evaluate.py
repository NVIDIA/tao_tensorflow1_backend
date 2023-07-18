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

"""Evaluation script for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from google.protobuf.json_format import MessageToDict
import six

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.mask_rcnn.dataloader import dataloader
from nvidia_tao_tf1.cv.mask_rcnn.executer import distributed_executer
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import mask_rcnn_params
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import params_io
from nvidia_tao_tf1.cv.mask_rcnn.models import mask_rcnn_model
from nvidia_tao_tf1.cv.mask_rcnn.utils.logging_formatter import log_cleaning
from nvidia_tao_tf1.cv.mask_rcnn.utils.spec_loader import load_experiment_spec
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# os.environ["TF_XLA_FLAGS"] = 'tf_xla_print_cluster_outputs=1'


def build_command_line_parser(parser=None):
    '''Parse command line arguments.'''
    if parser is None:
        parser = argparse.ArgumentParser(prog='Evaluate',
                                         description='Evaluate a MaskRCNN model.')
    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Path to spec file. Absolute path or relative to working directory. \
            If not specified, default spec from spec_loader.py is used.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        default="",
        required=False,
        help='Key to save or load a .tlt model.'
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        default=None,
        help='Path to a MaskRCNN model.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default='/tmp',
        required=False,
        help='Output directory where the status log is saved.'
    )
    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=False,
        default=None,
        help=argparse.SUPPRESS
    )

    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


def run_executer(runtime_config, train_input_fn=None, eval_input_fn=None):
    """Runs Mask RCNN model on distribution strategy defined by the user."""
    executer = distributed_executer.EstimatorExecuter(
        runtime_config, mask_rcnn_model.mask_rcnn_model_fn)
    eval_results = executer.eval(eval_input_fn=eval_input_fn)
    return eval_results


def main(args=None):
    """Run evaluation."""
    disable_eager_execution()
    tf.autograph.set_verbosity(0)
    log_cleaning(hide_deprecation_warnings=True)
    args = parse_command_line(args)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    status_file = os.path.join(args.results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=1,
            append=True
        )
    )
    s_logger = status_logging.get_status_logger()
    s_logger.write(
        status_level=status_logging.Status.STARTED,
        message="Starting MaskRCNN evaluation."
    )

    # ============================ Configure parameters ============================ #
    RUN_CONFIG = mask_rcnn_params.default_config()
    experiment_spec = load_experiment_spec(args.experiment_spec)
    temp_config = MessageToDict(experiment_spec,
                                preserving_proto_field_name=True,
                                including_default_value_fields=True)
    temp_config['mode'] = 'eval'
    if args.model_path:
        assert 'epoch' in args.model_path, "The pruned model must be retrained first."
        temp_config['model_path'] = args.model_path
    try:
        data_config = temp_config['data_config']
        maskrcnn_config = temp_config['maskrcnn_config']
    except Exception:
        print("Make sure data_config and maskrcnn_config are configured properly.")
    finally:
        del temp_config['data_config']
        del temp_config['maskrcnn_config']
    temp_config.update(data_config)
    temp_config.update(maskrcnn_config)
    # eval some string type params
    if 'freeze_blocks' in temp_config:
        temp_config['freeze_blocks'] = eval_str(temp_config['freeze_blocks'])
    if 'image_size' in temp_config:
        temp_config['image_size'] = eval_str(temp_config['image_size'])
    if 'learning_rate_steps' in temp_config:
        temp_config['learning_rate_steps'] = eval_str(temp_config['learning_rate_steps'])
    if 'learning_rate_decay_levels' in temp_config:
        temp_config['learning_rate_levels'] = \
            [decay * temp_config['init_learning_rate']
                for decay in eval_str(temp_config['learning_rate_decay_levels'])]
    if 'bbox_reg_weights' in temp_config:
        temp_config['bbox_reg_weights'] = eval_str(temp_config['bbox_reg_weights'])
    if 'aspect_ratios' in temp_config:
        temp_config['aspect_ratios'] = eval_str(temp_config['aspect_ratios'])
    # force some params to default value
    temp_config['use_fake_data'] = False
    temp_config['allow_xla_at_inference'] = False
    # load model from json graphs in the same dir as the checkpoint
    temp_config['pruned_model_path'] = os.path.dirname(args.model_path)
    assert temp_config['num_classes'] > 1, "Please verify num_classes in the spec file. \
        num_classes should be number of categories in json + 1."
    # use experiment spec to overwrite default hparams
    RUN_CONFIG = params_io.override_hparams(RUN_CONFIG, temp_config)
    RUN_CONFIG.key = args.key

    if not RUN_CONFIG.validation_file_pattern:
        raise RuntimeError('You must specify `validation_file_pattern` for evaluation.')

    if RUN_CONFIG.val_json_file == "" and not RUN_CONFIG.include_groundtruth_in_features:
        raise RuntimeError(
            'You must specify `val_json_file` or \
                include_groundtruth_in_features=True for evaluation.')

    if not (RUN_CONFIG.include_groundtruth_in_features or os.path.isfile(RUN_CONFIG.val_json_file)):
        raise FileNotFoundError("Validation JSON File not found: %s" % RUN_CONFIG.val_json_file)

    eval_input_fn = dataloader.InputReader(
        file_pattern=RUN_CONFIG.validation_file_pattern,
        mode=tf.estimator.ModeKeys.PREDICT,
        num_examples=RUN_CONFIG.eval_samples,
        use_fake_data=False,
        use_instance_mask=RUN_CONFIG.include_mask,
        seed=RUN_CONFIG.seed
    )

    eval_results = run_executer(RUN_CONFIG, None, eval_input_fn)
    for k, v in eval_results.items():
        s_logger.kpi[k] = float(v)
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Evaluation finished successfully."
    )


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
