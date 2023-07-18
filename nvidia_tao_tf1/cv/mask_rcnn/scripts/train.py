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

"""Training script for Mask-RCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import tempfile
from zipfile import BadZipFile, ZipFile
from google.protobuf.json_format import MessageToDict

import six

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

import wandb

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.mask_rcnn.dataloader import dataloader
from nvidia_tao_tf1.cv.mask_rcnn.executer import distributed_executer
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import mask_rcnn_params
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import params_io
from nvidia_tao_tf1.cv.mask_rcnn.models import mask_rcnn_model
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_is_distributed
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_rank
from nvidia_tao_tf1.cv.mask_rcnn.utils.logging_formatter import log_cleaning
from nvidia_tao_tf1.cv.mask_rcnn.utils.spec_loader import load_experiment_spec
from nvidia_tao_tf1.encoding import encoding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
# os.environ["TF_XLA_FLAGS"] = 'tf_xla_print_cluster_outputs=1'
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    '''Parse command line arguments.'''
    if parser is None:
        parser = argparse.ArgumentParser(prog='train', description='Train a MaskRCNN model.')
    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        required=True,
        help='Path to spec file. Absolute path or relative to working directory. \
            If not specified, default spec from spec_loader.py is used.')
    parser.add_argument(
        '-k',
        '--key',
        default="",
        type=str,
        required=False,
        help='Key to save or load a .tlt model.'
    )
    parser.add_argument(
        '-d',
        '--model_dir',
        type=str,
        required=True,
        default=None,
        help='Dir to save or load a .tlt model.'
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
    executer = distributed_executer.EstimatorExecuter(runtime_config,
                                                      mask_rcnn_model.mask_rcnn_model_fn)
    executer.train_and_eval(train_input_fn=train_input_fn, eval_input_fn=eval_input_fn)
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.SUCCESS,
        message="Training finished successfully."
    )


def extract_pruned_checkpoint(ckpt_zip_file):
    """Simple function to extract the checkpoint file."""
    temp_dir = tempfile.mkdtemp()
    with ZipFile(ckpt_zip_file, 'r') as zip_object:
        zip_object.extractall(path=temp_dir)
    return temp_dir


def main(args=None):
    """Start training."""
    disable_eager_execution()
    tf.autograph.set_verbosity(0)
    log_cleaning(hide_deprecation_warnings=True)
    args = parse_command_line(args)

    # ============================ Configure parameters ============================ #
    RUN_CONFIG = mask_rcnn_params.default_config()
    experiment_spec = load_experiment_spec(args.experiment_spec_file)
    wandb_config = None
    if experiment_spec.HasField("wandb_config"):
        wandb_config = experiment_spec.wandb_config
    clearml_config = None
    if experiment_spec.HasField("clearml_config"):
        clearml_config = experiment_spec.clearml_config
    temp_config = MessageToDict(experiment_spec,
                                preserving_proto_field_name=True,
                                including_default_value_fields=True)

    temp_config['mode'] = 'train'
    if args.model_dir:
        temp_config['model_dir'] = args.model_dir
    try:
        data_config = temp_config['data_config']
        maskrcnn_config = temp_config['maskrcnn_config']
    except ValueError:
        print("Make sure data_config and maskrcnn_config are configured properly.")
    finally:
        del temp_config['data_config']
        del temp_config['maskrcnn_config']
        if "wandb_config" in temp_config.keys():
            del temp_config['wandb_config']
        if "clearml_config" in temp_config.keys():
            del temp_config["clearml_config"]

    wandb_logged_in = False
    # Setup MLOPs only on rank 0.
    if not MPI_is_distributed() or MPI_rank() == 0:
        if wandb_config is not None:
            wandb_logged_in = check_wandb_logged_in()
            wandb_name = f"{wandb_config.name}" if wandb_config.name \
                else "mask_rcnn"
            wandb_stream_config = MessageToDict(
                experiment_spec,
                preserving_proto_field_name=True,
                including_default_value_fields=True
            )
            initialize_wandb(
                project=wandb_config.project if wandb_config.project else None,
                entity=wandb_config.entity if wandb_config.entity else None,
                config=wandb_stream_config,
                notes=wandb_config.notes if wandb_config.notes else None,
                tags=wandb_config.tags if wandb_config.tags else None,
                sync_tensorboard=True,
                save_code=False,
                results_dir=args.model_dir,
                wandb_logged_in=wandb_logged_in,
                name=wandb_name
            )
        if clearml_config is not None:
            get_clearml_task(clearml_config, "mask_rcnn")
    temp_config.update(data_config)
    temp_config.update(maskrcnn_config)
    # eval some string type params
    if 'freeze_blocks' in temp_config:
        temp_config['freeze_blocks'] = eval_str(temp_config['freeze_blocks'])
    if 'image_size' in temp_config:
        temp_config['image_size'] = eval_str(temp_config['image_size'])
    else:
        raise ValueError("image_size is not set.")
    max_stride = 2 ** temp_config['max_level']
    if temp_config['image_size'][0] % max_stride != 0 \
            or temp_config['image_size'][1] % max_stride != 0:
        raise ValueError('input size must be divided by the stride {}.'.format(max_stride))
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
    temp_config['max_num_instances'] = temp_config['max_num_instances'] or 200
    if 'num_steps_per_eval' in temp_config:
        temp_config['save_checkpoints_steps'] = temp_config['num_steps_per_eval']
    else:
        raise ValueError("num_steps_per_eval is not set.")
    assert temp_config['num_classes'] > 1, "Please verify num_classes in the spec file. \
        num_classes should be number of categories in json + 1."
    assert temp_config['num_examples_per_epoch'] > 0, "num_examples_per_epoch must be specified. \
        It should be the total number of images in the training set divided by the number of GPUs."
    if temp_config['num_epochs'] > 0:
        temp_config['total_steps'] = \
            (temp_config['num_epochs'] * temp_config['num_examples_per_epoch'] +
                temp_config['train_batch_size'] - 1) // temp_config['train_batch_size']
    elif temp_config['total_steps'] > 0:
        temp_config['num_epochs'] = \
            (temp_config['total_steps'] * temp_config['train_batch_size'] +
                temp_config['num_examples_per_epoch'] - 1) // temp_config['num_examples_per_epoch']
    else:
        raise ValueError("Either total_steps or num_epochs must be specified.")
    if temp_config['pruned_model_path']:
        if not os.path.exists(temp_config['pruned_model_path']):
            raise ValueError(
                "pruned_model_path doesn't exist. Please double check!")
        try:
            # Try to load the model without encryption.
            temp_dir = extract_pruned_checkpoint(temp_config['pruned_model_path'])
        except BadZipFile:
            # decode etlt for pruned model
            os_handle, temp_zip_path = tempfile.mkstemp()
            os.close(os_handle)

            # Decrypt the checkpoint file.
            with open(temp_config['pruned_model_path'], 'rb') as encoded_file, \
                    open(temp_zip_path, 'wb') as tmp_zipf:
                encoding.decode(encoded_file, tmp_zipf, args.key.encode())
            encoded_file.closed
            tmp_zipf.closed
            # Load zip file and extract members to a tmp_directory.
            temp_dir = extract_pruned_checkpoint(temp_zip_path)
            os.remove(temp_zip_path)
        except Exception:
            raise IOError("The pruned model wasn't saved properly. \
                Please delete it and rerun the pruning script.")
            # Removing the temporary zip path.
        temp_config['pruned_model_path'] = temp_dir
        temp_config['checkpoint'] = os.path.join(temp_dir, 'pruned.ckpt')
        if not os.path.exists(temp_config['checkpoint']):
            raise ValueError("An unpruned TLT model shouldn't be used with pruned_model_path!")

    # use experiment spec to overwrite default hparams
    RUN_CONFIG = params_io.override_hparams(RUN_CONFIG, temp_config)
    RUN_CONFIG.key = args.key
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
        # raise ValueError(
        #     "{} doesn't exist. Please create the output directory first.".format(args.model_dir))

    # ============================ Configure parameters ============================ #

    if not RUN_CONFIG.training_file_pattern:
        raise RuntimeError('You must specify `training_file_pattern` for training.')

    if not RUN_CONFIG.validation_file_pattern:
        raise RuntimeError('You must specify `validation_file_pattern` for evaluation.')

    if RUN_CONFIG.val_json_file == "" and not RUN_CONFIG.include_groundtruth_in_features:
        raise RuntimeError(
            'You must specify `val_json_file` or \
                include_groundtruth_in_features=True for evaluation.')

    if not (RUN_CONFIG.include_groundtruth_in_features or os.path.isfile(RUN_CONFIG.val_json_file)):
        raise FileNotFoundError("Validation JSON File not found: %s" % RUN_CONFIG.val_json_file)

    train_input_fn = dataloader.InputReader(
        file_pattern=RUN_CONFIG.training_file_pattern,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_examples=None,
        use_fake_data=RUN_CONFIG.use_fake_data,
        use_instance_mask=RUN_CONFIG.include_mask,
        seed=RUN_CONFIG.seed
    )

    eval_input_fn = dataloader.InputReader(
        file_pattern=RUN_CONFIG.validation_file_pattern,
        mode=tf.estimator.ModeKeys.PREDICT,
        num_examples=RUN_CONFIG.eval_samples,
        use_fake_data=False,
        use_instance_mask=RUN_CONFIG.include_mask,
        seed=RUN_CONFIG.seed
    )
    try:
        run_executer(RUN_CONFIG, train_input_fn, eval_input_fn)
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
        logger.info("Training was interrupted.")
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
    finally:
        # Close wandb session.
        if not MPI_is_distributed() or MPI_rank() == 0:
            wandb.finish()


if __name__ == '__main__':
    main()
