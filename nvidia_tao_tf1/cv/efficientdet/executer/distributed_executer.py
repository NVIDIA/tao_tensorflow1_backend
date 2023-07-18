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
"""Interface to run EfficientDet distributed strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import logging
import math
import multiprocessing
import operator
import os
import tempfile
from zipfile import BadZipFile, ZipFile

import horovod.tensorflow as hvd
import keras
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

from nvidia_tao_tf1.core.utils.path_utils import expand_path
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.efficientdet.hooks.enc_ckpt_hook import EncryptCheckpointSaverHook
from nvidia_tao_tf1.cv.efficientdet.hooks.logging_hook import TaskProgressMonitorHook
from nvidia_tao_tf1.cv.efficientdet.hooks.pretrained_restore_hook import \
    PretrainedWeightsLoadingHook
from nvidia_tao_tf1.cv.efficientdet.utils.distributed_utils import MPI_is_distributed
from nvidia_tao_tf1.cv.efficientdet.utils.distributed_utils import MPI_local_rank, MPI_rank
from nvidia_tao_tf1.cv.efficientdet.utils.model_loader import load_keras_model
from nvidia_tao_tf1.encoding import encoding
hvd.init()
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' if not os.environ.get('CUDA_VISIBLE_DEVICES') \
#     else os.environ['CUDA_VISIBLE_DEVICES'].split(',')[hvd.local_rank()]


@six.add_metaclass(abc.ABCMeta)
class BaseExecuter(object):
    """Interface to run EfficientDet model in GPUs.

    Arguments:
        model_config: Model configuration needed to run distribution strategy.
        model_fn: Model function to be passed to Estimator.
    """

    def __init__(self, runtime_config, model_fn):
        """Initialize."""
        self._runtime_config = runtime_config
        self._model_fn = model_fn
        self._temp_dir = tempfile.mkdtemp()
        self.curr_step = 0
        # To resume from checkpoint
        # old tmp dir need to be retrieved
        if runtime_config.mode == 'train':
            tmp_path = self.get_latest_checkpoint(runtime_config.model_dir, runtime_config.key)
            if tmp_path:
                with open(os.path.join(self._temp_dir, "checkpoint"), "r") as f:
                    old_path = f.readline()
                    old_path = eval(old_path.split(":")[-1])
                    self._temp_dir = os.path.dirname(old_path)
                    # shutil.rmtree(os.path.dirname(tmp_path))
                    ckpt_path = self.get_latest_checkpoint(runtime_config.model_dir,
                                                           runtime_config.key)
                    self._runtime_config.checkpoint = tmp_path
                    self.curr_step = int(ckpt_path.split('.')[1].split('-')[1])

            # Set status logger
            status_logging.set_status_logger(
                status_logging.StatusLogger(
                    filename=os.path.join(runtime_config.model_dir, "status.json"),
                    is_master=hvd.rank() == 0,
                    verbosity=status_logging.Verbosity.INFO,
                    append=True
                )
            )
            s_logger = status_logging.get_status_logger()
            s_logger.write(
                status_level=status_logging.Status.STARTED,
                message="Starting EfficientDet training."
            )
        os.environ['CUDA_CACHE_DISABLE'] = '0'

        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

        os.environ['TF_ADJUST_HUE_FUSED'] = '1'
        os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    @staticmethod
    def _get_session_config(mode, use_xla, use_amp, use_tf_distributed=False,
                            allow_xla_at_inference=False):

        assert mode in ('train', 'eval')

        rewrite_options = rewriter_config_pb2.RewriterConfig(
            meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.TWO)

        if use_amp:
            logging.info("[%s] AMP is activated - Experiment Feature" % mode)
            rewrite_options.auto_mixed_precision = True

        config = tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                graph_options=tf.compat.v1.GraphOptions(
                    rewrite_options=rewrite_options,
                    # infer_shapes=True  # Heavily drops throughput by 30%
                ),
                gpu_options=tf.compat.v1.GPUOptions(
                    allow_growth=True,
                )
        )

        if use_tf_distributed:
            config.gpu_options.force_gpu_compatible = False

        else:
            config.gpu_options.force_gpu_compatible = True  # Force pinned memory
            config.gpu_options.allow_growth = True
            if MPI_is_distributed():
                config.gpu_options.visible_device_list = str(MPI_local_rank())
        if use_xla and (mode == "train" or allow_xla_at_inference):
            logging.info("[%s] XLA is activated - Experiment Feature" % mode)
            config.graph_options.optimizer_options.global_jit_level = \
                tf.compat.v1.OptimizerOptions.ON_1

        if mode == 'train':
            config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads

            if MPI_is_distributed():
                config.inter_op_parallelism_threads = \
                    max(2, multiprocessing.cpu_count() // hvd.local_size())

            elif not use_tf_distributed:
                config.inter_op_parallelism_threads = 4

        return config

    @abc.abstractmethod
    def build_strategy_configuration(self, mode):
        """Builds run configuration for distributed train/eval.

        Returns:
            RunConfig with distribution strategy configurations
            to pass to the constructor of TPUEstimator/Estimator.
        """

        raise NotImplementedError('Must be implemented in subclass')

    def build_model_parameters(self, mode):
        """Builds model parameter."""

        assert mode in ('train', 'eval')

        batch_size = self._runtime_config.train_batch_size \
            if mode == 'train' else self._runtime_config.eval_batch_size

        params = dict(
            self._runtime_config.as_dict(),
            mode=mode,
            batch_size=batch_size,
            # model_dir=self._runtime_config.model_dir,
        )

        if mode == 'eval':
            params = dict(
                params,
                input_rand_hflip=False,
                is_training_bn=False,
                precision=None,
            )

        return params

    def build_efficientdet_estimator(self, params, run_config, mode):
        """Creates Estimator instance.

        Arguments:
            params: A dictionary to pass to Estimator `model_fn`.
            run_config: RunConfig instance specifying distribution strategy
                configurations.
            mode: Mode -- one of 'train` or `eval`.

        Returns:
            TFEstimator instance.
        """
        assert mode in ('train', 'eval')

        return tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self._temp_dir,
            config=run_config,
            params=params
        )

    def get_training_hooks(self, mode, params):
        """Set up training hooks."""
        assert mode in ('train', 'eval')

        training_hooks = []
        steps_per_epoch = (params['num_examples_per_epoch'] + params['batch_size'] - 1) \
            // params['batch_size']

        if not MPI_is_distributed() or MPI_rank() == 0:
            training_hooks.append(
                TaskProgressMonitorHook(
                    params['batch_size'],
                    epochs=params['num_epochs'],
                    steps_per_epoch=steps_per_epoch,
                    logging_frequency=params['logging_frequency']))

            training_hooks.append(EncryptCheckpointSaverHook(
                checkpoint_dir=params['model_dir'],
                temp_dir=self._temp_dir,
                key=params['key'],
                checkpoint_basename="model.ckpt",
                steps_per_epoch=steps_per_epoch
            ))

            if params.get('checkpoint', None):
                checkpoint_path = self.load_pretrained_model(
                    params['checkpoint'], params.get('pruned_model_path', ''))

                training_hooks.append(PretrainedWeightsLoadingHook(
                    prefix="",
                    checkpoint_path=checkpoint_path,
                    skip_variables_regex=params.get('skip_checkpoint_variables', None)
                ))

        if MPI_is_distributed() and mode == "train":
            training_hooks.append(hvd.BroadcastGlobalVariablesHook(root_rank=0))
            # stop training after x epochs
            if params['stop_at_epoch']:
                stop_hook = tf.estimator.StopAtStepHook(
                    last_step=params['stop_at_epoch'] * steps_per_epoch)
                training_hooks.append(stop_hook)

        return training_hooks

    def load_pretrained_model(self, checkpoint_path, pruned_model_path=''):
        """Load pretrained model."""
        is_pruned = bool(pruned_model_path)
        _, ext = os.path.splitext(checkpoint_path)
        if ext == '.hdf5':
            logging.info("Loading pretrained model...")
            load_keras_model(checkpoint_path, is_pruned)
            km_weights = tf.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                scope=None)
            with tempfile.NamedTemporaryFile() as f:
                if is_pruned:
                    checkpoint_path = tf.train.Saver(km_weights).save(
                        tf.keras.backend.get_session(), f.name)
                else:
                    checkpoint_path = tf.train.Saver(km_weights).save(
                        keras.backend.get_session(), f.name)
            return checkpoint_path
        if ext == '.tlt':
            """Get unencrypted checkpoint from tlt file."""
            raise ValueError("You shouldn't be here.")
        if '.ckpt' in ext:
            return checkpoint_path
        raise ValueError("Pretrained weights in only .hdf5 or .tlt format are supported.")

    def get_latest_checkpoint(self, results_dir, key):
        """Get the latest checkpoint path from a given results directory.

        Parses through the directory to look for the latest checkpoint file
        and returns the path to this file.

        Args:
                results_dir (str): Path to the results directory.

        Returns:
                ckpt_path (str): Path to the latest checkpoint.
        """
        if not os.path.exists(results_dir):
            return None
        trainable_ckpts = [int(item.split('.')[1].split('-')[1])
                           for item in os.listdir(results_dir) if item.endswith(".tlt")]
        num_ckpts = len(trainable_ckpts)
        if num_ckpts == 0:
            return None
        latest_step = sorted(trainable_ckpts, reverse=True)[0]
        latest_checkpoint = os.path.join(results_dir, "model.epoch-{}.tlt".format(latest_step))
        return self.extract_ckpt(latest_checkpoint, key)

    def extract_zip_file(self, zip_file):
        """Get the checkpoint file.

        Args:
            zip_file (str): Path to the zip file.
        """
        with ZipFile(zip_file, "r") as zip_object:
            for member in zip_object.namelist():
                zip_object.extract(member, path=self._temp_dir)
                if member.startswith('model.ckpt-'):
                    step = int(member.split('model.ckpt-')[-1].split('.')[0])
        return expand_path(f"{self._temp_dir}/model.ckpt-{step}")

    def extract_ckpt(self, encoded_checkpoint, key):
        """Get unencrypted checkpoint from tlt file."""
        logging.info("Loading weights from {}".format(encoded_checkpoint))
        try:
            extracted_ckpt_path = self.extract_zip_file(encoded_checkpoint)
        except BadZipFile:
            os_handle, temp_zip_path = tempfile.mkstemp()
            os.close(os_handle)

            # Decrypt the checkpoint file.
            with open(encoded_checkpoint, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zipf:
                encoding.decode(encoded_file, tmp_zipf, key.encode())
            encoded_file.closed
            tmp_zipf.closed
            extracted_ckpt_path = self.extract_zip_file(temp_zip_path)
        except Exception:
            raise IOError("The last checkpoint file is not saved properly. \
                Please delete it and rerun the script.")
        return extracted_ckpt_path

    def train_and_eval(self, train_input_fn, eval_input_fn):
        """Run distributed train and eval on EfficientDet model."""

        # check whether to resume
        ckpt_cycle = 0
        if self.curr_step:
            ckpt_epoch = math.ceil(
                self.curr_step * self._runtime_config.train_batch_size /
                self._runtime_config.num_examples_per_epoch)
            logging.info('Resume training from the latest checkpoint step: {}.'.format(
                self.curr_step))
            ckpt_cycle = ckpt_epoch // self._runtime_config.eval_epoch_cycle

        train_run_config = self.build_strategy_configuration('train')
        train_params = self.build_model_parameters('train')
        train_estimator = self.build_efficientdet_estimator(train_params, train_run_config, 'train')

        eval_estimator = None
        eval_results = None

        training_hooks = self.get_training_hooks(
            mode="train",
            params=train_params,
        )

        max_steps_per_cycle = self._runtime_config.eval_epoch_cycle * \
            self._runtime_config.num_examples_per_epoch // self._runtime_config.train_batch_size
        # Starting training cycle
        for cycle in range(ckpt_cycle + 1,
                           self._runtime_config.num_epochs //
                           self._runtime_config.eval_epoch_cycle + 1):
            epoch = (cycle - 1) * self._runtime_config.eval_epoch_cycle
            logging.info('Starting training cycle: %d, epoch: %d.', cycle, epoch)

            train_estimator.train(
                input_fn=train_input_fn,
                max_steps=int(max_steps_per_cycle * cycle),
                hooks=training_hooks
            )

            if (not MPI_is_distributed() or MPI_rank() == 0):

                print()  # Visual Spacing
                logging.info("=================================")
                logging.info('    Start evaluation cycle %02d' % cycle)
                logging.info("=================================\n")

                if eval_estimator is None:
                    eval_run_config = self.build_strategy_configuration('eval')
                    eval_params = self.build_model_parameters('eval')
                    eval_estimator = self.build_efficientdet_estimator(eval_params,
                                                                       eval_run_config, 'eval')

                ckpt_path = self.get_latest_checkpoint(self._runtime_config.model_dir,
                                                       self._runtime_config.key)
                eval_results = eval_estimator.evaluate(
                    input_fn=eval_input_fn,
                    checkpoint_path=ckpt_path,
                    steps=self._runtime_config.eval_samples // self._runtime_config.eval_batch_size,
                    name='Eval')
                for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
                    logging.info("%s: %.9f" % (key, value))
                print("Evaluation is complete.")

                kpi_data = {
                    k: float(round(v * 100, 4))
                    for k, v in sorted(eval_results.items(), key=operator.itemgetter(0))
                }

                s_logger = status_logging.get_status_logger()
                if isinstance(s_logger, status_logging.StatusLogger):
                    s_logger.kpi = kpi_data
                    s_logger.write(
                        status_level=status_logging.Status.RUNNING,
                        message="Evaluation metrics generated."
                    )

            if MPI_is_distributed():
                logging.info("Training Cycle: {} complete".format(cycle))
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()  # Waiting for all MPI processes to sync

        return eval_results

    def eval(self, eval_input_fn):
        """Run eval with EfficientDet model."""
        # model_path = os.path.join(self._runtime_config.model_dir)
        print(self._runtime_config.model_path)
        ckpt_path = self.extract_ckpt(self._runtime_config.model_path, self._runtime_config.key)
        if (not MPI_is_distributed() or MPI_rank() == 0):

            print()  # Visual Spacing
            logging.info("=================================")
            logging.info('    Start evaluation')
            logging.info("=================================\n")

            eval_run_config = self.build_strategy_configuration('eval')
            eval_params = self.build_model_parameters('eval')
            eval_estimator = self.build_efficientdet_estimator(eval_params,
                                                               eval_run_config, 'eval')

            eval_results = eval_estimator.evaluate(
                input_fn=eval_input_fn,
                steps=self._runtime_config.eval_samples // self._runtime_config.eval_batch_size,
                checkpoint_path=ckpt_path,
                name='Eval')
            for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
                logging.info("%s: %.9f" % (key, value))
            print("Evaluation is complete.")

        return eval_results


class EstimatorExecuter(BaseExecuter):
    """Interface that runs EfficientDet model using Estimator."""

    def __init__(self, runtime_config, model_fn):
        """Initialize."""
        super(EstimatorExecuter, self).__init__(runtime_config, model_fn)

        if MPI_is_distributed():
            os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
            os.environ['HOROVOD_NUM_NCCL_STREAMS'] = '1'
            # os.environ['HOROVOD_AUTOTUNE'] = '2'
            # hvd.init()
            logging.info("Horovod successfully initialized ...")

        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '1' if not MPI_is_distributed() else str(hvd.size())

        os.environ['TF_SYNC_ON_FINISH'] = '0'

    def build_strategy_configuration(self, mode):
        """Retrieves model configuration for running TF Estimator."""

        run_config = tf.estimator.RunConfig(
            tf_random_seed=(
                self._runtime_config.tf_random_seed
                if not MPI_is_distributed() or self._runtime_config.tf_random_seed is None else
                self._runtime_config.tf_random_seed + MPI_rank()
            ),
            # model_dir=self._runtime_config.model_dir,
            save_summary_steps=None,  # disabled
            save_checkpoints_steps=None,  # disabled
            save_checkpoints_secs=None,  # disabled
            keep_checkpoint_max=20,  # disabled
            keep_checkpoint_every_n_hours=None,  # disabled
            log_step_count_steps=None,  # disabled
            session_config=self._get_session_config(
                mode=mode,
                use_xla=False,  # self._runtime_config.use_xla
                use_amp=self._runtime_config.amp,
                use_tf_distributed=False,
                # TODO: Remove when XLA at inference fixed
                allow_xla_at_inference=False,  # self._runtime_config.allow_xla_at_inference
            ),
            protocol=None,
            device_fn=None,
            train_distribute=None,
            eval_distribute=None,
            experimental_distribute=None
        )
        return run_config
