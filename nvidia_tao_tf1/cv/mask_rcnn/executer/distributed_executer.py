"""Interface to run mask rcnn model in different distributed strategies."""
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
import multiprocessing
import os
import tempfile
from zipfile import BadZipFile, ZipFile

import horovod.tensorflow as hvd
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

from nvidia_tao_tf1.core.utils.path_utils import expand_path
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.mask_rcnn.hooks.enc_ckpt_hook import EncryptCheckpointSaverHook
from nvidia_tao_tf1.cv.mask_rcnn.hooks.logging_hook import TaskProgressMonitorHook
from nvidia_tao_tf1.cv.mask_rcnn.hooks.pretrained_restore_hook import PretrainedWeightsLoadingHook
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import params_io

from nvidia_tao_tf1.cv.mask_rcnn.utils import evaluation
from nvidia_tao_tf1.cv.mask_rcnn.utils import model_loader

from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_is_distributed
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_local_rank, MPI_rank, MPI_size
from nvidia_tao_tf1.cv.mask_rcnn.utils.logging_formatter import logging
from nvidia_tao_tf1.encoding import encoding

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if not os.environ.get('CUDA_VISIBLE_DEVICES') \
    else os.environ['CUDA_VISIBLE_DEVICES'].split(',')[MPI_local_rank()]
logging.debug(f"Number of GPU's for this process: {MPI_size()}")
logging.debug(
    f"Set environment variable for CUDA_VISIBLE_DEVICES from rank {MPI_local_rank()}: "
    f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
)


@six.add_metaclass(abc.ABCMeta)
class BaseExecuter(object):
    """Interface to run Mask RCNN model in GPUs.

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
                logging.debug(f"Contents of self._temp_dir: {tmp_path}")
                with open(os.path.join(os.path.dirname(tmp_path), "checkpoint"), "r") as f:
                    old_path = f.readline()
                    old_path = eval(old_path.split(":")[-1])
                    self._temp_dir = os.path.dirname(old_path)
                    # shutil.rmtree(os.path.dirname(tmp_path))
                    ckpt_path = self.get_latest_checkpoint(runtime_config.model_dir,
                                                           runtime_config.key)
                    self._runtime_config.checkpoint = tmp_path
                    self.curr_step = int(ckpt_path.split('.')[1].split('-')[1])
                    logging.info(f"current step from checkpoint: {self.curr_step}")

            # Set status logger
            status_logging.set_status_logger(
                status_logging.StatusLogger(
                    filename=os.path.join(runtime_config.model_dir, "status.json"),
                    is_master=(not MPI_is_distributed() or MPI_rank() == 0),
                    verbosity=status_logging.Verbosity.INFO,
                    append=True
                )
            )
            s_logger = status_logging.get_status_logger()
            s_logger.write(
                status_level=status_logging.Status.STARTED,
                message="Starting MaskRCNN training."
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
            self._runtime_config.values(),
            mode=mode,
            batch_size=batch_size,
            # model_dir=self._runtime_config.model_dir,
        )

        if mode == 'eval':
            params = dict(
                params,
                augment_input_data=False,
            )

        return params

    def build_mask_rcnn_estimator(self, params, run_config, mode):
        """Creates TPUEstimator/Estimator instance.

        Arguments:
            params: A dictionary to pass to Estimator `model_fn`.
            run_config: RunConfig instance specifying distribution strategy
                configurations.
            mode: Mode -- one of 'train` or `eval`.

        Returns:
            TFEstimator or TPUEstimator instance.
        """
        assert mode in ('train', 'eval')

        return tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self._temp_dir,
            config=run_config,
            params=params
        )

    def _save_config(self):
        """Save parameters to config files if model_dir is defined."""

        model_dir = self._runtime_config.model_dir

        if model_dir is not None:
            if not tf.io.gfile.exists(model_dir):
                tf.io.gfile.makedirs(model_dir)

            params_io.save_hparams_to_yaml(self._runtime_config, model_dir + '/params.yaml')

    def _write_summary(self, summary_dir, eval_results, predictions, current_step):

        if not self._runtime_config.visualize_images_summary:
            predictions = None

        evaluation.write_summary(eval_results, summary_dir, current_step, predictions=predictions)

    def extract_checkpoint(self, ckpt_zip_file):
        """Extract the checkpoint zip file."""
        with ZipFile(ckpt_zip_file, 'r') as zip_object:
            for member in zip_object.namelist():
                if member != 'checkpoint' and not member.endswith('.json'):
                    zip_object.extract(member, path=self._temp_dir)
                    # model.ckpt-20000.index/meta/data-00000-of-00001
                    step = int(member.split('model.ckpt-')[-1].split('.')[0])
                if 'pruned' in member:
                    raise ValueError("A pruned TLT model should only be used \
                        with pruned_model_path.")
        return os.path.join(self._temp_dir, "model.ckpt-{}".format(step))

    def load_pretrained_model(self, checkpoint_path):
        """Load pretrained model."""
        _, ext = os.path.splitext(checkpoint_path)
        if ext == '.hdf5':
            logging.info("Loading pretrained model...")
            # with tf.variable_scope('resnet50'):
            model_loader.load_keras_model(checkpoint_path)
            km_weights = tf.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                           scope=None)
            with tempfile.NamedTemporaryFile() as f:
                checkpoint_path = tf.train.Saver(km_weights).save(
                    tf.keras.backend.get_session(), f.name)
            return checkpoint_path
        if ext == '.tlt':
            """Get unencrypted checkpoint from tlt file."""
            try:
                extracted_checkpoint_path = self.extract_checkpoint(ckpt_zip_file=checkpoint_path)
            except BadZipFile:
                os_handle, temp_zip_path = tempfile.mkstemp()
                os.close(os_handle)
                # Decrypt the checkpoint file.
                with open(checkpoint_path, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zipf:
                    encoding.decode(encoded_file, tmp_zipf, self._runtime_config.key.encode())
                encoded_file.closed
                tmp_zipf.closed
                extracted_checkpoint_path = self.extract_checkpoint(temp_zip_path)
            except Exception:
                raise IOError("The last checkpoint file is not saved properly. \
                    Please delete it and rerun the script.")
            return extracted_checkpoint_path
        if '.ckpt' in ext:
            return checkpoint_path
        raise ValueError("Pretrained weights in only .hdf5 or .tlt format are supported.")

    def get_training_hooks(self, mode, model_dir, checkpoint_path=None,
                           skip_checkpoint_variables=None):
        """Set up training hooks."""
        assert mode in ('train', 'eval')

        training_hooks = []
        steps_per_epoch = (
            self._runtime_config.num_examples_per_epoch +
            self._runtime_config.train_batch_size - 1) \
            // self._runtime_config.train_batch_size
        if not MPI_is_distributed() or MPI_rank() == 0:
            training_hooks.append(
                TaskProgressMonitorHook(
                    self._runtime_config.train_batch_size,
                    epochs=self._runtime_config.num_epochs,
                    steps_per_epoch=steps_per_epoch,
                    logging_frequency=self._runtime_config.logging_frequency or 10))

        if checkpoint_path:
            if not MPI_is_distributed() or MPI_rank() == 0:
                checkpoint_path = self.load_pretrained_model(checkpoint_path)
                training_hooks.append(PretrainedWeightsLoadingHook(
                    prefix="",
                    checkpoint_path=checkpoint_path,
                    skip_variables_regex=skip_checkpoint_variables
                ))

        if MPI_is_distributed() and mode == "train":
            training_hooks.append(hvd.BroadcastGlobalVariablesHook(root_rank=0))

        if not MPI_is_distributed() or MPI_rank() == 0:
            training_hooks.append(EncryptCheckpointSaverHook(
                checkpoint_dir=model_dir,
                temp_dir=self._temp_dir,
                key=self._runtime_config.key,
                checkpoint_basename="model.ckpt",
                steps_per_epoch=steps_per_epoch,
            ))

        return training_hooks

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

    def extract_ckpt(self, encoded_checkpoint, key):
        """Get unencrypted checkpoint from tlt file."""
        logging.info("Loading weights from {}".format(encoded_checkpoint))
        checkpoint_path = model_loader.load_mrcnn_tlt_model(
            encoded_checkpoint,
            key,
            self._temp_dir
        )
        return checkpoint_path

    def train_and_eval(self, train_input_fn, eval_input_fn):
        """Run distributed train and eval on Mask RCNN model."""

        # self._save_config()
        output_dir = os.path.join(self._runtime_config.model_dir, 'eval')
        tf.io.gfile.makedirs(output_dir)

        train_run_config = self.build_strategy_configuration('train')
        train_params = self.build_model_parameters('train')
        train_estimator = self.build_mask_rcnn_estimator(train_params, train_run_config, 'train')

        eval_estimator = None
        eval_results = None

        num_cycles = math.ceil(
            self._runtime_config.total_steps / self._runtime_config.num_steps_per_eval)
        curr_cycle = self.curr_step // self._runtime_config.num_steps_per_eval
        training_hooks = self.get_training_hooks(
            mode="train",
            model_dir=self._runtime_config.model_dir,
            checkpoint_path=self._runtime_config.checkpoint,
            skip_checkpoint_variables=self._runtime_config.skip_checkpoint_variables
        )
        from contextlib import suppress

        def _profiler_context_manager(*args, **kwargs):
            """profiler manager."""
            return suppress()

        for cycle in range(1, num_cycles + 1):

            if (not MPI_is_distributed() or MPI_rank() == 0) and cycle > curr_cycle:

                print()  # Visual Spacing
                logging.info("=================================")
                logging.info('     Start training cycle %02d' % cycle)
                logging.info("=================================\n")

            max_cycle_step = min(int(cycle * self._runtime_config.num_steps_per_eval),
                                 self._runtime_config.total_steps)

            PROFILER_ENABLED = False

            if (not MPI_is_distributed() or MPI_rank() == 0) and PROFILER_ENABLED:
                profiler_context_manager = tf.contrib.tfprof.ProfileContext

            else:
                # No-Op context manager
                profiler_context_manager = _profiler_context_manager

            with profiler_context_manager('/workspace/profiling/',
                                          trace_steps=range(100, 200, 3),
                                          dump_steps=[200]) as pctx:
                if (not MPI_is_distributed() or MPI_rank() == 0) and PROFILER_ENABLED:
                    opts = tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory()
                    pctx.add_auto_profiling('op', opts, [150, 200])

                train_estimator.train(
                    input_fn=train_input_fn,
                    max_steps=max_cycle_step,
                    hooks=training_hooks,
                )

            if (not MPI_is_distributed() or MPI_rank() == 0) and cycle > curr_cycle:

                print()  # Visual Spacing
                logging.info("=================================")
                logging.info('    Start evaluation cycle %02d' % cycle)
                logging.info("=================================\n")

                if eval_estimator is None:
                    eval_run_config = self.build_strategy_configuration('eval')
                    eval_params = self.build_model_parameters('eval')
                    eval_estimator = self.build_mask_rcnn_estimator(eval_params,
                                                                    eval_run_config, 'eval')

                ckpt_path = self.get_latest_checkpoint(self._runtime_config.model_dir,
                                                       self._runtime_config.key)

                eval_results, predictions = evaluation.evaluate(
                    eval_estimator,
                    eval_input_fn,
                    self._runtime_config.eval_samples,
                    self._runtime_config.eval_batch_size,
                    include_mask=self._runtime_config.include_mask,
                    validation_json_file=self._runtime_config.val_json_file,
                    report_frequency=self._runtime_config.report_frequency,
                    checkpoint_path=ckpt_path
                )

                self._write_summary(output_dir, eval_results, predictions, max_cycle_step)

            if MPI_is_distributed():
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()  # Waiting for all MPI processes to sync

        return eval_results

    def eval(self, eval_input_fn):
        """Run distributed eval on Mask RCNN model."""

        output_dir = os.path.join(os.path.dirname(self._runtime_config.model_path), 'eval')
        tf.io.gfile.makedirs(output_dir)

        # Summary writer writes out eval metrics.
        run_config = self.build_strategy_configuration('eval')
        eval_params = self.build_model_parameters('eval')
        eval_estimator = self.build_mask_rcnn_estimator(eval_params, run_config, 'eval')

        logging.info('Starting to evaluate.')

        ckpt_path = self.extract_ckpt(self._runtime_config.model_path, self._runtime_config.key)
        current_step = eval(ckpt_path.split("-")[-1])

        eval_results, predictions = evaluation.evaluate(
            eval_estimator,
            eval_input_fn,
            self._runtime_config.eval_samples,
            self._runtime_config.eval_batch_size,
            include_mask=self._runtime_config.include_mask,
            validation_json_file=self._runtime_config.val_json_file,
            report_frequency=self._runtime_config.report_frequency,
            checkpoint_path=ckpt_path
        )

        self._write_summary(output_dir, eval_results, predictions, current_step)

        if current_step >= self._runtime_config.total_steps:
            logging.info('Evaluation finished after training step %d' % current_step)

        return eval_results

    def infer(self, infer_input_fn):
        """Run inference on Mask RCNN model."""
        # Summary writer writes out eval metrics.
        run_config = self.build_strategy_configuration('eval')
        infer_params = self.build_model_parameters('eval')
        infer_estimator = self.build_mask_rcnn_estimator(infer_params, run_config, 'eval')

        logging.info('Running inference...')
        ckpt_path = self.extract_ckpt(self._runtime_config.model_path, self._runtime_config.key)

        evaluation.infer(
            infer_estimator,
            infer_input_fn,
            self._runtime_config.num_infer_samples,
            self._runtime_config.batch_size,
            self._runtime_config.output_dir,
            self._runtime_config.threshold,
            self._runtime_config.label_dict,
            self._runtime_config.include_mask,
            checkpoint_path=ckpt_path
        )


class EstimatorExecuter(BaseExecuter):
    """Interface that runs Mask RCNN model using TPUEstimator."""

    def __init__(self, runtime_config, model_fn):
        """Initialize."""
        super(EstimatorExecuter, self).__init__(runtime_config, model_fn)

        if MPI_is_distributed():
            os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
            os.environ['HOROVOD_NUM_NCCL_STREAMS'] = '1'
            # os.environ['HOROVOD_AUTOTUNE'] = '2'
            hvd.init()
            logging.info("Horovod successfully initialized ...")

        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '1' if not MPI_is_distributed() else str(hvd.size())

        os.environ['TF_SYNC_ON_FINISH'] = '0'

    def build_strategy_configuration(self, mode):
        """Retrieves model configuration for running TF Estimator."""

        run_config = tf.estimator.RunConfig(
            tf_random_seed=(
                self._runtime_config.seed
                if not MPI_is_distributed() or self._runtime_config.seed is None else
                self._runtime_config.seed + MPI_rank()
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
                use_xla=self._runtime_config.use_xla,
                use_amp=self._runtime_config.use_amp,
                use_tf_distributed=False,
                # TODO: Remove when XLA at inference fixed
                allow_xla_at_inference=self._runtime_config.allow_xla_at_inference
            ),
            protocol=None,
            device_fn=None,
            train_distribute=None,
            eval_distribute=None,
            experimental_distribute=None
        )
        return run_config
