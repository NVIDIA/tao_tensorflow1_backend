"""YOLO v3 class to build the model and pipelines."""
from contextlib import contextmanager
from math import ceil
from multiprocessing import cpu_count
import os
import shutil
import tempfile

import keras
from keras.backend import set_learning_phase
from keras.callbacks import TerminateOnNaN
from keras.layers import Input
from keras.models import Model
import numpy as np
import six
import tensorflow as tf

from nvidia_tao_tf1.cv.common.callbacks.detection_metric_callback import DetectionMetricCallback
from nvidia_tao_tf1.cv.common.callbacks.enc_model_saver_callback import KerasModelSaver
from nvidia_tao_tf1.cv.common.callbacks.loggers import TAOStatusLogger
from nvidia_tao_tf1.cv.common.evaluator.ap_evaluator import APEvaluator
from nvidia_tao_tf1.cv.common.utils import (
    build_optimizer_from_config,
    build_regularizer_from_config,
    CUSTOM_OBJS,
    TensorBoard
)
from nvidia_tao_tf1.cv.common.utils import OneIndexedCSVLogger as CSVLogger
from nvidia_tao_tf1.cv.common.utils import SoftStartAnnealingLearningRateScheduler as LRS
from nvidia_tao_tf1.cv.yolo_v3.architecture.yolo_arch import YOLO
from nvidia_tao_tf1.cv.yolo_v3.builders import eval_builder
from nvidia_tao_tf1.cv.yolo_v3.builders.model_builder import _load_pretrain_weights
from nvidia_tao_tf1.cv.yolo_v3.data_loader.data_loader import YOLOv3DataPipe
from nvidia_tao_tf1.cv.yolo_v3.data_loader.generate_shape_tensors import gen_random_shape_tensors
from nvidia_tao_tf1.cv.yolo_v3.losses.yolo_loss import YOLOv3Loss
from nvidia_tao_tf1.cv.yolo_v3.metric.yolov3_metric_callback import YOLOv3MetricCallback
from nvidia_tao_tf1.cv.yolo_v3.utils import model_io
from nvidia_tao_tf1.cv.yolo_v3.utils.model_io import get_model_with_input
from nvidia_tao_tf1.cv.yolo_v3.utils.spec_loader import validation_labels_format
from nvidia_tao_tf1.cv.yolo_v4.utils.fit_generator import fit_generator


@contextmanager
def patch_freeze_bn(freeze_bn):
    """context for patching BN to freeze it during model creation."""

    def compose_call(prev_call_method):
        def call(self, inputs, training=False):
            return prev_call_method(self, inputs, training)

        return call
    prev_batchnorm_call = keras.layers.normalization.BatchNormalization.call
    if freeze_bn:
        keras.layers.normalization.BatchNormalization.call = compose_call(
            prev_batchnorm_call
        )
    yield
    if freeze_bn:
        keras.layers.normalization.BatchNormalization.call = prev_batchnorm_call


class YOLOv3Model(object):
    """YOLO v3 model."""

    def __init__(self, spec, key):
        """Initialize."""
        self.spec = spec
        self.yolov3_config = spec.yolov3_config
        self.key = key
        # dataset classes
        self.class_mapping = spec.dataset_config.target_class_mapping
        self.classes = sorted({str(x).lower() for x in self.class_mapping.values()})
        self.n_classes = len(self.classes)
        # model architecture
        self.arch = spec.yolov3_config.arch
        self.arch_name = self.arch
        if self.arch_name in ['resnet', 'darknet', 'vgg']:
            # append nlayers into meta_arch_name
            self.arch_name += str(spec.yolov3_config.nlayers)
        self.nlayers = spec.yolov3_config.nlayers
        self.freeze_blocks = spec.yolov3_config.freeze_blocks
        self.freeze_bn = spec.yolov3_config.freeze_bn
        self.arch_conv_blocks = spec.yolov3_config.arch_conv_blocks
        self.force_relu = spec.yolov3_config.force_relu
        self.qat = spec.training_config.enable_qat
        # NMS config
        self.nms_confidence_thresh = spec.nms_config.confidence_threshold
        self.nms_iou_threshold = spec.nms_config.clustering_iou_threshold
        self.nms_top_k = spec.nms_config.top_k
        # If using TFRecords, force NMS on CPU
        self.nms_on_cpu = False
        if self.train_labels_format == "tfrecords" or self.val_labels_format == "tfrecords":
            self.nms_on_cpu = True
        # evaluation params
        self.ap_mode = spec.eval_config.average_precision_mode
        matching_iou = spec.eval_config.matching_iou_threshold
        self.matching_iou = matching_iou if matching_iou > 0 else 0.5
        self.ap_mode_dict = {0: "sample", 1: "integrate"}
        self.average_precision_mode = self.ap_mode_dict[self.ap_mode]
        # training
        self.training_config = spec.training_config
        self.use_mp = spec.training_config.use_multiprocessing
        self.n_workers = spec.training_config.n_workers or (cpu_count()-1)
        self.max_queue_size = spec.training_config.max_queue_size or 20
        self.num_epochs = spec.training_config.num_epochs
        self.bs = spec.training_config.batch_size_per_gpu
        self.lrconfig = spec.training_config.learning_rate.soft_start_annealing_schedule
        self.ckpt_interval = spec.training_config.checkpoint_interval
        self.augmentation_config = spec.augmentation_config
        self.image_channels = int(self.augmentation_config.output_channel)
        self.image_width = int(self.augmentation_config.output_width)
        self.image_height = int(self.augmentation_config.output_height)
        self.shape_period = int(self.augmentation_config.randomize_input_shape_period)
        self.load_type = spec.training_config.WhichOneof('load_model')
        self.hmin_ratio = 0.6
        self.hmax_ratio = 1.5
        self.wmin_ratio = 0.6
        self.wmax_ratio = 1.5
        self.build_regularizer()
        self.generate_random_shape()
        self.h_tensor_val = tf.constant(
            self.image_height,
            dtype=tf.int32
        )
        self.w_tensor_val = tf.constant(
            self.image_width,
            dtype=tf.int32
        )
        self.get_val_fmap_stride()
        self.parse_init_epoch()
        self.callbacks = []
        self.losses = None
        self.metrics = None
        self.optimizer = None
        self.target_tensors = None
        self.tb_callback = None

    def generate_random_shape(self):
        """generate random shape for multi-scale training."""
        if self.shape_period > 0:
            self.h_tensor, self.w_tensor = gen_random_shape_tensors(
                self.shape_period,
                int(self.image_height * self.hmin_ratio),
                int(self.image_height * self.hmax_ratio),
                int(self.image_width * self.wmin_ratio),
                int(self.image_width * self.wmax_ratio)
            )
        else:
            self.h_tensor = tf.constant(
                self.image_height,
                dtype=tf.int32
            )
            self.w_tensor = tf.constant(
                self.image_width,
                dtype=tf.int32
            )

    def parse_init_epoch(self):
        """Parse initial epoch."""
        if self.load_type == 'resume_model_path':
            try:
                epoch = int(self.training_config.resume_model_path.split('.')[-2].split('_')[-1])
            except Exception:
                raise ValueError("Cannot parse the checkpoint path. Did you rename it?")
        else:
            epoch = 0
        self.init_epoch = epoch

    @property
    def train_labels_format(self):
        """The format of the labels of training set."""
        if self.spec.dataset_config.data_sources[0].WhichOneof("labels_format") == \
           "tfrecords_path":
            return "tfrecords"
        return "keras_sequence"

    @property
    def val_labels_format(self):
        """The format of the labels of validation set."""
        return validation_labels_format(self.spec)

    def build_regularizer(self):
        """build regularizer."""
        self.regularizer = build_regularizer_from_config(
            self.training_config.regularizer
        )

    def build_optimizer(self, hvd):
        """build optimizer."""
        optim = build_optimizer_from_config(
            self.training_config.optimizer
        )
        self.set_optimizer(optim, hvd)

    def eval_str(self, s):
        """If s is a string, return the eval results. Else return itself."""
        if isinstance(s, six.string_types):
            if len(s) > 0:
                return eval(s)
            return None
        return s

    @property
    def big_anchor_shape(self):
        """big anchor shape."""
        big_anchor = self.eval_str(self.yolov3_config.big_anchor_shape)
        assert len(big_anchor) > 0, "big_anchor_shape in spec cannot be empty"
        return big_anchor

    @property
    def mid_anchor_shape(self):
        """middle anchor shape."""
        mid_anchor = self.eval_str(self.yolov3_config.mid_anchor_shape)
        assert len(mid_anchor) > 0, "mid_anchor_shape in spec cannot be empty"
        return mid_anchor

    @property
    def small_anchor_shape(self):
        """small anchor shape."""
        small_anchor = self.eval_str(self.yolov3_config.small_anchor_shape)
        assert len(small_anchor) > 0, "small_anchor_shape in spec cannot be empty"
        return small_anchor

    def anchor_to_relative(self, x):
        """convert absolute anchors to relative anchors."""
        return (np.array(x, dtype=np.float).reshape(-1, 2) / np.array(
            [self.image_width, self.image_height]).reshape(1, 2)).tolist()

    @property
    def all_anchors(self):
        """all absolute anchors."""
        return [self.big_anchor_shape, self.mid_anchor_shape, self.small_anchor_shape]

    @property
    def all_relative_anchors(self):
        """all relative anchors."""
        return [self.anchor_to_relative(x) for x in self.all_anchors]

    def build_keras_model(self, input_image=None, input_shape=None, val=False):
        """build a keras model from scratch."""
        model_input = Input(
            shape=input_shape or (self.image_channels, None, None),
            tensor=input_image,
            name="Input"
        )
        yolo_model = YOLO(
            model_input,
            self.arch,
            self.nlayers,
            num_classes=self.n_classes,
            kernel_regularizer=self.regularizer,
            anchors=self.all_relative_anchors,
            freeze_blocks=self.freeze_blocks,
            freeze_bn=self.freeze_bn,
            arch_conv_blocks=self.arch_conv_blocks,
            qat=self.qat,
            force_relu=self.force_relu
        )
        if val:
            # if it is a validation model, return it directly
            return yolo_model
        # rename it
        self.keras_model = Model(
            inputs=model_input,
            outputs=yolo_model.outputs,
            name='yolo_' + self.arch
        )
        self.inputs = self.keras_model.inputs
        self.outputs = self.keras_model.outputs
        return None

    def load_pretrained_model(self, model_path):
        """load pretrained model's weights."""
        pretrained_model = model_io.load_model(
            model_path,
            self.spec,
            key=self.key
        )
        _load_pretrain_weights(pretrained_model, self.keras_model)

    def override_regularizer(self, train_model):
        """override regularizer."""
        model_config = train_model.get_config()
        for layer, layer_config in zip(train_model.layers, model_config['layers']):
            if hasattr(layer, 'kernel_regularizer'):
                layer_config['config']['kernel_regularizer'] = self.regularizer
        reg_model = Model.from_config(
            model_config,
            custom_objects=CUSTOM_OBJS
        )
        reg_model.set_weights(train_model.get_weights())
        return reg_model

    def apply_model_to_new_inputs(self, model, tensor, input_shape):
        """Apply model to new inputs."""
        input_layer = keras.layers.InputLayer(
            input_shape=input_shape,
            input_tensor=tensor,
            name="Input",
        )
        _, temp_model_path = tempfile.mkstemp()
        os.remove(temp_model_path)
        model.save(temp_model_path)
        with patch_freeze_bn(self.freeze_bn):
            new_model = get_model_with_input(temp_model_path, input_layer)
        os.remove(temp_model_path)
        return new_model

    def load_pruned_model(self, pruned_model_path, input_tensor, input_shape):
        """load pruned model."""
        pruned_model = model_io.load_model(
            pruned_model_path,
            self.spec,
            key=self.key,
            input_shape=input_shape
        )
        pruned_model = self.override_regularizer(
            pruned_model
        )
        if input_tensor is not None:
            self.keras_model = self.apply_model_to_new_inputs(
                pruned_model,
                input_tensor,
                input_shape
            )
        else:
            self.keras_model = pruned_model
        self.inputs = self.keras_model.inputs
        self.outputs = self.keras_model.outputs

    def set_optimizer(self, opt, hvd):
        '''setup optimizer.'''
        if self.optimizer is not None:
            return
        self.optimizer = hvd.DistributedOptimizer(opt)

    def resume_model(self, checkpoint_path, input_tensor, input_shape, hvd):
        '''resume model from checkpoints and continue to train.'''
        resumed_model = model_io.load_model(
            checkpoint_path,
            self.spec,
            key=self.key,
            input_shape=input_shape
        )
        optimizer = resumed_model.optimizer
        if input_tensor is not None:
            resumed_model = self.apply_model_to_new_inputs(
                resumed_model,
                input_tensor,
                input_shape
            )
        self.keras_model = resumed_model
        self.inputs = self.keras_model.inputs
        self.outputs = self.keras_model.outputs
        self.set_optimizer(optimizer, hvd)

    def set_target_tensors(self, encoded_labels):
        """set target tensors."""
        if self.target_tensors is not None:
            return
        self.target_tensors = [encoded_labels]

    def build_losses(self):
        """build loss."""
        if self.losses is not None:
            return
        yololoss = YOLOv3Loss(
            self.spec.yolov3_config.loss_loc_weight,
            self.spec.yolov3_config.loss_neg_obj_weights,
            self.spec.yolov3_config.loss_class_weights,
            self.spec.yolov3_config.matching_neutral_box_iou
        )
        self.losses = [yololoss.compute_loss]

    def build_hvd_callbacks(self, hvd):
        '''setup horovod callbacks.'''
        self.callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        self.callbacks.append(hvd.callbacks.MetricAverageCallback())
        self.callbacks.append(TerminateOnNaN())

    def build_lr_scheduler(self, train_dataset, hvd):
        """build LR scheduler."""
        init_epoch = self.init_epoch
        if type(train_dataset) == YOLOv3DataPipe:
            total_num = train_dataset.num_samples
        else:
            total_num = train_dataset.n_samples
        iters_per_epoch = int(ceil(total_num / self.bs / hvd.size()))
        max_iterations = self.num_epochs * iters_per_epoch
        lr_scheduler = LRS(
            base_lr=self.lrconfig.max_learning_rate * hvd.size(),
            min_lr_ratio=self.lrconfig.min_learning_rate / self.lrconfig.max_learning_rate,
            soft_start=self.lrconfig.soft_start,
            annealing_start=self.lrconfig.annealing,
            max_iterations=max_iterations
        )
        init_step = init_epoch * iters_per_epoch
        lr_scheduler.reset(init_step)
        self.callbacks.append(lr_scheduler)
        self.iters_per_epoch = iters_per_epoch

    def build_checkpointer(self, ckpt_path, verbose):
        """build checkpointer."""
        model_checkpointer = KerasModelSaver(
            ckpt_path,
            self.key,
            self.ckpt_interval,
            last_epoch=self.num_epochs,
            verbose=verbose
        )
        self.callbacks.append(model_checkpointer)

    def build_csvlogger(self, csv_path):
        """build CSV logger."""
        csv_logger = CSVLogger(
            filename=csv_path,
            separator=',',
            append=False
        )
        self.callbacks.append(csv_logger)

    def build_training_model(self, hvd):
        """build the training model in various cases."""
        if type(self.train_dataset) == YOLOv3DataPipe:
            input_image = self.train_dataset.images
        else:
            input_image = None
        if self.load_type == "resume_model_path":
            self.resume_model(
                self.training_config.resume_model_path,
                input_image,
                (self.image_channels, None, None),
                hvd
            )
        elif self.load_type == "pruned_model_path":
            self.load_pruned_model(
                self.training_config.pruned_model_path,
                input_image,
                (self.image_channels, None, None)
            )
        else:
            self.build_keras_model(
                input_image
            )
            if self.training_config.pretrain_model_path:
                self.load_pretrained_model(
                    self.training_config.pretrain_model_path
                )
        # get predictor sizes for later use
        predictor_names = [
            'conv_big_object',
            'conv_mid_object',
            'conv_sm_object'
        ]
        predictor_layers = [
            self.keras_model.get_layer(n) for n in predictor_names
        ]
        self.predictor_sizes = [tf.shape(l.output)[2:4] for l in predictor_layers]

    def build_validation_model(self):
        """build validation model."""
        # set eval phase at first
        assert self.keras_model is not None, (
            """Training model has to be built before validation model."""
        )
        set_learning_phase(0)
        input_shape = (self.image_channels, self.image_height, self.image_width)
        input_layer = keras.layers.InputLayer(
            input_shape=input_shape,
            input_tensor=None,
            name="Input",
        )
        _, temp_model_path = tempfile.mkstemp()
        os.remove(temp_model_path)
        self.keras_model.save(temp_model_path)
        with patch_freeze_bn(self.freeze_bn):
            val_model = get_model_with_input(temp_model_path, input_layer)
        os.remove(temp_model_path)
        self._val_model = val_model
        # setup validation model predictor sizes for later use
        predictor_names = [
            'conv_big_object',
            'conv_mid_object',
            'conv_sm_object'
        ]
        predictor_layers = [
            self._val_model.get_layer(n) for n in predictor_names
        ]
        self.val_predictor_sizes = [l.output_shape[2:] for l in predictor_layers]
        self.val_fmap_stride = [
            (self.image_height // x[0], self.image_width // x[1]) for x in self.val_predictor_sizes
        ]
        # restore learning phase to 1
        set_learning_phase(1)
        self.val_model = eval_builder.build(
            val_model,
            confidence_thresh=self.nms_confidence_thresh,
            iou_threshold=self.nms_iou_threshold,
            top_k=self.nms_top_k,
            include_encoded_head=True,
            nms_on_cpu=self.nms_on_cpu
        )

    def get_val_fmap_stride(self):
        """build a dummy validation model to get val_fmap_stride."""
        # set eval phase at first
        set_learning_phase(0)
        # it doesn't matter whether the train model is pruned or not,
        # since we just care about the height/width of the predictor
        # feature maps. Channel number is irrelevant.
        val_model = self.build_keras_model(
            input_shape=(self.image_channels, self.image_height, self.image_width),
            val=True
        )
        # restore learning phase to 1
        set_learning_phase(1)
        # setup validation model predictor sizes for later use
        predictor_names = [
            'conv_big_object',
            'conv_mid_object',
            'conv_sm_object'
        ]
        predictor_layers = [
            val_model.get_layer(n) for n in predictor_names
        ]
        val_predictor_sizes = [l.output_shape[2:4] for l in predictor_layers]
        fmap_stride = [
            (self.image_height // x[0], self.image_width // x[1]) for x in val_predictor_sizes
        ]
        self.val_fmap_stride = fmap_stride

    def build_ap_evaluator(self):
        """build_ap_evaluator."""
        self.ap_evaluator = APEvaluator(
            self.n_classes,
            conf_thres=self.nms_confidence_thresh,
            matching_iou_threshold=self.matching_iou,
            average_precision_mode=self.average_precision_mode
        )

    def build_loss_ops(self):
        """build loss ops."""
        n_box, n_attr = self._val_model.layers[-1].output_shape[1:]
        op_pred = tf.placeholder(tf.float32, shape=(None, n_box, n_attr))
        op_true = tf.placeholder(tf.float32, shape=(None, n_box, n_attr - 6))
        self.loss_ops = [op_true, op_pred, self.losses[0](op_true, op_pred)]

    def build_validation_callback(
        self,
        val_dataset,
        verbose=False
    ):
        """Build validation model."""
        # build validation model
        self.build_loss_ops()
        self.build_ap_evaluator()
        # build validation callback
        if type(val_dataset) == YOLOv3DataPipe:
            eval_callback = YOLOv3MetricCallback(
                ap_evaluator=self.ap_evaluator,
                built_eval_model=self.val_model,
                generator=val_dataset.generator(),
                classes=self.classes,
                n_batches=val_dataset.n_batches,
                loss_ops=self.loss_ops,
                eval_model=self._val_model,
                metric_interval=self.ckpt_interval,
                last_epoch=self.num_epochs,
                verbose=verbose
            )
        else:
            eval_callback = DetectionMetricCallback(
                ap_evaluator=self.ap_evaluator,
                built_eval_model=self.val_model,
                eval_sequence=val_dataset,
                loss_ops=self.loss_ops,
                eval_model=self._val_model,
                metric_interval=self.ckpt_interval,
                last_epoch=self.num_epochs,
                verbose=verbose
            )
        return self.callbacks.append(eval_callback)

    def build_savers(self, results_dir, verbose):
        """build several savers."""
        if not os.path.exists(os.path.join(results_dir, 'weights')):
            os.mkdir(os.path.join(results_dir, 'weights'))
        ckpt_path = str(os.path.join(
                results_dir,
                'weights',
                'yolov3_' + self.arch_name + '_epoch_{epoch:03d}.hdf5'
            )
        )
        # checkpointer
        self.build_checkpointer(ckpt_path, verbose)
        # output label file
        with open(os.path.join(results_dir, 'model_output_labels.txt'), 'w') as f:
            f.write('\n'.join(self.classes))
        csv_path = os.path.join(results_dir, 'yolov3_training_log_' + self.arch_name + '.csv')
        # CSV logger
        self.build_csvlogger(csv_path)

    def build_tensorboard_callback(self, output_dir):
        """Build TensorBoard callback for visualization."""
        tb_path = os.path.join(
            output_dir,
            "logs"
        )
        if os.path.exists(tb_path) and os.path.isdir(tb_path):
            shutil.rmtree(tb_path)
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        tb_callback = TensorBoard(
            log_dir=tb_path,
            write_graph=False,
            weight_hist=False
        )
        self.tb_callback = tb_callback
        self.callbacks.append(tb_callback)

    def build_status_logging_callback(self, results_dir, num_epochs, is_master):
        """Build status logging for TAO API."""
        status_logger = TAOStatusLogger(
            results_dir,
            append=True,
            num_epochs=num_epochs,
            is_master=is_master,
        )
        self.callbacks.append(status_logger)

    def compile(self):
        '''compile the keras model.'''
        self.keras_model.compile(
            optimizer=self.optimizer,
            loss=self.losses,
            target_tensors=self.target_tensors
        )

    def summary(self):
        """print keras model summary."""
        self.keras_model.summary()

    def train(self, verbose=1):
        """training."""
        if type(self.train_dataset) == YOLOv3DataPipe:
            self.keras_model.fit(
                epochs=self.num_epochs,
                steps_per_epoch=self.iters_per_epoch,
                callbacks=self.callbacks,
                initial_epoch=self.init_epoch,
                verbose=verbose
            )
        else:
            # Use the patched fit_generator
            # TensorBoard image summary only supports 8-bit images
            if (self.tb_callback is not None) and (self.image_depth == 8):
                writer = self.tb_callback.writer
            else:
                writer = None
            default_img_mean = (103.939, 116.779, 123.68)
            fit_generator(
                self.keras_model,
                writer,
                img_means=self.augmentation_config.image_mean or default_img_mean,
                max_image_num=self.spec.training_config.visualizer.num_images,
                steps_per_epoch=self.iters_per_epoch,
                generator=self.train_dataset,
                epochs=self.num_epochs,
                callbacks=self.callbacks,
                initial_epoch=self.init_epoch,
                workers=self.n_workers,
                max_queue_size=self.max_queue_size,
                verbose=verbose,
                use_multiprocessing=self.use_mp,
                shuffle=False
            )
