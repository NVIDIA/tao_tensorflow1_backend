"""Utils to build the model, data loader and entire pipeline."""

from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.common.utils import build_class_weights
from nvidia_tao_tf1.cv.common.visualizer.tensorboard_visualizer import TensorBoardVisualizer
from nvidia_tao_tf1.cv.yolo_v4.dataio.data_sequence import YOLOv4DataSequence
from nvidia_tao_tf1.cv.yolo_v4.dataio.input_encoder import (
    YOLOv4InputEncoder,
    YOLOv4InputEncoderTensor
)
from nvidia_tao_tf1.cv.yolo_v4.dataio.tf_data_pipe import YOLOv4TFDataPipe
from nvidia_tao_tf1.cv.yolo_v4.models.yolov4_model import YOLOv4Model


def build_training_pipeline(spec, results_dir, key, hvd, sess, verbose):
    """Build the training pipeline."""
    # Define visualizer
    visualizer = TensorBoardVisualizer()
    visualizer.build_from_config(
        spec.training_config.visualizer
    )
    visualizer_config = spec.training_config.visualizer
    is_master = hvd.rank() == 0
    if is_master and visualizer_config.HasField("clearml_config"):
        clearml_config = visualizer_config.clearml_config
        get_clearml_task(clearml_config, "yolo_v4")
    if is_master and visualizer_config.HasField("wandb_config"):
        wandb_config = visualizer_config.wandb_config
        wandb_logged_in = check_wandb_logged_in()
        wandb_name = f"{wandb_config.name}" if wandb_config.name else \
            "yolov4_training"
        initialize_wandb(
            project=wandb_config.project if wandb_config.project else None,
            entity=wandb_config.entity if wandb_config.entity else None,
            notes=wandb_config.notes if wandb_config.notes else None,
            tags=wandb_config.tags if wandb_config.tags else None,
            sync_tensorboard=True,
            save_code=False,
            results_dir=results_dir,
            wandb_logged_in=wandb_logged_in,
            name=wandb_name
        )
    # instantiate the model
    yolov4 = YOLOv4Model(
        spec,
        key
    )
    cls_weights = build_class_weights(spec)
    train_encoder = YOLOv4InputEncoder(
        yolov4.n_classes,
        spec.yolov4_config.box_matching_iou,
        yolov4.val_fmap_stride,
        yolov4.all_relative_anchors,
        class_weights=cls_weights
    )

    def eval_encode_fn(output_img_size, gt_label):
        return (train_encoder(output_img_size, gt_label), gt_label)

    if yolov4.train_labels_format == "tfrecords":
        # tfrecord data loader
        train_dataset = YOLOv4TFDataPipe(
            spec,
            label_encoder=None,
            training=True,
            h_tensor=yolov4.h_tensor,
            w_tensor=yolov4.w_tensor,
            visualizer=visualizer,
            rank=hvd.rank()
        )
        yolov4.train_dataset = train_dataset
        # build the training model
        yolov4.build_training_model(hvd)
        # setup target tensors
        yolo_input_encoder = \
            YOLOv4InputEncoderTensor(
                img_height=yolov4.h_tensor,
                img_width=yolov4.w_tensor,
                n_classes=yolov4.n_classes,
                matching_box_iou_thres=spec.yolov4_config.box_matching_iou,
                feature_map_size=yolov4.predictor_sizes,
                anchors=yolov4.all_relative_anchors,
                class_weights=cls_weights
            )
        train_dataset.set_encoder(yolo_input_encoder)
        yolov4.set_target_tensors(train_dataset.encoded_labels)
    else:
        # keras sequence data loader
        train_sequence = YOLOv4DataSequence(
            spec.dataset_config,
            spec.augmentation_config,
            spec.training_config.batch_size_per_gpu,
            is_training=True,
            encode_fn=train_encoder,
            output_raw_label=spec.training_config.visualizer.enabled
        )
        yolov4.train_dataset = train_sequence
        # build the training model
        yolov4.build_training_model(hvd)
    # Visualize model weights histogram
    if hvd.rank() == 0 and spec.training_config.visualizer.enabled:
        visualizer.keras_model_weight_histogram(yolov4.keras_model)
    # setup optimizer, if any
    yolov4.build_optimizer(hvd)
    # buld loss functions
    yolov4.build_losses()
    # build callbacks
    yolov4.build_hvd_callbacks(hvd)
    # build learning rate scheduler
    yolov4.build_lr_scheduler(yolov4.train_dataset, hvd)
    # build validation callback
    if yolov4.val_labels_format == "tfrecords":
        val_dataset = YOLOv4TFDataPipe(
            spec,
            training=False,
            sess=sess,
            h_tensor=yolov4.h_tensor_val,
            w_tensor=yolov4.w_tensor_val
        )
        yolov4.val_dataset = val_dataset
        yolov4.build_validation_model()
        val_input_encoder = \
            YOLOv4InputEncoderTensor(
                img_height=yolov4.h_tensor_val,
                img_width=yolov4.w_tensor_val,
                n_classes=yolov4.n_classes,
                matching_box_iou_thres=spec.yolov4_config.box_matching_iou,
                feature_map_size=yolov4.val_predictor_sizes,
                anchors=yolov4.all_relative_anchors
            )
        val_dataset.set_encoder(val_input_encoder)
        yolov4.build_validation_callback(
            val_dataset,
            verbose=verbose
        )
    else:
        yolov4.build_validation_model()
        eval_sequence = YOLOv4DataSequence(
            spec.dataset_config,
            spec.augmentation_config,
            spec.eval_config.batch_size,
            is_training=False,
            encode_fn=eval_encode_fn
        )
        yolov4.val_dataset = eval_sequence
        yolov4.build_validation_callback(
            eval_sequence,
            verbose=verbose
        )
    # build checkpointer
    yolov4.compile()
    if spec.class_weighting_config.enable_auto:
        yolov4.build_auto_class_weighting_callback(yolov4.train_dataset)
    if hvd.rank() == 0:
        yolov4.build_savers(results_dir, verbose)
        if spec.training_config.visualizer.enabled:
            yolov4.build_tensorboard_callback(results_dir)
        yolov4.build_status_logging_callback(results_dir, yolov4.num_epochs, True)
    if spec.training_config.model_ema:
        yolov4.build_model_ema_callback()
    yolov4.build_early_stopping_callback()
    return yolov4
