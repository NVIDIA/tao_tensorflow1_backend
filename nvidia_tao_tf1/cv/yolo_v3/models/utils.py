"""Utils to build the model, data loader and entire pipeline."""

from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.common.visualizer.tensorboard_visualizer import TensorBoardVisualizer
from nvidia_tao_tf1.cv.yolo_v3.data_loader.data_loader import YOLOv3DataPipe
from nvidia_tao_tf1.cv.yolo_v3.dataio.data_sequence import YOLOv3DataSequence
from nvidia_tao_tf1.cv.yolo_v3.dataio.input_encoder import (
    YOLOv3InputEncoder,
    YOLOv3InputEncoderTensor
)
from nvidia_tao_tf1.cv.yolo_v3.models.yolov3_model import YOLOv3Model


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
        get_clearml_task(clearml_config, "yolo_v3")
    if is_master and visualizer_config.HasField("wandb_config"):
        wandb_config = visualizer_config.wandb_config
        wandb_logged_in = check_wandb_logged_in()
        wandb_name = f"{wandb_config.name}" if wandb_config.name else \
            "yolov3_training"
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
    yolov3 = YOLOv3Model(
        spec,
        key
    )
    train_encoder = YOLOv3InputEncoder(
        yolov3.n_classes,
        yolov3.val_fmap_stride,
        yolov3.all_relative_anchors
    )

    def eval_encode_fn(output_img_size, gt_label):
        return (train_encoder(output_img_size, gt_label), gt_label)
    if yolov3.train_labels_format == "tfrecords":
        # tfrecord data loader
        train_dataset = YOLOv3DataPipe(
            spec,
            label_encoder=None,
            training=True,
            h_tensor=yolov3.h_tensor,
            w_tensor=yolov3.w_tensor,
            visualizer=visualizer,
            rank=hvd.rank()
        )
        yolov3.train_dataset = train_dataset
        # build the training model
        yolov3.build_training_model(hvd)
        # setup target tensors
        yolo_input_encoder = \
            YOLOv3InputEncoderTensor(
                img_height=yolov3.h_tensor,
                img_width=yolov3.w_tensor,
                n_classes=yolov3.n_classes,
                feature_map_size=yolov3.predictor_sizes,
                anchors=yolov3.all_relative_anchors
            )
        train_dataset.set_encoder(yolo_input_encoder)
        yolov3.set_target_tensors(train_dataset.encoded_labels)
    else:
        # keras sequence data loader
        train_sequence = YOLOv3DataSequence(
            spec.dataset_config,
            spec.augmentation_config,
            spec.training_config.batch_size_per_gpu,
            is_training=True,
            encode_fn=train_encoder,
            output_raw_label=spec.training_config.visualizer.enabled
        )
        yolov3.train_dataset = train_sequence
        # build the training model
        yolov3.build_training_model(hvd)
    # Visualize model weights histogram
    if hvd.rank() == 0 and spec.training_config.visualizer.enabled:
        visualizer.keras_model_weight_histogram(yolov3.keras_model)
    # setup optimizer, if any
    yolov3.build_optimizer(hvd)
    # buld loss functions
    yolov3.build_losses()
    # build callbacks
    yolov3.build_hvd_callbacks(hvd)
    # build learning rate scheduler
    yolov3.build_lr_scheduler(yolov3.train_dataset, hvd)
    # build validation callback
    if yolov3.val_labels_format == "tfrecords":
        val_dataset = YOLOv3DataPipe(
            spec,
            training=False,
            sess=sess,
            h_tensor=yolov3.h_tensor_val,
            w_tensor=yolov3.w_tensor_val
        )
        yolov3.val_dataset = val_dataset
        yolov3.build_validation_model()
        val_input_encoder = \
            YOLOv3InputEncoderTensor(
                img_height=yolov3.h_tensor_val,
                img_width=yolov3.w_tensor_val,
                n_classes=yolov3.n_classes,
                feature_map_size=yolov3.val_predictor_sizes,
                anchors=yolov3.all_relative_anchors
            )
        val_dataset.set_encoder(val_input_encoder)
        yolov3.build_validation_callback(
            val_dataset,
            verbose=verbose
        )
    else:
        yolov3.build_validation_model()
        eval_sequence = YOLOv3DataSequence(
            spec.dataset_config,
            spec.augmentation_config,
            spec.eval_config.batch_size,
            is_training=False,
            encode_fn=eval_encode_fn
        )
        yolov3.val_dataset = eval_sequence
        yolov3.build_validation_callback(
            eval_sequence,
            verbose=verbose
        )
    # build checkpointer
    if hvd.rank() == 0:
        yolov3.build_savers(results_dir, verbose)
        if spec.training_config.visualizer.enabled:
            yolov3.build_tensorboard_callback(results_dir)
        yolov3.build_status_logging_callback(results_dir, yolov3.num_epochs, True)
    yolov3.compile()
    return yolov3
