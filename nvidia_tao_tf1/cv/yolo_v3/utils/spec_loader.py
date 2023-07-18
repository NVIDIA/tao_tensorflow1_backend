# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

"""Load an experiment spec file to run YOLOv3 training, evaluation, pruning."""

from google.protobuf.text_format import Merge as merge_text_proto

import nvidia_tao_tf1.cv.yolo_v3.proto.experiment_pb2 as experiment_pb2


def load_experiment_spec(spec_path=None):
    """Load experiment spec from a .txt file and return an experiment_pb2.Experiment object.

    Args:
        spec_path (str): location of a file containing the custom experiment spec proto.

    Returns:
        experiment_spec: protocol buffer instance of type experiment_pb2.Experiment.
    """
    experiment_spec = experiment_pb2.Experiment()
    merge_text_proto(open(spec_path, "r").read(), experiment_spec)

    # dataset_config
    assert len(experiment_spec.dataset_config.target_class_mapping.values()) > 0, \
        "Please specify target_class_mapping"
    data_sources = experiment_spec.dataset_config.data_sources
    assert len(data_sources) > 0, "Please specify training data sources"
    train_label_types = [
        s.WhichOneof("labels_format") for s in data_sources
    ]
    assert len(list(set(train_label_types))) == 1, (
        "Label format should be identical for all training data sources. Got {}".format(
            train_label_types
        )
    )
    if train_label_types[0] == "tfrecords_path":
        assert len(experiment_spec.dataset_config.image_extension) > 0, (
            "`image_extension` should be specified in `dataset_config` if training "
            " label format is TFRecord."
        )
    if len(experiment_spec.dataset_config.validation_data_sources) > 0:
        val_data_source = experiment_spec.dataset_config.validation_data_sources
        val_label_types = [
            s.WhichOneof("labels_format") for s in val_data_source
        ]
        assert len(list(set(val_label_types))) == 1, (
            "Label format should be identical for all validation data sources. Got {}".format(
                val_label_types
            )
        )
        if val_label_types[0] == "tfrecords_path":
            assert len(experiment_spec.dataset_config.image_extension) > 0, (
                "`image_extension` should be specified in `dataset_config` if validation "
                " label format is TFRecord."
            )
    else:
        assert data_sources[0].WhichOneof("labels_format") == "tfrecords_path", (
            "Validation dataset specified by `validation_fold` requires the training label format "
            "to be TFRecords."
        )
    # augmentation config
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
    assert 0.0 <= experiment_spec.augmentation_config.hue <= 1.0, "hue must be within [0, 1]"
    assert experiment_spec.augmentation_config.saturation >= 1.0, "saturation must be at least 1.0"
    assert experiment_spec.augmentation_config.exposure >= 1.0, "exposure must be at least 1.0"
    assert 0.0 <= experiment_spec.augmentation_config.vertical_flip <= 1.0, \
        "vertical_flip must be within [0, 1]"
    assert 0.0 <= experiment_spec.augmentation_config.horizontal_flip <= 1.0, \
        "horizontal_flip must be within [0, 1]"
    assert 0.0 <= experiment_spec.augmentation_config.jitter <= 1.0, "jitter must be within [0, 1]"
    assert experiment_spec.augmentation_config.output_width >= 32, "width must be at least 32"
    assert experiment_spec.augmentation_config.output_width % 32 == 0, \
        "width must be multiple of 32"
    assert experiment_spec.augmentation_config.output_height >= 32, "height must be at least 32"
    assert experiment_spec.augmentation_config.output_height % 32 == 0, \
        "height must be multiple of 32"
    assert experiment_spec.augmentation_config.randomize_input_shape_period >= 0, \
        "randomize_input_shape_period should be non-negative"

    # training config
    assert experiment_spec.training_config.batch_size_per_gpu > 0, "batch size must be positive"
    assert experiment_spec.training_config.num_epochs > 0, \
        "number of training batchs must be positive"
    assert experiment_spec.training_config.checkpoint_interval > 0, \
        "checkpoint interval must be positive"

    # eval config
    assert experiment_spec.eval_config.batch_size > 0, "batch size must be positive"
    assert 0.0 < experiment_spec.eval_config.matching_iou_threshold <= 1.0, \
        "matching_iou_threshold must be within (0, 1]"

    # nms config
    assert 0.0 < experiment_spec.nms_config.clustering_iou_threshold <= 1.0, \
        "clustering_iou_threshold must be within (0, 1]"

    # yolo_v3 config
    assert 0.0 < experiment_spec.yolov3_config.matching_neutral_box_iou < 1.0, \
        "matching_neutral_box_iou must be within (0, 1]"
    assert experiment_spec.yolov3_config.arch_conv_blocks in [0, 1, 2], \
        "arch_conv_blocks must be either 0, 1 or 2"
    assert experiment_spec.yolov3_config.loss_loc_weight >= 0.0, \
        "all loss weights must be non-negative"
    assert experiment_spec.yolov3_config.loss_neg_obj_weights >= 0.0, \
        "all loss weights must be non-negative"
    assert experiment_spec.yolov3_config.loss_class_weights >= 0.0, \
        "all loss weights must be non-negative"

    return experiment_spec


def validation_labels_format(spec):
    """The format of the labels of validation set."""
    if len(spec.dataset_config.validation_data_sources) > 0:
        if (
            spec.dataset_config.validation_data_sources[0].WhichOneof("labels_format") ==
            "label_directory_path"
        ):
            return "keras_sequence"
    return "tfrecords"
