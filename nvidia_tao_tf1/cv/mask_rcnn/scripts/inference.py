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
"""Mask RCNN Inference script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import math
import os
from google.protobuf.json_format import MessageToDict
import six

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.mask_rcnn.executer import distributed_executer
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import mask_rcnn_params
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import params_io
from nvidia_tao_tf1.cv.mask_rcnn.models import mask_rcnn_model
from nvidia_tao_tf1.cv.mask_rcnn.ops import preprocess_ops
from nvidia_tao_tf1.cv.mask_rcnn.scripts.inference_trt import Inferencer
from nvidia_tao_tf1.cv.mask_rcnn.utils.spec_loader import load_experiment_spec


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


def get_label_dict(label_txt):
    """Create label dict from txt file."""

    with open(label_txt, 'r') as f:
        labels = f.readlines()
        return {i+1 : label[:-1] for i, label in enumerate(labels)}


def postprocess_fn(y_pred, nms_size, mask_size, n_classes):
    """Proccess raw output from TRT engine."""
    y_detection = y_pred[0].reshape((-1, nms_size, 6))
    y_mask = y_pred[1].reshape((-1, nms_size, n_classes, mask_size, mask_size))
    y_mask[y_mask < 0] = 0
    return [y_detection, y_mask]


def resize_and_pad(image, target_size, stride=64):
    """Resize and pad images, boxes and masks.

    Resize and pad images, given the desired output
    size of the image and stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `target_size`.
    2. Pad the rescaled image such that the height and width of the image become
     the smallest multiple of the stride that is larger or equal to the desired
     output diemension.

    Args:
    image: an image tensor of shape [original_height, original_width, 3].
    target_size: a tuple of two integers indicating the desired output
      image size. Note that the actual output size could be different from this.
    stride: the stride of the backbone network. Each of the output image sides
      must be the multiple of this.

    Returns:
    image: the processed image tensor after being resized and padded.
    image_info: a tensor of shape [5] which encodes the height, width before
      and after resizing and the scaling factor.
    """

    input_height, input_width, _ = tf.unstack(
        tf.cast(tf.shape(input=image), dtype=tf.float32),
        axis=0
    )

    target_height, target_width = target_size

    scale_if_resize_height = target_height / input_height
    scale_if_resize_width = target_width / input_width

    scale = tf.minimum(scale_if_resize_height, scale_if_resize_width)

    scaled_height = tf.cast(scale * input_height, dtype=tf.int32)
    scaled_width = tf.cast(scale * input_width, dtype=tf.int32)

    image = tf.image.resize(image, [scaled_height, scaled_width],
                            method=tf.image.ResizeMethod.BILINEAR)

    padded_height = int(math.ceil(target_height * 1.0 / stride) * stride)
    padded_width = int(math.ceil(target_width * 1.0 / stride) * stride)

    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_height, padded_width)
    image.set_shape([padded_height, padded_width, 3])

    image_info = tf.stack([
        tf.cast(scaled_height, dtype=tf.float32),
        tf.cast(scaled_width, dtype=tf.float32),
        1.0 / scale,
        input_height,
        input_width]
    )
    return image, image_info


def build_command_line_parser(parser=None):
    '''Parse command line arguments.'''
    if parser is None:
        parser = argparse.ArgumentParser(description="Run MaskRCNN inference.")
    parser.add_argument('-m', '--model_path',
                        type=str,
                        help="Path to a MaskRCNN model.",
                        required=True)
    parser.add_argument('-i', '--image_dir',
                        type=str,
                        required=True,
                        help="Path to the input image directory.")
    parser.add_argument('-k',
                        '--key',
                        type=str,
                        default="",
                        required=False,
                        help="Encryption key.")
    parser.add_argument('-c', '--class_map',
                        type=str,
                        required=False,
                        default='',
                        help="Path to the label file.")
    parser.add_argument('-t', '--threshold',
                        type=float,
                        required=False,
                        default=0.6,
                        help="Bbox confidence threshold.")
    parser.add_argument('--include_mask', action='store_true',
                        help="Whether to draw masks.")
    parser.add_argument('-e', '--experiment_spec',
                        type=str,
                        required=True,
                        help='Path to spec file. Absolute path or relative to working directory. \
                              If not specified, default spec from spec_loader.py is used.')
    parser.add_argument('-r', '--results_dir',
                        type=str,
                        default='/tmp',
                        required=False,
                        help='Output directory where the status log is saved.')
    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def pad_img_list(img_list, batch_size):
    """Pad image list to fit batch size."""
    if not img_list:
        return img_list
    assert batch_size > 0, "Batch size should be greater than 0."
    num_to_pad = 0
    if len(img_list) % batch_size != 0:
        num_to_pad = batch_size - len(img_list) % batch_size
    return img_list + [img_list[0]] * num_to_pad


def infer(args):
    """Run MaskRCNN TLT inference."""
    VALID_IMAGE_EXT = ['.jpg', '.png', '.jpeg']
    assert args.key is not None, "Did you forget to specify your encryption key?"
    assert 'epoch' in args.model_path, "The pruned model must be retrained first."
    # Check directories
    assert os.path.exists(args.image_dir), "Image directory does not exist."
    imgpath_list = [os.path.join(args.image_dir, imgname)
                    for imgname in sorted(os.listdir(args.image_dir))
                    if os.path.splitext(imgname)[1].lower()
                    in VALID_IMAGE_EXT]
    if os.path.exists(args.class_map):
        label_dict = get_label_dict(args.class_map)
    else:
        label_dict = {}
        print('Label file does not exist. Skipping...')

    out_image_path = os.path.join(args.results_dir, "images_annotated")
    os.makedirs(out_image_path, exist_ok=True)

    # ============================ Configure parameters ============================ #
    RUN_CONFIG = mask_rcnn_params.default_config()
    experiment_spec = load_experiment_spec(args.experiment_spec)
    temp_config = MessageToDict(experiment_spec,
                                preserving_proto_field_name=True,
                                including_default_value_fields=True)
    try:
        data_config = temp_config['data_config']
        maskrcnn_config = temp_config['maskrcnn_config']
    except ValueError:
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
    else:
        raise ValueError("image_size is not set.")
    max_stride = 2 ** temp_config['max_level']
    if temp_config['image_size'][0] % max_stride != 0 or \
            temp_config['image_size'][1] % max_stride != 0:
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
    if 'num_steps_per_eval' in temp_config:
        temp_config['save_checkpoints_steps'] = temp_config['num_steps_per_eval']
    else:
        raise ValueError("num_steps_per_eval is not set.")
    # force some params to default value
    temp_config['use_fake_data'] = False
    temp_config['allow_xla_at_inference'] = False
    temp_config['use_xla'] = False
    # load model from json graphs in the same dir as the checkpoint
    temp_config['pruned_model_path'] = os.path.dirname(args.model_path)
    infer_batch_size = eval_str(temp_config.get('eval_batch_size', 0))
    assert infer_batch_size > 0, \
        "eval_batch_size will be used for inference. \
        It should be set to the same value that you use during training."
    assert temp_config['num_classes'] > 1, "Please verify num_classes in the spec file. \
        num_classes should be number of categories in json + 1."
    imgpath_list = pad_img_list(imgpath_list, infer_batch_size)
    RUN_CONFIG = params_io.override_hparams(RUN_CONFIG, temp_config)
    RUN_CONFIG.model_path = args.model_path
    RUN_CONFIG.key = args.key
    RUN_CONFIG.mode = 'eval'
    RUN_CONFIG.threshold = args.threshold
    RUN_CONFIG.label_dict = label_dict
    RUN_CONFIG.batch_size = infer_batch_size
    RUN_CONFIG.num_infer_samples = len(imgpath_list)
    RUN_CONFIG.include_mask = args.include_mask
    RUN_CONFIG.output_dir = out_image_path
    """Initialize executer."""
    executer = distributed_executer.EstimatorExecuter(RUN_CONFIG,
                                                      mask_rcnn_model.mask_rcnn_model_fn)

    def infer_input_fn():

        def process_path(file_path):
            img = tf.io.read_file(file_path)
            orig_image = decode_img(img)
            img, info = resize_and_pad(preprocess_ops.normalize_image(orig_image),
                                       eval_str(RUN_CONFIG.image_size), max_stride)
            orig_image = tf.image.resize(orig_image, eval_str(RUN_CONFIG.image_size))

            features = {}
            features["image_path"] = file_path
            features["images"] = img
            features["image_info"] = info
            features["orig_images"] = orig_image
            features["source_ids"] = 0
            return {"features": features}

        def decode_img(img):
            # convert the compressed string to a 3D uint8 tensor
            # Note: use decode_png to decode both jpg and png
            # decode_image doesn't return image shape
            img = tf.image.decode_png(img, channels=3)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, tf.float32)
            return img

        list_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(imgpath_list))
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        processed_ds = list_ds.map(map_func=process_path,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        processed_ds = processed_ds.batch(
            batch_size=infer_batch_size,
            drop_remainder=True
        )
        processed_ds = processed_ds.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )
        return processed_ds

    # Run inference
    executer.infer(infer_input_fn=infer_input_fn)


def infer_trt(args):
    """Run MaskRCNN TRT inference."""
    out_image_path = os.path.join(args.results_dir, "images_annotated")
    out_label_path = os.path.join(args.results_dir, "labels")
    os.makedirs(out_image_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)

    spec = load_experiment_spec(args.experiment_spec)
    mask_size = int(spec.maskrcnn_config.mrcnn_resolution)
    nms_size = int(spec.maskrcnn_config.test_detections_per_image)
    assert nms_size > 0, "test_detections_per_image must be greater than 0."
    assert mask_size > 1, "mask_size must be greater than 1."
    n_classes = int(spec.data_config.num_classes)
    assert n_classes > 1, "Please verify num_classes in the spec file. \
        num_classes should be number of categories in json + 1."
    trt_output_process_fn = partial(postprocess_fn, mask_size=mask_size,
                                    n_classes=n_classes, nms_size=nms_size)
    if os.path.exists(args.class_map):
        label_dict = get_label_dict(args.class_map)
    else:
        label_dict = {}
        print('Label file does not exist. Skipping...')
    # Initialize inferencer
    inferencer = Inferencer(trt_engine_path=args.model_path,
                            infer_process_fn=trt_output_process_fn,
                            class_mapping=label_dict,
                            threshold=args.threshold)
    inferencer.infer(args.image_dir,
                     out_image_path,
                     out_label_path,
                     args.include_mask)


def main(args=None):
    """Run the inference process."""
    args = parse_command_line(args)
    model_ext = os.path.splitext(args.model_path)[1]

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
        message="Starting MaskRCNN inference."
    )
    if model_ext == '.tlt':
        disable_eager_execution()
        infer(args)
    elif model_ext == '.engine':
        infer_trt(args)
    else:
        raise ValueError("Model extension needs to be either .engine or .tlt.")
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Inference finished successfully."
    )


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
