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
"""FasterRCNN inference script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import logging
import os
import sys

import cv2
from keras import backend as K
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tqdm import tqdm

try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    from nvidia_tao_tf1.cv.faster_rcnn.tensorrt_inference.tensorrt_model import TrtModel
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
from nvidia_tao_tf1.cv.common import utils as iva_utils
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.faster_rcnn.models.utils import build_inference_model
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader import spec_loader, spec_wrapper
from nvidia_tao_tf1.cv.faster_rcnn.utils import utils

KERAS_MODEL_EXTENSIONS = ["tlt", "hdf5"]


def build_command_line_parser(parser=None):
    """Build a command line parser for inference."""
    if parser is None:
        parser = argparse.ArgumentParser(description='''Do inference on the pretrained model
                                            and visualize the results.''')
    parser.add_argument("-e",
                        "--experiment_spec",
                        type=str,
                        required=True,
                        help="Experiment spec file has all the training params.")
    parser.add_argument("-k",
                        "--key",
                        type=str,
                        required=False,
                        help="TLT encoding key, can override the one in the spec file.")
    parser.add_argument("-m",
                        "--model_path",
                        type=str,
                        required=False,
                        default=None,
                        help="Path to the model to be used for inference")
    parser.add_argument("-r",
                        "--results_dir",
                        type=str,
                        default=None,
                        help="Path to the files where the logs are stored.")
    return parser


def parse_args(args=None):
    '''Parser arguments.'''
    parser = build_command_line_parser()
    return parser.parse_known_args(args)[0]


def main(args=None):
    """Do inference on a pretrained model."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    K.set_image_data_format('channels_first')
    K.set_learning_phase(0)
    options = parse_args(args)
    spec = spec_loader.load_experiment_spec(options.experiment_spec)
    # enc key in CLI will override the one in the spec file.
    if options.key is not None:
        spec.enc_key = options.key
    # model in CLI will override the one in the spec file.
    if options.model_path is not None:
        spec.inference_config.model = options.model_path
    spec = spec_wrapper.ExperimentSpec(spec)
    # Set up status logging
    if options.results_dir:
        if not os.path.exists(options.results_dir):
            os.makedirs(options.results_dir)
        status_file = os.path.join(options.results_dir, "status.json")
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
            message="Starting FasterRCNN inference."
        )
    verbosity = 'INFO'
    if spec.verbose:
        verbosity = 'DEBUG'
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=verbosity)
    logger = logging.getLogger(__name__)
    # set radom seed
    utils.set_random_seed(spec.random_seed)

    img_path = spec.inference_images_dir
    class_names = spec.class_to_id.keys()
    class_to_color = {v: np.random.randint(0, 255, 3) for v in class_names}

    # load model and convert train model to infer model
    if spec.inference_trt_config is not None:
        # spec.inference_trt_engine will be deprecated, use spec.inference_model
        logger.info('Running inference with TensorRT as backend.')
        logger.warning(
            "`spec.inference_config.trt_inference` is deprecated, "
            "please use `spec.inference_config.model` in spec file or provide "
            "the model/engine as a command line argument instead."
        )
        if (spec.image_h == 0 or spec.image_w == 0):
            raise(
                ValueError("TensorRT inference is not supported when using dynamic input shape.")
            )
        infer_model = TrtModel(spec.inference_trt_engine,
                               spec.infer_batch_size,
                               spec.image_h,
                               spec.image_w)
        infer_model.build_or_load_trt_engine()
    elif spec.inference_model.split('.')[-1] in KERAS_MODEL_EXTENSIONS:
        # spec.inference_model is a TLT model
        logger.info('Running inference with TLT as backend.')
        # in case of dynamic shape, batch size has to be 1
        if (
            (spec.image_h == 0 or spec.image_w == 0) and
            spec.infer_batch_size != 1
        ):
            raise(ValueError("Only batch size 1 is supported when using dynamic input shapes."))
        train_model = iva_utils.decode_to_keras(spec.inference_model,
                                                str.encode(spec.enc_key),
                                                input_model=None,
                                                compile_model=False,
                                                by_name=None)
        config_override = {'pre_nms_top_N': spec.infer_rpn_pre_nms_top_N,
                           'post_nms_top_N': spec.infer_rpn_post_nms_top_N,
                           'nms_iou_thres': spec.infer_rpn_nms_iou_thres,
                           'bs_per_gpu': spec.infer_batch_size}
        logger.info("Building inference model, may take a while...")
        infer_model = build_inference_model(
            train_model,
            config_override,
            create_session=False,
            max_box_num=spec.infer_rcnn_post_nms_top_N,
            regr_std_scaling=spec.rcnn_regr_std,
            iou_thres=spec.infer_rcnn_nms_iou_thres,
            score_thres=spec.infer_confidence_thres,
            eval_rois=spec.infer_rpn_post_nms_top_N
        )
        infer_model.summary()
    else:
        # spec.inference_model is a TensorRT engine
        logger.info('Running inference with TensorRT as backend.')
        if (spec.image_h == 0 or spec.image_w == 0):
            raise(
                ValueError("TensorRT inference is not supported when using dynamic input shape.")
            )
        infer_model = TrtModel(spec.inference_model,
                               spec.infer_batch_size,
                               spec.image_h,
                               spec.image_w)
        infer_model.build_or_load_trt_engine()
    output_dir = spec.inference_output_images_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    old_files = glob.glob(output_dir+'/*')
    for of in old_files:
        if os.path.isfile(of):
            os.remove(of)

    image_set = os.listdir(img_path)
    for img_name in image_set:
        if not img_name.endswith(('.jpeg', '.jpg', '.png')):
            logger.info('Invalid image found: {}, '
                        'please ensure the image extension '
                        'is jpg, jpeg or png'
                        ', exit.'.format(img_name))
            sys.exit(1)
    image_num = len(image_set)
    num_iters = (image_num + spec.infer_batch_size - 1) // spec.infer_batch_size
    im_type = cv2.IMREAD_COLOR if spec.image_c == 3 else cv2.IMREAD_GRAYSCALE

    for idx in tqdm(range(num_iters)):
        # the last batch can be smaller
        image_batch = image_set[idx*spec.infer_batch_size:(idx+1)*spec.infer_batch_size]
        filepaths = [os.path.join(img_path, img_name) for img_name in image_batch]
        img_list = [cv2.imread(filepath, im_type) for filepath in filepaths]
        X, ratio, orig_shape = utils.preprocess_image_batch(
            img_list,
            spec.image_h,
            spec.image_w,
            spec.image_c,
            spec.image_min,
            spec.image_scaling_factor,
            spec.image_mean_values,
            spec.image_channel_order
        )
        # The Keras model is sensitive to batch size so we have to pad for the last
        # batch if it is a smaller batch
        use_pad = False
        if X.shape[0] < spec.infer_batch_size:
            use_pad = True
            X_pad = np.zeros((spec.infer_batch_size,) + X.shape[1:], dtype=X.dtype)
            X_pad[0:X.shape[0], ...] = X
        else:
            X_pad = X
        nmsed_boxes, nmsed_scores, nmsed_classes, num_dets, _ = \
            infer_model.predict(X_pad)
        if use_pad:
            nmsed_boxes = nmsed_boxes[0:X.shape[0], ...]
            nmsed_scores = nmsed_scores[0:X.shape[0], ...]
            nmsed_classes = nmsed_classes[0:X.shape[0], ...]
            num_dets = num_dets[0:X.shape[0], ...]

        for image_idx in range(nmsed_boxes.shape[0]):
            img = img_list[image_idx]
            # use PIL for TrueType fonts, for better visualization
            # openCV: BGR, PIL: RGB
            img_pil = Image.fromarray(np.array(img)[:, :, ::-1])
            imgd = ImageDraw.Draw(img_pil)
            all_dets_dump = []
            orig_h, orig_w = orig_shape[image_idx]
            for jk in range(num_dets[image_idx]):
                new_probs = nmsed_scores[image_idx, jk]
                # skip boxes whose confidences are lower than visualize thres
                if (new_probs < spec.vis_conf or
                   nmsed_classes[image_idx, jk] not in spec.id_to_class):
                    continue
                cls_name = spec.id_to_class[nmsed_classes[image_idx, jk]]
                y1, x1, y2, x2 = nmsed_boxes[image_idx, jk, :]
                (real_x1, real_y1, real_x2, real_y2) = utils.get_original_coordinates(
                    ratio[image_idx],
                    x1,
                    y1,
                    x2,
                    y2,
                    orig_h,
                    orig_w
                )
                p1 = (real_x1, real_y1)
                p2 = (real_x2, real_y2)
                p3 = (int(class_to_color[cls_name][0]),
                      int(class_to_color[cls_name][1]),
                      int(class_to_color[cls_name][2]))
                # draw bbox and caption
                imgd.rectangle([p1, p2], outline=p3)
                textLabel = '{}: {:.4f}'.format(cls_name, new_probs)
                if spec.inference_config.bbox_caption_on:
                    text_size = [p1, (real_x1 + 100, real_y1 + 10)]
                    imgd.rectangle(text_size, outline='white', fill='white')
                    imgd.text(p1, textLabel, fill='black')
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
                       'class': cls_name, 'prob': new_probs}
                all_dets_dump.append(det)
            if type(ratio[image_idx]) is tuple:
                ratio_2 = (1.0/ratio[image_idx][1], 1.0/ratio[image_idx][0])
            elif type(ratio[image_idx]) is float:
                ratio_2 = (1.0/ratio[image_idx], 1.0/ratio[image_idx])
            else:
                raise TypeError('invalid data type for ratio.')

            utils.dump_kitti_labels(filepaths[image_idx],
                                    all_dets_dump,
                                    ratio_2,
                                    spec.inference_output_labels_dir,
                                    spec.vis_conf)
            img_pil.save(os.path.join(output_dir, image_batch[image_idx]))
    logger.info("Inference output images directory: {}".format(output_dir))
    logger.info("Inference output labels directory: {}".format(spec.inference_output_labels_dir))
    if options.results_dir:
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
        if type(e) == tf.errors.ResourceExhaustedError:
            logger = logging.getLogger(__name__)
            logger.error(
                "Ran out of GPU memory, please lower the batch size, use a smaller input "
                "resolution, use a smaller backbone or try model parallelism. See TLT "
                "documentation on how to enable model parallelism for FasterRCNN."
            )
            status_logging.get_status_logger().write(
                message="Ran out of GPU memory, please lower the batch size, use a smaller input "
                        "resolution, use a smaller backbone or try model parallelism. See TLT "
                        "documentation on how to enable model parallelism for FasterRCNN.",
                status_level=status_logging.Status.FAILURE
            )
            sys.exit(1)
        else:
            # throw out the error as-is if they are not OOM error
            status_logging.get_status_logger().write(
                message=str(e),
                status_level=status_logging.Status.FAILURE
            )
            raise e
