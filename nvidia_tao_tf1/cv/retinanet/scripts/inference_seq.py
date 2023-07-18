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
"""Dump kitti label for KPI computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import shutil

from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
import numpy as np
from PIL import Image

from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.retinanet.builders import eval_builder
from nvidia_tao_tf1.cv.retinanet.utils.model_io import load_model
from nvidia_tao_tf1.cv.retinanet.utils.spec_loader import load_experiment_spec

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']


def parse_command_line():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description='Keras KPI Sequence Inference Tool')
    parser.add_argument('-m',
                        '--model',
                        help='Keras model file',
                        default=None)
    parser.add_argument('-i',
                        '--in_seq_dir',
                        help='The Directory to the input images')
    parser.add_argument('-e',
                        '--config_path',
                        help='Experiment spec file for training')
    parser.add_argument('-o',
                        '--out_seq_dir',
                        help='The Directory to the output kitti labels.'
                        )
    parser.add_argument('-k',
                        '--key',
                        required=False,
                        type=str,
                        default="",
                        help='Key to save or load a .tlt model.')
    parser.add_argument('-t',
                        '--out_thres',
                        type=float,
                        help='Threshold of confidence score for dumping kitti labels.',
                        default=0.3
                        )
    arguments = parser.parse_args()
    return arguments


def inference(arguments):
    '''make inference on a folder of images.'''
    config_path = arguments.config_path
    if config_path is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", config_path)
        # The spec in config_path has to be complete.
        # Default spec is not merged into experiment_spec.
        experiment_spec = load_experiment_spec(config_path, merge_from_default=False)
    else:
        logger.info("Loading default experiment spec.")
        experiment_spec = load_experiment_spec()

    K.clear_session()  # Clear previous models from memory.

    model = load_model(arguments.model, experiment_spec, key=arguments.key)

    # Load evaluation parameters
    conf_th = experiment_spec.nms_config.confidence_threshold
    iou_th = experiment_spec.nms_config.clustering_iou_threshold
    top_k = experiment_spec.nms_config.top_k
    nms_max_output = top_k
    # Build evaluation model
    model = eval_builder.build(model, conf_th, iou_th, top_k, nms_max_output)
    img_channel = model.layers[0].output_shape[-3]
    img_width = model.layers[0].output_shape[-1]
    img_height = model.layers[0].output_shape[-2]
    # check if it's a monochrome model or RGB model
    img_mode = 'RGB' if img_channel == 3 else 'L'
    classes = sorted({str(x) for x in
                     experiment_spec.dataset_config.target_class_mapping.values()})
    class_mapping = dict(zip(range(len(classes)), classes))
    # Create output directory
    if os.path.exists(expand_path(arguments.out_seq_dir)):
        shutil.rmtree(arguments.out_seq_dir)
    os.mkdir(arguments.out_seq_dir)

    inf_seq_list = []
    for folder in os.listdir(arguments.in_seq_dir):
        f = expand_path(f"{arguments.in_seq_dir}/{folder}")
        if os.path.isdir(f) and os.path.exists(os.path.join(f, 'images')):
            inf_seq_list.append(folder)
    print('seqs:', inf_seq_list)

    for folder in inf_seq_list:
        in_base = expand_path(f"{arguments.in_seq_dir}/{folder}/images")
        os.mkdir(expand_path(f"{arguments.out_seq_dir}/{folder}"))
        out_base = expand_path(f"{arguments.out_seq_dir}/{folder}/labels")
        os.mkdir(expand_path(out_base))
        if os.path.isdir(in_base):
            for img_path in os.listdir(in_base):
                if os.path.splitext(img_path)[1] not in image_extensions:
                    continue
                img = Image.open(os.path.join(in_base, img_path))
                orig_w, orig_h = [float(x) for x in img.size]
                ratio = min(img_width/orig_w, img_height/orig_h)

                # do not change aspect ratio
                new_w = int(round(orig_w*ratio))
                new_h = int(round(orig_h*ratio))
                im = img.resize((new_w, new_h), Image.ANTIALIAS)
                if im.mode in ('RGBA', 'LA') or \
                        (im.mode == 'P' and 'transparency' in im.info) and img_channel == 1:
                    bg_colour = (255, 255, 255)
                    # Need to convert to RGBA if LA format due to a bug in PIL
                    alpha = im.convert('RGBA').split()[-1]
                    # Create a new background image of our matt color.
                    # Must be RGBA because paste requires both images have the same format
                    bg = Image.new("RGBA", im.size, bg_colour + (255,))
                    bg.paste(im, mask=alpha)
                    inf_img = im.convert(img_mode)
                else:
                    inf_img = Image.new(img_mode, (img_width, img_height))
                    inf_img.paste(im, (0, 0))
                inf_img = np.array(inf_img).astype(np.float32)
                if img_mode == 'L':
                    inf_img = np.expand_dims(inf_img, axis=2)
                    inference_input = inf_img.transpose(2, 0, 1) - 117.3786
                else:
                    inference_input = preprocess_input(inf_img.transpose(2, 0, 1))

                # run inference
                y_pred_decoded = model.predict(np.array([inference_input]))
                kitti_txt = ""
                decode_ratio = (orig_w/new_w, orig_h/new_h)

                for i in y_pred_decoded[0]:
                    if i[1] < arguments.out_thres:
                        continue
                    xmin = decode_ratio[0]*i[2]
                    xmax = decode_ratio[0]*i[4]
                    ymin = decode_ratio[1]*i[3]
                    ymax = decode_ratio[1]*i[5]
                    kitti_txt += class_mapping[int(i[0])] + ' 0 0 0 ' + \
                        ' '.join([str(x) for x in [xmin, ymin, xmax, ymax]]) + \
                        ' 0 0 0 0 0 0 0 ' + str(i[1]) + '\n'

                open(os.path.join(out_base, os.path.splitext(img_path)[0] + '.txt'),
                    'w').write(kitti_txt)


if __name__ == "__main__":
    arguments = parse_command_line()
    inference(arguments)
