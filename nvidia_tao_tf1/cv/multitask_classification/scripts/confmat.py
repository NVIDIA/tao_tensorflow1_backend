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

"""Generates confusion matrix on evaluation dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import keras
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from nvidia_tao_tf1.cv.common.utils import check_tf_oom
from nvidia_tao_tf1.cv.multitask_classification.utils.model_io import load_model


def build_command_line_parser(parser=None):
    """Build a command line parser for confmat generation."""
    if parser is None:
        parser = argparse.ArgumentParser(description="TLT MultiTask Confusion Matrix Generator")

    parser.add_argument("--model",
                        "-m",
                        type=str,
                        help="TLT model file")
    parser.add_argument("--img_root",
                        "-i",
                        type=str,
                        help="test image dir")
    parser.add_argument("--target_csv",
                        "-l",
                        type=str,
                        help="Target CSV file")
    parser.add_argument("--key",
                        "-k",
                        default="",
                        type=str,
                        help="TLT model key")

    return parser


def parse_command_line_arguments(args=None):
    """Parse command line arguments for confmat."""
    parser = build_command_line_parser()
    return vars(parser.parse_known_args(args)[0])


def confmat(model_file, image_dir, csv_file, key):
    """Get prediction confusion matrix."""
    # get class mapping
    df = pd.read_csv(csv_file)
    tasks_header = sorted(df.columns.tolist()[1:])
    class_num = []
    class_mapping = []
    conf_matrix = {}
    for task in tasks_header:
        unique_vals = sorted(df.loc[:, task].unique())
        class_num.append(len(unique_vals))
        class_mapping.append(dict(zip(range(len(unique_vals)), unique_vals)))
        # initialize confusion matrix
        conf_matrix[task] = pd.DataFrame(0, index=unique_vals, columns=unique_vals)

    # get model
    # set custom_object to arbitrary function to avoid not_found error.
    keras.backend.set_learning_phase(0)
    model = load_model(model_file, key=key)

    # Use list() so tqdm knows total size
    for _, row in tqdm(list(df.iterrows())):
        true_label = [row[l] for l in tasks_header]
        pred_label = [class_mapping[i][val] for i, val in enumerate(
            inference(model, os.path.join(image_dir, row.values[0]), class_num))]
        for i in range(len(true_label)):
            conf_matrix[tasks_header[i]].at[pred_label[i], true_label[i]] += 1
    return conf_matrix


@check_tf_oom
def inference(model, img_path, class_num):
    """Performing Inference."""
    # extracting the data format parameter to detect input shape
    data_format = model.layers[1].data_format

    # Computing shape of input tensor
    image_shape = model.layers[0].input_shape[1:4]

    # Setting input shape
    if data_format == "channels_first":
        image_height, image_width = image_shape[1:3]
    else:
        image_height, image_width = image_shape[0:2]

    # Open image and preprocessing
    image = Image.open(img_path)
    image = image.resize((image_width, image_height), Image.ANTIALIAS).convert('RGB')
    inference_input = preprocess_input(np.array(image).astype(np.float32).transpose(2, 0, 1))
    inference_input.shape = (1, ) + inference_input.shape

    # Keras inference
    raw_predictions = model.predict(inference_input, batch_size=1)

    return [np.argmax(x.reshape(-1)) for x in raw_predictions]


if __name__ == "__main__":
    arguments = parse_command_line_arguments()

    # Do not omit rows / cols
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    conf_matrix = confmat(arguments['model'], image_dir=arguments['img_root'],
                          csv_file=arguments['target_csv'], key=arguments['key'])
    print('Row corresponds to predicted label and column corresponds to ground-truth')
    for task, table in list(conf_matrix.items()):
        print("********")
        print("For task", task)
        print(table)
        print("Accuracy:", table.values.trace() / table.values.sum())
