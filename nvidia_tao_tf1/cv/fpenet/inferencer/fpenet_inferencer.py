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
"""FpeNet Inference Class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import cv2
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
import tqdm

from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import load_model
from nvidia_tao_tf1.cv.fpenet.models.custom.softargmax import Softargmax
from nvidia_tao_tf1.cv.fpenet.models.fpenet_basemodel import FpeNetBaseModel

logger = logging.getLogger(__name__)


# Color definition for stdout logs.
CRED = '\033[91m'
CEND = '\033[0m'


class FpeNetInferencer(object):
    """FpeNet Inference Class."""

    def __init__(self,
                 experiment_spec,
                 data_path,
                 output_folder,
                 model_path,
                 image_root_path,
                 key):
        """__init__ method.

        Args:
            experiment_spec (object): config file for the experiments
            data_path (str): path to json with image paths and ground truth face box
            output_folder (str): folder for the output files
            model_path (str): path to pre-trained model
            image_root_path (str): parent directory for the image paths in json.
            key (str): model encryption key
        """
        self._spec = experiment_spec
        self._data_path = data_path
        self._output_folder = output_folder
        if not os.path.exists(self._output_folder):
            mkdir_p(self._output_folder)
        self._model_path = model_path
        self._image_root_path = image_root_path
        self._key = key
        # Set up model
        self._model = FpeNetBaseModel(self._spec['model']['model_parameters'])
        # Set test phase.
        keras.backend.set_learning_phase(0)

    def load_model(self):
        """Load model from path and args."""
        if self._model_path is None or not os.path.isfile(self._model_path):
            raise ValueError("Please provide a valid fpenet file path for evaluation.")

        if self._model_path.endswith('.engine'):
            # Use TensorRT for inference
            # import TRTInferencer only if it's a TRT Engine.
            from nvidia_tao_tf1.cv.core.inferencer.trt_inferencer import TRTInferencer
            self.model = TRTInferencer(self._model_path)
            self.is_trt_model = True
        else:
            self._model.keras_model = load_model(self._model_path,
                                                 key=self._key,
                                                 custom_objects={'Softargmax': Softargmax})
            self.is_trt_model = False

    def infer_model(self):
        """Prepare data for inferencing and run inference."""
        # Get data from json
        json_data = json.loads(open(self._data_path , 'r').read())
        self.results = []
        for img in tqdm.tqdm(json_data):
            try:
                fname = str(img['filename'])

                if not os.path.exists(os.path.join(self._image_root_path, fname)):
                    print(CRED + 'Image does not exist: {}'.format(fname) + CEND)
                    continue

                for chunk in img['annotations']:
                    if 'facebbox' not in chunk['class'].lower():
                        continue

                    bbox_data = (entry for entry in chunk if ('class' not in entry and
                                                              'version' not in entry))
                    for entry in bbox_data:
                        if 'face_tight_bboxheight' in str(entry).lower():
                            height = int(float(chunk[entry]))
                        if 'face_tight_bboxwidth' in str(entry).lower():
                            width = int(float(chunk[entry]))
                        if 'face_tight_bboxx' in str(entry).lower():
                            x = int(float(chunk[entry]))
                        if 'face_tight_bboxy' in str(entry).lower():
                            y = int(float(chunk[entry]))

                    sample = dict()
                    sample['image_path'] = fname

                    image = cv2.imread(os.path.join(self._image_root_path, fname))
                    if image is None:
                        print(CRED + 'Bad image:{}'.format(fname) + CEND)
                        continue

                    image_shape = image.shape
                    image_height = image_shape[0]
                    image_width = image_shape[1]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.float32(image)

                    # transform it into a square bbox wrt the longer side
                    longer_side = max(width, height)
                    new_width = longer_side
                    new_height = longer_side
                    x = int(x - (new_width - width) / 2)
                    y = int(y - (new_height - height) / 2)
                    x = min(max(x, 0), image_width)
                    y = min(max(y, 0), image_height)
                    new_width = min(new_width, image_width - x)
                    new_height = min(new_height, image_height - y)
                    new_width = min(new_width, new_height)
                    new_height = new_width  # make it a square bbox
                    sample['facebox'] = [x, y, new_width, new_height]

                    # crop the face bounding box
                    img_crop = image[y:y + new_height, x:x + new_width, :]  # pylint:disable=E1136
                    target_height = self._spec['dataloader']['image_info']['image']['height']
                    target_width = self._spec['dataloader']['image_info']['image']['width']
                    image_resized = cv2.resize(img_crop,
                                               (target_height, target_width),
                                               interpolation=cv2.INTER_CUBIC)
                    if self._spec['dataloader']['image_info']['image']['channel'] == 1:
                        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                        image_resized = np.expand_dims(image_resized, 2)
                    # make it channel first (channel, height, width)
                    image_resized = np.transpose(image_resized, (2, 0, 1))
                    image_resized = np.expand_dims(image_resized, 0)  # add batch dimension
                    sample['img'] = image_resized

            except Exception as e:
                print(CRED + str(e) + CEND)

            # Run inference on current sample
            if self.is_trt_model:
                # Run model prediction using trt engine
                try:
                    output_blobs = self.model.predict(sample['img'])
                except Exception as error:
                    logger.error("TRT execution failed. Please ensure that the `input_shape` "
                                 "matches the model input dims")
                    logger.error(error)
                    raise error
                # engine generates 3 outputs including penultimate layer at [0]
                predictions_coord_results = list(output_blobs.values())[1:]
                assert len(predictions_coord_results) == 2,\
                    "Number of outputs more than 2. Please verify."
            else:
                # Get input tensors.
                input_face_tensor = keras.layers.Input(tensor=tf.convert_to_tensor(sample['img']),
                                                       name='input_face_images')
                predictions = self._model.keras_model(input_face_tensor)

                # extract keypoints and confidence from model output
                predictions_coord = K.reshape(predictions[0], (1, self._spec['num_keypoints'], 2))

                self.evaluation_tensors = [predictions_coord]

                sess = keras.backend.get_session()
                sess.run(tf.group(tf.local_variables_initializer(),
                                  tf.tables_initializer(),
                                  *tf.get_collection('iterator_init')))

                predictions_coord_results = sess.run(self.evaluation_tensors)

            # rescale predictions back to original image coordinates
            scale = float(sample['facebox'][2]) / \
                self._spec['dataloader']['image_info']['image']['height']
            shift = np.tile(np.array((sample['facebox'][0], sample['facebox'][1])),
                            (self._spec['num_keypoints'], 1))
            result = (predictions_coord_results[0][0, :, :] * scale) + shift
            self.results.append((sample['image_path'], result.flatten()))

    def save_results(self):
        """Save inference results to output folder."""
        # Write predictions and txt files
        output_filename = os.path.join(self._output_folder, 'result.txt')
        if len(self.results) > 0:
            print('Results file created: ' + output_filename)

            file_contents = []
            for frame_id, result in self.results:
                line = ' '.join([str(x) for x in result.tolist()]) + '\n'
                line = '{} {}'.format(frame_id, line)
                file_contents.append(line)

            with open(output_filename, 'w') as f:
                for line in file_contents:
                    f.write(line)
        else:
            print(CRED + 'No results to write.' + CEND)
