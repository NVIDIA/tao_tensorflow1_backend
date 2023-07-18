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

'''build model for evaluation.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.models import Model
from nvidia_tao_tf1.cv.retinanet.layers.output_decoder_layer import DecodeDetections


def build(training_model,
          confidence_thresh=0.05,
          iou_threshold=0.5,
          top_k=200,
          nms_max_output_size=1000,
          include_encoded_pred=False):
    '''build model for evaluation.'''

    im_channel, im_height, im_width = training_model.layers[0].input_shape[1:]
    decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k,
                                           nms_max_output_size=nms_max_output_size,
                                           img_height=im_height,
                                           img_width=im_width,
                                           name='decoded_predictions')

    if include_encoded_pred:
        model_output = [training_model.layers[-1].output,
                        decoded_predictions(training_model.layers[-1].output)]
    else:
        model_output = decoded_predictions(training_model.layers[-1].output)

    eval_model = Model(inputs=training_model.layers[1].input,
                       outputs=model_output)
    new_input = Input(shape=(im_channel, im_height, im_width))
    eval_model = Model(inputs=new_input, outputs=eval_model(new_input))

    return eval_model
