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

from keras.models import Model
from nvidia_tao_tf1.cv.yolo_v3.layers.nms_layer import NMSLayer
from nvidia_tao_tf1.cv.yolo_v4.layers.decode_layer import YOLOv4DecodeLayer


def build(training_model,
          confidence_thresh=0.05,
          iou_threshold=0.5,
          top_k=200,
          include_encoded_head=False,
          nms_on_cpu=False):
    '''
    build model for evaluation.

    Args:
        training_model: Keras model built from model_builder. Last layer is encoded prediction.
        confidence_thresh: ignore all boxes with confidence less then threshold
        iou_threshold: iou threshold for NMS
        top_k: pick top k boxes with highest confidence scores after NMS
        include_encoded_head: whether to include original model output into final model output.
        nms_on_cpu(bool): Flag to force NMS to run on CPU as GPU NMS is flaky with tfrecord dataset.

    Returns:
        eval_model: keras model that outputs at most top_k detection boxes.
    '''

    decoded_predictions = YOLOv4DecodeLayer(name='decoded_predictions')

    x = decoded_predictions(training_model.layers[-1].output)

    nms = NMSLayer(output_size=top_k,
                   iou_threshold=iou_threshold,
                   score_threshold=confidence_thresh,
                   force_on_cpu=nms_on_cpu,
                   name="NMS")
    x = nms(x)

    x = [training_model.layers[-1].output, x] if include_encoded_head else x

    eval_model = Model(inputs=training_model.layers[0].input,
                       outputs=x)

    return eval_model
