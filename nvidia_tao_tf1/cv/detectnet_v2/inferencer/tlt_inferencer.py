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

"""Simple inference handler for TLT trained DetectNet_v2 models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import keras
from keras import backend as K

import numpy as np

import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.inferencer.base_inferencer import Inferencer
from nvidia_tao_tf1.cv.detectnet_v2.model.utilities import model_io

logger = logging.getLogger(__name__)


class TLTInferencer(Inferencer):
    """Network handler for inference tool."""

    def __init__(self, framework='tlt', target_classes=None,
                 image_height=544, image_width=960, image_channels=3,
                 enc_key=None, tlt_model=None, gpu_set=0,
                 batch_size=1):
        """Setting up handler class for TLT DetectNet_v2 model.

        Args:
            framework (str): The framework in which the model under inference was serialized.
            target_classes (list): List of target classes in order of the network output.
            image_height (int): Height of the image at inference.
            image_width (int): Width of the image under inference.
            enc_key (str): Key to decode tlt model.
            tlt_model (str): Path to the .tlt model file generated post training.
            gpu_set (int): Id of the GPU in which inference will be run.
            batch_size (int): Number of images per batch when inferred.
        """
        # Initialize base class.
        super(TLTInferencer, self).__init__(target_classes=target_classes,
                                            image_height=image_height,
                                            image_width=image_width,
                                            image_channels=image_channels,
                                            gpu_set=gpu_set,
                                            batch_size=batch_size)
        self._key = enc_key
        self._model = tlt_model
        self.framework = framework

        # Initializing the input output nodes.
        self._set_input_output_nodes()
        for node in self.output_nodes:
            if "cov" in node:
                self.cov_blob = node
            elif "bbox" in node:
                self.bbox_blob = node
            else:
                raise ValueError("Invalid output blobs mentioned.")
        self.constructed = False

    def _set_input_output_nodes(self):
        """Set the input output nodes of the inferencer."""
        self.input_node = "input_1"
        self.output_nodes = ["output_bbox", "output_cov"]

    def network_init(self):
        """Initializing the keras model and compiling it for inference.

        Args:
            None

        Returns:
            No explicit returns. Defines the self.mdl attribute to the intialized
            keras model.
        """
        # Limit keras to using only 1 gpu of gpu id.
        gpu_id = str(self.gpu_set)

        # Restricting the number of GPU's to be used.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = gpu_id
        K.set_session(tf.Session(config=config))

        logger.info("Loading model from {}:".format(self._model))
        model = model_io(self._model, enc_key=self._key)

        # Check for model encapsulation.
        layer_types = {type(layer) for layer in model.layers}
        if keras.engine.training.Model in layer_types:
            # Model in model case.
            if layer_types != set([keras.engine.topology.InputLayer, keras.engine.training.Model]):
                raise NotImplementedError("Model encapsulation is only supported if outer model "
                                          "consists of input layers")
            # Extracting only model.
            model = [l for l in model.layers if (type(l) == keras.engine.training.Model)][0]

        # Setting data format for loaded model. This can be derived from the last layer
        # since all the layers in a DNv2 model follows the same channel order.
        self.data_format = model.get_layer(self.cov_blob).data_format
        assert self.data_format == "channels_first", "Only channels first supported"
        self.num_channels = model.layers[0].input_shape[1]
        input_shape = (self.num_channels, self.image_height, self.image_width)

        # Reshaping input to inference shape defined in the clusterfile and
        # encapusulating new model.
        # Peeling out reshape layers.
        intermediate_outputs = [model.get_layer(self.cov_blob).output,
                                model.get_layer(self.bbox_blob).output]
        model = keras.models.Model(inputs=model.inputs, outputs=intermediate_outputs)
        logger.debug("Reshaping inputs to clusterfile dimensions")
        inputs = keras.layers.Input(shape=input_shape)
        model = keras.models.Model(inputs=inputs, outputs=model(inputs))
        model.summary()
        self.mdl = model
        self.constructed = True

    def infer_batch(self, chunk):
        """Function to infer a batch of images using trained keras model.

        Args:
            chunk (array): list of images in the batch to infer.
        Returns:
            infer_out: raw_predictions from model.predict.
            resized: resized size of the batch.
        """
        if not self.constructed:
            raise ValueError("Cannot run inference. Run Inferencer.network_init() first.")

        infer_shape = (len(chunk),) + self.mdl.layers[0].input_shape[1:]
        logger.debug("Inference shape per batch: {}".format(infer_shape))
        infer_input = np.zeros(infer_shape)

        # Prepare image batches.
        logger.debug("Inferring images")
        for idx, image in enumerate(chunk):
            input_image, resized = self.input_preprocessing(image)
            infer_input[idx, :, :, :] = input_image

        # Infer on image batches.
        output = self.mdl.predict(infer_input, batch_size=len(chunk))
        infer_dict = self.predictions_to_dict(output)
        logger.debug("Inferred_outputs: {}".format(len(output)))
        infer_out = self.keras_output_map(infer_dict)
        return infer_out, resized
