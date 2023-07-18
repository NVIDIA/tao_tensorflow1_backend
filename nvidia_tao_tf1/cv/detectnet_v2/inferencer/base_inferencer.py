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
from addict import Dict

import numpy as np
from PIL import Image

from six.moves import range

logger = logging.getLogger(__name__)


class Inferencer(object):
    """Base inference handler for TLT generated models."""

    def __init__(self,
                 target_classes=None,
                 image_height=544,
                 image_width=960,
                 image_channels=3,
                 gpu_set=0,
                 batch_size=1):
        """Setting up init for the base inference module.

        Args:
            target_classes (list): List of target classes in order of the network output.
            image_height (int): Height of the image at inference.
            image_width (int): Width of the image under inference.
            gpu_set (int): Id of the GPU in which inference will be run.
            batch_size (int): Number of images per batch when inferred.
        """
        self.gpu_set = gpu_set
        self.batch_size = batch_size
        self.num_channels = None
        self.target_classes = target_classes
        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = image_channels
        assert self.num_channels in [1, 3], (
            "Number of channels in the input: {}".format(self.num_channels)
        )

    def _set_input_output_nodes(self):
        """Set the input output nodes of the inferencer."""
        raise NotImplementedError("Implemented in the derived classes.")

    def network_init(self):
        """Initializing the keras model and compiling it for inference.

        Args:
            None

        Returns:
            No explicit returns. Defines the self.mdl attribute to the intialized
            keras model.
        """
        raise NotImplementedError("Implemented in derived classes.")

    def infer_batch(self, chunk):
        """Function to infer a batch of images using trained keras model.

        Args:
            chunk (array): list of images in the batch to infer.
        Returns:
            infer_out: raw_predictions from model.predict.
            resized: resized size of the batch.
        """
        raise NotImplementedError("Implemented in derived classes.")

    def predictions_to_dict(self, outputs):
        """Function to convert raw predictions into a dictionary.

        Args:
            outputs (array): Raw outputs from keras model.predict.
        Returns:
            out_dict (Dictionary): Output predictions in a dictionary of coverages and bboxes.
        """
        out_dict = {}
        for out in outputs:
            if out.shape[1] == len(self.target_classes):
                out_dict["cov"] = out
            if out.shape[1] == len(self.target_classes) * 4:
                out_dict["bbox"] = out
        return out_dict

    def input_preprocessing(self, image):
        """Pre processing an image before preparing the batch."""
        mdl_size = (self.image_width, self.image_height)
        im = image.resize(mdl_size, Image.ANTIALIAS)
        if self.num_channels == 1:
            logger.debug("Converting image from RGB to Grayscale")
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                bg_colour = (255, 255, 255)
                # Need to convert to RGBA if LA format due to a bug in PIL
                alpha = im.convert('RGBA').split()[-1]
                # Create a new background image of our matt color.
                # Must be RGBA because paste requires both images have the same format
                bg = Image.new("RGBA", im.size, bg_colour + (255,))
                bg.paste(im, mask=alpha)
            # Convert image to grayscale.
            im = im.convert('L')
            keras_input = np.asarray(im).astype(np.float32)
            keras_input = keras_input[:, :, np.newaxis]
        elif self.num_channels == 3:
            keras_input = np.asarray(im).astype(np.float32)
        else:
            raise NotImplementedError("Inference can only be run for 1 or 3 channels. "
                                      "Did you forget to run Inference.network_init(). "
                                      "Number of channels: {}".format(self.num_channels))
        keras_input = keras_input.transpose(2, 0, 1) / 255.0
        keras_input.shape = (1, ) + keras_input.shape
        return keras_input, im.size

    def keras_output_map(self, output):
        """Function to map outputs from a cov and bbox to classwise dictionary.

        Realigns outputs from coverage and bbox dictionary to class-wise dictionary of
        coverage and bbox blobs. So now the output dictionary looks like:
        {'class': {'cov': coverage outputs blob of shape [n, 1, output height, output_width],
                  {'bbox': bbox rects outputs blob of shape [n, 4, output height, output_width]}
        }

        Args:
            output (dict): from predictions to dict member function

        Returns:
            out2cluster (dict): output dictionary for bbox post processing
        """
        out2cluster = Dict()
        blobs = list(output.keys())
        target_classes = self.target_classes

        # Separating and reshaping keras outputs blobs to classwise outputs.
        for blob in blobs:
            if 'cov' in blob:
                output_meta_cov = output[blob].transpose(0, 1, 3, 2)
            elif 'bbox' in blob:
                output_meta_bbox = output[blob].transpose(0, 1, 3, 2)
            else:
                raise ValueError('Invalid output blob: cov and bbox expected in output blob names')

        # Remapping output to a nested dictionary.
        for i in range(len(target_classes)):
            key = target_classes[i]
            classwise = Dict()
            for blob_name in blobs:
                if 'cov' in blob_name:
                    classwise['cov'] = output_meta_cov[:, i, :, :]
                elif 'bbox' in blob_name:
                    classwise['bbox'] = output_meta_bbox[:, 4*i: 4*i+4, :, :]
            out2cluster[key] = classwise

        return out2cluster
