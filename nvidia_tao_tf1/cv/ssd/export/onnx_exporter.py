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

"""SSD export model to encrypted ONNX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tempfile

import keras.backend as K
from keras.layers import Permute, Reshape
from keras.models import Model

import numpy as np
import onnx
import onnx_graphsurgeon as onnx_gs

import tensorflow as tf

from nvidia_tao_tf1.core.export._onnx import keras_to_onnx

# Import quantization layer processing.
from nvidia_tao_tf1.core.export._quantized import (
    check_for_quantized_layers,
    process_quantized_layers,
)
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.ssd.layers.anchor_box_layer import AnchorBoxes
from nvidia_tao_tf1.cv.ssd.utils.model_io import load_model
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import load_experiment_spec

NUM_FEATURE_MAPS = 6

logger = logging.getLogger(__name__)


class SSDOnnxExporter(Exporter):
    """Exporter class to export a trained RetinaNet model."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 backend="onnx",
                 **kwargs):
        """Instantiate the SSD exporter to export a trained SSD .tlt model.

        Args:
            model_path(str): Path to the SSD model file.
            key (str): Key to decode the model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            experiment_spec_path (str): Path to SSD experiment spec file.
            backend (str): Type of intermediate backend parser to be instantiated.
        """
        super(SSDOnnxExporter, self).__init__(model_path=model_path,
                                              key=key,
                                              data_type=data_type,
                                              strict_type=strict_type,
                                              backend=backend)
        self.experiment_spec_path = experiment_spec_path
        assert os.path.isfile(self.experiment_spec_path), \
            "Experiment spec file not found at {}.".format(self.experiment_spec_path)
        self.experiment_spec = None
        self.is_dssd = None
        self.backend = 'onnx'

    def load_model(self, backend="onnx"):
        """Simple function to load the SSD Keras model."""
        experiment_spec, is_dssd = load_experiment_spec(self.experiment_spec_path)
        K.clear_session()
        K.set_learning_phase(0)
        model = load_model(self.model_path, experiment_spec,
                           is_dssd, key=self.key)
        outputs = self.generate_trt_output(model.get_layer('mbox_loc').output,
                                           model.get_layer('mbox_conf_softmax').output,
                                           model.get_layer('mbox_priorbox').output)
        model = Model(inputs=model.input, outputs=outputs)

        if check_for_quantized_layers(model):
            model, tensor_scale_dict = process_quantized_layers(
                model, backend,
                calib_cache=None,
                calib_json=None)

            nodes = list(tensor_scale_dict.keys())
            for k in nodes:
                if k.find('Input') != -1:
                    tensor_scale_dict['Input'] = tensor_scale_dict.pop(k)
            # ZeroPadding is fused with its following conv2d/depthwiseconv2d, collapse them.
            padding_nodes = []
            for k in tensor_scale_dict:
                if '/Pad' in k:
                    # this is a ZeroPadding node
                    padding_nodes.append(k)
            for n in padding_nodes:
                tensor_scale_dict.pop(n)

            self.tensor_scale_dict = tensor_scale_dict
        self.experiment_spec = experiment_spec

        # @tylerz: clear the session and reload the model to remove _1 suffix
        # Save model to a temp file so we can reload it later.
        os_handle, tmp_model_file_name = tempfile.mkstemp(suffix=".hdf5")
        os.close(os_handle)
        model.save(tmp_model_file_name)
        # Make sure Keras session is clean and tuned for inference.
        K.clear_session()
        K.set_learning_phase(0)
        model = load_model(tmp_model_file_name, experiment_spec,
                           is_dssd, key=self.key)
        # Delete temp file.
        os.remove(tmp_model_file_name)

        img_mean = experiment_spec.augmentation_config.image_mean
        self.image_mean = [103.939, 116.779, 123.68] \
            if experiment_spec.augmentation_config.output_channel == 3 else [117.3786]
        if img_mean:
            if experiment_spec.augmentation_config.output_channel == 3:
                self.image_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
            else:
                self.image_mean = [img_mean['l']]

        return model

    def generate_trt_output(self, loc, conf, anchor):
        """Manipulate model outputs so we can use TRT NMS plugin."""

        out_loc = Reshape((-1, 1, 1), name='loc_data')(loc)
        out_conf = Reshape((-1, 1, 1), name='conf_data')(conf)
        out_anchor = Reshape((-1, 2, 4), name="anchor_reshape")(anchor)
        out_anchor = Permute((2, 1, 3), name="anchor_permute")(out_anchor)
        out_anchor = Reshape((2, -1, 1), name='anchor_data')(out_anchor)
        return [out_loc, out_conf, out_anchor]

    def save_exported_file(self, model, output_file_name):
        """Save the exported model file.

        This routine converts a keras model to onnx/uff model
        based on the backend the exporter was initialized with.

        Args:
            model (keras.model.Model): Decoded keras model to be exported.
            output_file_name (str): Path to the output file.

        Returns:
            output_file_name (str): Path to the output ONNX file.
        """
        if self.backend == "onnx":
            keras_to_onnx(model, output_file_name,
                          custom_objects={'AnchorBoxes': AnchorBoxes})
            tf.reset_default_graph()
            onnx_model = onnx.load(output_file_name)
            onnx_model = self.node_process(onnx_model)

            os.remove(output_file_name)
            onnx.save(onnx_model, output_file_name)
            logger.info("Converted model was saved into %s", output_file_name)
            return output_file_name
        raise NotImplementedError("Invalid backend provided. {}".format(self.backend))

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["NMS"]
        self.input_node_names = ["Input"]

    def node_process(self, ssd_graph):
        """Manipulating the dynamic graph to make it compatible with TRT.

        Args:
            ssd_graph (gs.DynamicGraph): Dynamic graph from the TF Proto file.

        Returns:
            ssd_graph (gs.DymanicGraph): Post processed dynamic graph which is ready to be
                serialized as a uff file.
        """
        ssd_graph = onnx_gs.import_onnx(ssd_graph)
        spec = self.experiment_spec

        num_classes = len({str(x) for x in
                           spec.dataset_config.target_class_mapping.values()})

        anchor_data = self._get_onnx_node_by_name(
            ssd_graph, 'anchor_data').outputs[0]
        loc_data = self._get_onnx_node_by_name(
            ssd_graph, 'loc_data').outputs[0]
        conf_data = self._get_onnx_node_by_name(
            ssd_graph, 'conf_data').outputs[0]

        nms_out = onnx_gs.Variable(
            "NMS",
            dtype=np.float32
        )
        nms_out_1 = onnx_gs.Variable(
            "NMS_1",
            dtype=np.float32
        )
        nms_attrs = dict()
        nms_attrs["shareLocation"] = 1
        nms_attrs["varianceEncodedInTarget"] = 0
        nms_attrs["backgroundLabelId"] = 0
        nms_attrs["confidenceThreshold"] = spec.nms_config.confidence_threshold
        nms_attrs["nmsThreshold"] = spec.nms_config.clustering_iou_threshold
        nms_attrs["topK"] = 2*spec.nms_config.top_k
        nms_attrs["codeType"] = 1
        nms_attrs["keepTopK"] = spec.nms_config.top_k
        nms_attrs["numClasses"] = num_classes + 1
        nms_attrs["inputOrder"] = [1, 2, 0]
        nms_attrs["confSigmoid"] = 0
        nms_attrs["isNormalized"] = 1
        nms_attrs["scoreBits"] = spec.nms_config.infer_nms_score_bits
        nms_plugin = onnx_gs.Node(
            op="NMSDynamic_TRT",
            name="NMS",
            inputs=[anchor_data, loc_data, conf_data],
            outputs=[nms_out, nms_out_1],
            attrs=nms_attrs
        )
        ssd_graph.nodes.append(nms_plugin)
        ssd_graph.outputs = nms_plugin.outputs
        ssd_graph.cleanup().toposort()
        self._fix_onnx_paddings(ssd_graph)

        return onnx_gs.export_onnx(ssd_graph)
