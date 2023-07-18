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

"""Base class to export trained .tlt models to etlt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from keras import backend as K
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
from nvidia_tao_tf1.core.export._uff import _reload_model_for_inference
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.yolo_v3.layers.export_layers import BoxLayer, ClsLayer
from nvidia_tao_tf1.cv.yolo_v3.layers.yolo_anchor_box_layer import YOLOAnchorBox
from nvidia_tao_tf1.cv.yolo_v3.utils.model_io import load_model
from nvidia_tao_tf1.cv.yolo_v3.utils.spec_loader import load_experiment_spec


CUSTOM_OBJ = {'YOLOAnchorBox': YOLOAnchorBox,
              'BoxLayer': BoxLayer,
              'ClsLayer': ClsLayer}


class YOLOv3Exporter(Exporter):
    """Exporter class to export a trained yolo model."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 backend="onnx",
                 **kwargs):
        """Instantiate the yolo exporter to export a trained yolo .tlt model.

        Args:
            model_path(str): Path to the yolo model file.
            key (str): Key to decode the model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            experiment_spec_path (str): Path to yolo experiment spec file.
            backend (str): Type of intermediate backend parser to be instantiated.
        """
        super(YOLOv3Exporter, self).__init__(model_path=model_path,
                                             key=key,
                                             data_type=data_type,
                                             strict_type=strict_type,
                                             backend=backend,
                                             **kwargs)
        self.experiment_spec_path = experiment_spec_path
        assert os.path.isfile(self.experiment_spec_path), \
            "Experiment spec file not found at {}.".format(self.experiment_spec_path)
        self.experiment_spec = None

    def load_model(self, backend="onnx"):
        """Simple function to load the yolo Keras model."""
        experiment_spec = load_experiment_spec(self.experiment_spec_path)
        K.clear_session()
        K.set_learning_phase(0)
        img_channel = experiment_spec.augmentation_config.output_channel
        img_height = experiment_spec.augmentation_config.output_height
        img_width = experiment_spec.augmentation_config.output_width
        model = load_model(
            self.model_path,
            experiment_spec,
            (img_channel, img_height, img_width),
            key=self.key
        )
        last_layer_out = model.layers[-1].output
        r_boxes = BoxLayer(name="box")(last_layer_out)
        r_cls = ClsLayer(name="cls")(last_layer_out)
        model = Model(inputs=model.inputs, outputs=[r_boxes, r_cls])

        if check_for_quantized_layers(model):
            model, self.tensor_scale_dict = process_quantized_layers(
                model, backend,
                calib_cache=None,
                calib_json=None)

            # plugin nodes will have different names in TRT
            nodes = list(self.tensor_scale_dict.keys())
            for k in nodes:
                if k.find('Input') != -1:
                    self.tensor_scale_dict['Input'] = self.tensor_scale_dict.pop(k)

            # ZeroPadding is fused with its following conv2d/depthwiseconv2d, collapse them.
            padding_nodes = []
            for k in self.tensor_scale_dict:
                if '/Pad' in k:
                    # this is a ZeroPadding node
                    padding_nodes.append(k)
            for n in padding_nodes:
                self.tensor_scale_dict.pop(n)

        self.experiment_spec = experiment_spec
        img_mean = experiment_spec.augmentation_config.image_mean
        self.image_mean = [103.939, 116.779, 123.68] \
            if experiment_spec.augmentation_config.output_channel == 3 else [117.3786]
        if img_mean:
            if experiment_spec.augmentation_config.output_channel == 3:
                self.image_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
            else:
                self.image_mean = [img_mean['l']]
        # @zeyuz: must reload so tensorname won't have _1 suffix
        model = _reload_model_for_inference(model, custom_objects=CUSTOM_OBJ)

        return model

    def save_exported_file(self, model, output_file_name):
        """Save the exported model file.

        This routine converts a keras model to onnx/uff model
        based on the backend the exporter was initialized with.

        Args:
            model (keras.model.Model): Decoded keras model to be exported.
            output_file_name (str): Path to the output file.

        Returns:
            output_file_name (str): Path to the onnx file.
        """
        if self.backend == "onnx":
            keras_to_onnx(model,
                          output_file_name,
                          custom_objects=CUSTOM_OBJ,
                          target_opset=self.target_opset)
            tf.reset_default_graph()
            onnx_model = onnx.load(output_file_name)
            onnx_model = self.node_process(onnx_model)

            os.remove(output_file_name)
            onnx.save(onnx_model, output_file_name)
            return output_file_name

        # @zeyuz: UFF export not supported in YOLOv3 due to NvBug200697725
        raise NotImplementedError("Invalid backend provided. {}".format(self.backend))

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["BatchedNMS"]
        self.input_node_names = ["Input"]

    def process_nms_node(self, onnx_graph):
        """Process the NMS ONNX node."""

        spec = self.experiment_spec
        box_data = self._get_onnx_node_by_name(onnx_graph, 'box/concat_concat').outputs[0]
        cls_data = self._get_onnx_node_by_name(onnx_graph, 'cls/mul').outputs[0]

        nms_out_0 = onnx_gs.Variable(
            "BatchedNMS",
            dtype=np.int32
        )
        nms_out_1 = onnx_gs.Variable(
            "BatchedNMS_1",
            dtype=np.float32
        )
        nms_out_2 = onnx_gs.Variable(
            "BatchedNMS_2",
            dtype=np.float32
        )
        nms_out_3 = onnx_gs.Variable(
            "BatchedNMS_3",
            dtype=np.float32
        )

        nms_attrs = dict()

        nms_attrs["shareLocation"] = 1
        nms_attrs["backgroundLabelId"] = -1
        nms_attrs["scoreThreshold"] = spec.nms_config.confidence_threshold
        nms_attrs["iouThreshold"] = spec.nms_config.clustering_iou_threshold
        nms_attrs["topK"] = 2*spec.nms_config.top_k
        nms_attrs["keepTopK"] = spec.nms_config.top_k
        nms_attrs["numClasses"] = len(
            {str(x) for x in spec.dataset_config.target_class_mapping.values()}
        )
        nms_attrs["clipBoxes"] = 1
        nms_attrs["isNormalized"] = 1
        nms_attrs["scoreBits"] = spec.nms_config.infer_nms_score_bits

        nms_plugin = onnx_gs.Node(
            op="BatchedNMSDynamic_TRT",
            name="BatchedNMS_N",
            inputs=[box_data, cls_data],
            outputs=[nms_out_0, nms_out_1, nms_out_2, nms_out_3],
            attrs=nms_attrs
        )

        onnx_graph.nodes.append(nms_plugin)
        onnx_graph.outputs = nms_plugin.outputs
        onnx_graph.cleanup().toposort()

    def node_process(self, yolo_graph):
        """Manipulating the yolo dynamic graph to make it compatible with TRT.

        Args:
            yolo_graph (onnx_gs.DynamicGraph): Dynamic graph of the yolo model from the TF Proto
                file.

        Returns:
            yolo_graph (onnx_gs.DynamicGraph): Post processed dynamic graph which is ready to be
                serialized as a ONNX file.
        """

        graph = onnx_gs.import_onnx(yolo_graph)
        self.process_nms_node(graph)
        self._fix_onnx_paddings(graph)
        return onnx_gs.export_onnx(graph)

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        classes = sorted({str(x) for x in
                          self.experiment_spec.dataset_config.target_class_mapping.values()})
        return classes
