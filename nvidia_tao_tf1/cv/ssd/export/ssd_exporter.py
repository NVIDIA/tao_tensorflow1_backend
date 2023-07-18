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

import logging
import os
import tempfile
import graphsurgeon as gs

import keras
from keras import backend as K
from keras.layers import Permute, Reshape

import tensorflow as tf

import uff

# Import quantization layer processing.
from nvidia_tao_tf1.core.export._quantized import (
    check_for_quantized_layers,
    process_quantized_layers,
)
from nvidia_tao_tf1.core.export._uff import keras_to_pb
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.common.types.base_ds_config import BaseDSConfig
from nvidia_tao_tf1.cv.ssd.layers.anchor_box_layer import AnchorBoxes
from nvidia_tao_tf1.cv.ssd.utils.model_io import load_model
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import EXPORT_EXP_REQUIRED_MSG, \
    load_experiment_spec, spec_validator

NUM_FEATURE_MAPS = 6

logger = logging.getLogger(__name__)


class SSDExporter(Exporter):
    """Exporter class to export a trained SSD model."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 backend="uff",
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
        super(SSDExporter, self).__init__(model_path=model_path,
                                          key=key,
                                          data_type=data_type,
                                          strict_type=strict_type,
                                          backend=backend)
        self.experiment_spec_path = experiment_spec_path
        assert os.path.isfile(self.experiment_spec_path), \
            "Experiment spec file not found at {}.".format(self.experiment_spec_path)
        self.experiment_spec = None
        self.is_dssd = None

    def load_model(self, backend="uff"):
        """Simple function to load the SSD Keras model."""
        experiment_spec, is_dssd = load_experiment_spec(self.experiment_spec_path)
        spec_validator(experiment_spec, EXPORT_EXP_REQUIRED_MSG)
        K.clear_session()
        K.set_learning_phase(0)
        model = load_model(self.model_path, experiment_spec,
                           is_dssd, key=self.key)
        outputs = self.generate_trt_output(model.get_layer('mbox_loc').output,
                                           model.get_layer('mbox_conf_softmax').output,
                                           model.get_layer('mbox_priorbox').output)
        model = keras.models.Model(inputs=model.input,
                                   outputs=outputs)

        if check_for_quantized_layers(model):
            model, tensor_scale_dict = process_quantized_layers(
                model, backend,
                calib_cache=None,
                calib_json=None)

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
            output_file_name (str): Path to the output UFF model.
        """
        os_handle, tmp_pb_file = tempfile.mkstemp(suffix=".pb")
        os.close(os_handle)

        if self.backend == "uff":
            keras_to_pb(model, tmp_pb_file, None,
                        custom_objects={'AnchorBoxes': AnchorBoxes})
            tf.reset_default_graph()
            dynamic_graph = gs.DynamicGraph(tmp_pb_file)
            dynamic_graph = self.node_process(dynamic_graph)

            os.remove(tmp_pb_file)
            uff.from_tensorflow(dynamic_graph.as_graph_def(),
                                ['NMS'],
                                output_filename=output_file_name,
                                text=False,
                                quiet=True)
            return output_file_name
        raise NotImplementedError("Invalid backend provided. {}".format(self.backend))

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["NMS"]
        self.input_node_names = ["Input"]

    def node_process(self, ssd_graph):
        """Manipulating the ssd dynamic graph to make it compatible with TRT.

        Args:
            ssd_graph (gs.DynamicGraph): Dynamic graph of the SSD model from the TF Proto file.

        Returns:
            ssd_graph (gs.DymanicGraph): Post processed dynamic graph which is ready to be
                serialized as a uff file.
        """
        spec = self.experiment_spec
        FirstDimTile = [
            gs.create_node(name="FirstDimTile_{}".format(i),
                           trt_plugin=True,
                           op="BatchTilePlugin_TRT")
            for i in range(NUM_FEATURE_MAPS)
        ]

        num_classes = len({str(x) for x in
                           spec.dataset_config.target_class_mapping.values()})

        # TensorRT Bug 2603572, anchor_data/Reshape must be at the very beginning!
        NMS = gs.create_plugin_node(name='NMS', op='NMS_TRT',
                                    inputs=['anchor_data/Reshape',
                                            'loc_data/Reshape',
                                            'conf_data/Reshape'],
                                    shareLocation=1,
                                    varianceEncodedInTarget=0,
                                    backgroundLabelId=0,
                                    confidenceThreshold=spec.nms_config.confidence_threshold,
                                    nmsThreshold=spec.nms_config.clustering_iou_threshold,
                                    topK=2*spec.nms_config.top_k,  # topK as NMS input
                                    codeType=1,
                                    keepTopK=spec.nms_config.top_k,  # NMS output topK
                                    numClasses=num_classes+1,  # +1 for background class
                                    inputOrder=[1, 2, 0],
                                    confSigmoid=0,
                                    isNormalized=1,
                                    scoreBits=spec.nms_config.infer_nms_score_bits)

        # Create a mapping of namespace names -> plugin nodes.
        namespace_plugin_map = {"ssd_anchor_{}/FirstDimTile".format(i): FirstDimTile[i] for i in
                                range(NUM_FEATURE_MAPS)}

        softmax_remove_list = ["mbox_conf_softmax_/transpose",
                               "mbox_conf_softmax_/transpose_1"]

        softmax_connect_list = [("mbox_conf_softmax_/Softmax", "mbox_conf_softmax/transpose"),
                                ("before_softmax_permute/transpose", "mbox_conf_softmax_/Softmax")]

        def connect(dynamic_graph, connections_list):

            for node_a_name, node_b_name in connections_list:
                if node_a_name not in dynamic_graph.node_map[node_b_name].input:
                    dynamic_graph.node_map[node_b_name].input.insert(0, node_a_name)

        # Create a new graph by collapsing namespaces
        ssd_graph.remove(softmax_remove_list)
        connect(ssd_graph, softmax_connect_list)

        ssd_graph.append(NMS)
        ssd_graph.collapse_namespaces(namespace_plugin_map)
        return ssd_graph

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        classes = sorted({str(x) for x in
                          self.experiment_spec.dataset_config.target_class_mapping.values()})
        # add background label at idx=0:
        classes = ["background"] + classes
        return classes

    def generate_ds_config(self, input_dims, num_classes=None):
        """Generate Deepstream config element for the exported model."""
        if input_dims[0] == 1:
            color_format = "l"
        else:
            color_format = "bgr" if self.preprocessing_arguments["flip_channel"] else "rgb"
        kwargs = {
            "data_format": self.data_format,
            "backend": self.backend,
            # Setting this to 0 by default because there are more
            # detection networks.
            "network_type": 0,
            "maintain_aspect_ratio": False
        }
        if num_classes:
            kwargs["num_classes"] = num_classes
        if self.backend == "uff":
            kwargs.update({
                "input_names": self.input_node_names,
                "output_names": self.output_node_names
            })

        ds_config = BaseDSConfig(
            self.preprocessing_arguments["scale"],
            self.preprocessing_arguments["means"],
            input_dims,
            color_format,
            self.key,
            **kwargs
        )
        return ds_config
