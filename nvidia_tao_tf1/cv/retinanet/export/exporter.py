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

"""RetinaNet export model to UFF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import struct
import tempfile
import graphsurgeon as gs

import keras.backend as K
from keras.layers import Permute, Reshape
from keras.models import Model
import tensorflow as tf
import uff

# Import quantization layer processing.
from nvidia_tao_tf1.core.export._quantized import (
    check_for_quantized_layers,
    process_quantized_layers,
)
from nvidia_tao_tf1.core.export._uff import keras_to_pb
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.retinanet.initializers.prior_prob import PriorProbability
from nvidia_tao_tf1.cv.retinanet.layers.anchor_box_layer import RetinaAnchorBoxes

from nvidia_tao_tf1.cv.retinanet.utils.model_io import load_model
from nvidia_tao_tf1.cv.retinanet.utils.spec_loader import load_experiment_spec

NUM_FEATURE_MAPS = 5
logger = logging.getLogger(__name__)


class RetinaNetExporter(Exporter):
    """Exporter class to export a trained RetinaNet model."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 backend="uff",
                 **kwargs):
        """Instantiate the RetinaNet exporter to export etlt model.

        Args:
            model_path(str): Path to the RetinaNet model file.
            key (str): Key to decode the model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            experiment_spec_path (str): Path to RetinaNet experiment spec file.
            backend (str): Type of intermediate backend parser to be instantiated.
        """
        super(RetinaNetExporter, self).__init__(model_path=model_path,
                                                key=key,
                                                data_type=data_type,
                                                strict_type=strict_type,
                                                backend=backend)
        self.experiment_spec_path = experiment_spec_path
        assert os.path.isfile(self.experiment_spec_path), \
            "Experiment spec file not found at {}.".format(self.experiment_spec_path)
        self.experiment_spec = load_experiment_spec(self.experiment_spec_path)
        self.custom_objects = {'RetinaAnchorBoxes': RetinaAnchorBoxes,
                               'PriorProbability': PriorProbability}
        self.tlt2 = False
        self.num_classes = len({str(x) for x in
                               self.experiment_spec.dataset_config.target_class_mapping.values()})

    def load_model(self, backend="uff"):
        """Simple function to load the RetinaNet Keras model."""
        experiment_spec = self.experiment_spec
        K.clear_session()
        K.set_learning_phase(0)
        model = load_model(self.model_path, experiment_spec, key=self.key)
        if model.get_layer('mbox_conf').output.shape[3] == self.num_classes:
            self.tlt2 = True
        outputs = self.generate_trt_output(model.get_layer('mbox_loc').output,
                                           model.get_layer('mbox_conf').output,
                                           model.get_layer('mbox_priorbox').output)
        model = Model(inputs=model.inputs, outputs=outputs)

        if check_for_quantized_layers(model):
            model, self.tensor_scale_dict = process_quantized_layers(
                model, backend,
                calib_cache=None,
                calib_json=None)

            # plugin nodes will have different names in TRT
            nodes = list(self.tensor_scale_dict.keys())
            for k in nodes:
                if k.find('upsample') != -1:
                    node_name_in_trt = k.split('/')[0]
                    self.tensor_scale_dict[node_name_in_trt] = self.tensor_scale_dict.pop(k)

            # ZeroPadding is fused with its following conv2d/depthwiseconv2d, collapse them.
            padding_nodes = []
            for k in self.tensor_scale_dict:
                if '/Pad' in k:
                    # this is a ZeroPadding node
                    padding_nodes.append(k)
            for n in padding_nodes:
                self.tensor_scale_dict.pop(n)

        img_mean = experiment_spec.augmentation_config.image_mean
        self.image_mean = [103.939, 116.779, 123.68] \
            if experiment_spec.augmentation_config.output_channel == 3 else [117.3786]
        if img_mean:
            if experiment_spec.augmentation_config.output_channel == 3:
                self.image_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
            else:
                self.image_mean = [img_mean['l']]
        return model

    def _calibration_cache_from_dict(self, tensor_scale_dict,
                                     calibration_cache=None,
                                     calib_json=None):
        """Write calibration cache file for QAT model.

        This function converts a tensor scale dictionary generated by processing
        QAT models to TRT readable format. By default we set it as a
        trt.IInt8.EntropyCalibrator2 cache file.

        Args:
            tensor_scale_dict (dict): The dictionary of parameters: scale_value file.
            calibration_cache (str): Path to output calibration cache file.

        Returns:
            No explicit returns.
        """
        if calibration_cache is not None:
            cal_cache_str = "TRT-{}-EntropyCalibration2\n".format(self._trt_version_number)
            assert not os.path.exists(calibration_cache), (
                "A pre-existing cache file exists. Please delete this "
                "file and re-run export."
            )
            # Converting float numbers to hex representation.
            for tensor in tensor_scale_dict:
                if tensor in ["P4_upsampled", "P5_upsampled"]:
                    continue
                scaling_factor = tensor_scale_dict[tensor] / 127.0
                cal_scale = hex(struct.unpack("i", struct.pack("f", scaling_factor))[0])
                assert cal_scale.startswith(
                    "0x"), "Hex number expected to start with 0x."
                cal_scale = cal_scale[2:]
                cal_cache_str += tensor + ": " + cal_scale + "\n"
            with open(calibration_cache, "w") as f:
                f.write(cal_cache_str)

        if calib_json is not None:
            calib_json_data = {"tensor_scales": {}}
            for tensor in tensor_scale_dict:
                calib_json_data["tensor_scales"][tensor] = float(
                    tensor_scale_dict[tensor])
            with open(calib_json, "w") as outfile:
                json.dump(calib_json_data, outfile, indent=4)

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
            tmp_uff_file (str): Path to the temporary uff file.
        """
        os_handle, tmp_pb_file = tempfile.mkstemp(suffix=".pb")
        os.close(os_handle)

        if self.backend == "uff":
            keras_to_pb(model, tmp_pb_file, None,
                        custom_objects=self.custom_objects)
            tf.reset_default_graph()
            dynamic_graph = gs.DynamicGraph(tmp_pb_file)
            dynamic_graph = self.node_process(dynamic_graph)

            os.remove(tmp_pb_file)

            uff.from_tensorflow(dynamic_graph.as_graph_def(),
                                self.output_node_names,
                                output_filename=output_file_name,
                                text=False,
                                quiet=True)
            logger.info("Converted model was saved into %s", output_file_name)
            return output_file_name
        raise NotImplementedError("Invalid backend provided. {}".format(self.backend))

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["NMS"]
        self.input_node_names = ["Input"]

    def node_process(self, retinanet_graph):
        """Manipulating the dynamic graph to make it compatible with TRT.

        Args:
            retinanet_graph (gs.DynamicGraph): Dynamic graph from the TF Proto file.

        Returns:
            retinanet_graph (gs.DymanicGraph): Post processed dynamic graph which is ready to be
                serialized as a uff file.
        """
        spec = self.experiment_spec
        FirstDimTile = [
            gs.create_node(name="FirstDimTile_{}".format(i), trt_plugin=True,
                           op="BatchTilePlugin_TRT")
            for i in range(NUM_FEATURE_MAPS)
        ]

        # TensorRT Bug 2603572, anchor_data/Reshape must be at the very beginning!
        if self.tlt2:
            background_id = -1
            num_classes = self.num_classes
        else:
            background_id = 0
            num_classes = self.num_classes + 1

        NMS = gs.create_plugin_node(name='NMS', op='NMS_TRT',
                                    inputs=['anchor_data/Reshape',
                                            'loc_data/Reshape',
                                            'conf_data/Reshape'],
                                    shareLocation=1,
                                    varianceEncodedInTarget=0,
                                    backgroundLabelId=background_id,
                                    confidenceThreshold=spec.nms_config.confidence_threshold,
                                    nmsThreshold=spec.nms_config.clustering_iou_threshold,
                                    topK=2*spec.nms_config.top_k,  # topK as NMS input
                                    codeType=1,
                                    keepTopK=spec.nms_config.top_k,  # NMS output topK
                                    numClasses=num_classes,
                                    inputOrder=[1, 2, 0],
                                    confSigmoid=1,
                                    isNormalized=1,
                                    scoreBits=spec.nms_config.infer_nms_score_bits)

        # Create a mapping of namespace names -> plugin nodes.
        namespace_plugin_map = {"retinanet_anchor_{}/FirstDimTile".format(i): FirstDimTile[i]
                                for i in range(NUM_FEATURE_MAPS)}
        resizenearest_map = {'P4_upsampled': gs.create_plugin_node(name='P4_upsampled',
                                                                   op="ResizeNearest_TRT",
                                                                   scale=2.0),
                             'P5_upsampled': gs.create_plugin_node(name='P5_upsampled',
                                                                   op="ResizeNearest_TRT",
                                                                   scale=2.0)}
        namespace_plugin_map.update(dict(resizenearest_map))
        # Create a new graph by collapsing namespaces
        retinanet_graph.append(NMS)
        retinanet_graph.collapse_namespaces(namespace_plugin_map)
        return retinanet_graph

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        classes = sorted({str(x) for x in
                          self.experiment_spec.dataset_config.target_class_mapping.values()})
        # add background label at idx=0:
        classes = ["background"] + classes
        return classes
