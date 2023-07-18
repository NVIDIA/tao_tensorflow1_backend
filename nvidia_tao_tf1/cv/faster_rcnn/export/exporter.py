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
from keras import backend as K
import numpy as np
import onnx
import onnx_graphsurgeon as onnx_gs
import tensorflow as tf

from nvidia_tao_tf1.core.export._onnx import keras_to_onnx
from nvidia_tao_tf1.core.export._uff import keras_to_pb
try:
    from nvidia_tao_tf1.cv.common.export.tensorfile_calibrator import TensorfileCalibrator
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.common.utils import decode_to_keras
try:
    from nvidia_tao_tf1.cv.faster_rcnn.export.faster_rcnn_calibrator import FasterRCNNCalibrator
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
from nvidia_tao_tf1.cv.faster_rcnn.export.utils import (
    _delete_td_reshapes,
    _onnx_delete_td_reshapes,
    _remove_node_input
)
from nvidia_tao_tf1.cv.faster_rcnn.layers.custom_layers import (
    CropAndResize, NmsInputs, OutputParser, Proposal,
    ProposalTarget, TFReshape,
)
from nvidia_tao_tf1.cv.faster_rcnn.models.utils import build_inference_model
from nvidia_tao_tf1.cv.faster_rcnn.patched_uff import patched_uff
from nvidia_tao_tf1.cv.faster_rcnn.qat._quantized import check_for_quantized_layers, \
                                           process_quantized_layers
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader import spec_loader, spec_wrapper

logger = logging.getLogger(__name__)


class FrcnnExporter(Exporter):
    """Exporter class to export a trained FasterRCNN model."""

    def __init__(self,
                 model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 backend="uff",
                 data_format="channels_first",
                 **kwargs):
        """Instantiate the exporter to export a trained FasterRCNN .tlt model.

        Args:
            model_path(str): Path to the .tlt model file.
            key (str): Key to decode the model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            experiment_spec_path (str): Path to the experiment spec file.
            backend (str): Type of intermediate backend parser to be instantiated.
            data_format(str): The keras data format.
        """
        super(FrcnnExporter, self).__init__(model_path=model_path,
                                            key=key,
                                            data_type=data_type,
                                            strict_type=strict_type,
                                            backend=backend,
                                            data_format=data_format,
                                            **kwargs)
        self.experiment_spec_path = experiment_spec_path
        # Exception handling
        assert experiment_spec_path is not None, \
            "Experiment spec file should not be None when exporting a FasterRCNN model."
        assert os.path.isfile(self.experiment_spec_path), \
            "Experiment spec file not found at {}".format(self.experiment_spec_path)
        assert os.path.isfile(model_path), \
            "Model to export is not found at {}".format(model_path)
        self.spec = None
        self.tensor_scale_dict = None

    def load_model(self):
        """Simple function to load the FasterRCNN Keras model."""
        spec = \
            spec_wrapper.ExperimentSpec(spec_loader.load_experiment_spec(self.experiment_spec_path))
        if not(spec.image_h > 0 and spec.image_w > 0):
            raise(
                ValueError(
                    "Exporting a FasterRCNN model with dynamic input shape is not supported."
                )
            )
        self.spec = spec
        K.clear_session()
        K.set_learning_phase(0)
        force_batch_size = self.static_batch_size
        # get the training model
        if isinstance(self.key, str):
            enc_key = self.key.encode()
        else:
            enc_key = self.key
        train_model = decode_to_keras(self.model_path, enc_key, compile_model=False)
        # convert training model to inference model: remove ProposalTarget layer, etc.
        if force_batch_size > 0:
            proposal_force_bs = force_batch_size
        else:
            proposal_force_bs = 1
        config_override = {'pre_nms_top_N': spec.infer_rpn_pre_nms_top_N,
                           'post_nms_top_N': spec.infer_rpn_post_nms_top_N,
                           'nms_iou_thres': spec.infer_rpn_nms_iou_thres,
                           'bs_per_gpu': proposal_force_bs}
        model = build_inference_model(
            train_model,
            config_override,
            max_box_num=spec.infer_rcnn_post_nms_top_N,
            regr_std_scaling=spec.rcnn_regr_std,
            iou_thres=spec.infer_rcnn_nms_iou_thres,
            score_thres=spec.infer_confidence_thres,
            attach_keras_parser=False,
            eval_rois=spec.infer_rpn_post_nms_top_N,
            force_batch_size=force_batch_size
        )
        model.summary()
        if check_for_quantized_layers(model):
            model, self.tensor_scale_dict = process_quantized_layers(model, self.backend)
        return model

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
        # find the base featuremap(input[0] of CropAndResize)
        base_feature_name = None
        for kl in model.layers:
            if kl.name.startswith('crop_and_resize_'):
                inbound_layers = [l for n in kl._inbound_nodes for l in n.inbound_layers]
                assert len(inbound_layers) == 3, 'CropAndResize should have exactly 3 inputs.'
                if self.backend == "uff":
                    base_feature_name = inbound_layers[0].output.op.name
                else:
                    base_feature_name = inbound_layers[0].output.name
                break
        assert (base_feature_name is not None) and (len(base_feature_name) > 0), \
            ''''Base feature map of FasterRCNN model cannot be found,
             please check if the model is a valid FasterRCNN model.'''
        if self.backend == "uff":
            os_handle, tmp_pb_file = tempfile.mkstemp()
            os.close(os_handle)
            custom_objects = {'CropAndResize': CropAndResize,
                              'TFReshape': TFReshape,
                              'Proposal': Proposal,
                              'ProposalTarget': ProposalTarget,
                              'OutputParser': OutputParser,
                              "NmsInputs": NmsInputs}
            keras_to_pb(model, tmp_pb_file, None, custom_objects=custom_objects)
            tf.reset_default_graph()
            dynamic_graph = gs.DynamicGraph(tmp_pb_file)
            dynamic_graph = self.node_process(dynamic_graph, base_feature_name)

            os.remove(tmp_pb_file)
            patched_uff.from_tensorflow(dynamic_graph.as_graph_def(),
                                        self.output_node_names,
                                        output_filename=output_file_name,
                                        text=False,
                                        quiet=True)
            return output_file_name
        if self.backend == "onnx":
            os_handle, tmp_onnx_file = tempfile.mkstemp()
            os.close(os_handle)
            custom_objects = {'CropAndResize': CropAndResize,
                              'TFReshape': TFReshape,
                              'Proposal': Proposal,
                              'ProposalTarget': ProposalTarget,
                              'OutputParser': OutputParser,
                              "NmsInputs": NmsInputs}
            keras_to_onnx(model,
                          tmp_onnx_file,
                          custom_objects=custom_objects,
                          target_opset=self.target_opset)
            tf.reset_default_graph()
            onnx_model = onnx.load(tmp_onnx_file)
            os.remove(tmp_onnx_file)
            new_onnx_model = self.onnx_node_process(
                onnx_model,
                base_feature_name
            )
            onnx.save(new_onnx_model, output_file_name)
            return output_file_name
        raise NotImplementedError("Invalid backend provided. {}".format(self.backend))

    def set_input_output_node_names(self):
        """Set input output node names."""
        if self.backend == "uff":
            self.input_node_names = ["input_image"]
        else:
            self.input_node_names = [""]
        self.output_node_names = ["NMS"]

    def _get_node_by_name(self, onnx_graph, node_name):
        nodes = [n for n in onnx_graph.nodes if n.name == node_name]
        assert len(nodes) == 1, (
            "Expect only 1 node of the name: {}, got {}".format(node_name, len(nodes))
        )
        return nodes[0]

    def _get_node_by_output_name(self, onnx_graph, output_name):
        nodes = [n for n in onnx_graph.nodes if n.outputs[0].name == output_name]
        assert len(nodes) == 1, (
            "Expect only 1 node of the name: {}, got {}".format(output_name, len(nodes))
        )
        return nodes[0]

    def _get_node_by_op(self, onnx_graph, op_name):
        nodes = [n for n in onnx_graph.nodes if n.op == op_name]
        assert len(nodes) == 1, (
            "Expect only 1 node of the op: {}, got {}".format(op_name, len(nodes))
        )
        return nodes[0]

    def onnx_node_process(self, onnx_graph, base_feature_name):
        """Manipulating the onnx graph for plugins."""
        graph = onnx_gs.import_onnx(onnx_graph)
        bfn = self._get_node_by_output_name(graph, base_feature_name).name
        self._onnx_process_proposal(graph, self.spec)
        self._onnx_process_crop_and_resize(graph, self.spec, bfn)
        self._onnx_process_nms(graph, self.spec)
        _onnx_delete_td_reshapes(graph)
        self._fix_paddings(graph)
        graph.cleanup().toposort()
        # apply a filter to tensor_scale_dict with processed onnx graph
        # in case there are some extra tensor scales unpresent in
        # onnx model
        if self.tensor_scale_dict:
            retained_tensor_names = []
            for n in graph.nodes:
                for n_o in n.outputs:
                    if n_o.name not in retained_tensor_names:
                        retained_tensor_names.append(n_o.name)
            dict_names = list(self.tensor_scale_dict.keys())
            for tn in dict_names:
                if tn not in retained_tensor_names:
                    self.tensor_scale_dict.pop(tn)
        return onnx_gs.export_onnx(graph)

    def _fix_paddings(self, graph):
        """Fix the paddings in onnx graph so it aligns with the Keras patch."""
        # third_party/keras/tensorflow_backend.py patched the semantics of
        # SAME padding, the onnx model has to align with it.
        for node in graph.nodes:
            if node.op == "Conv":
                # in case of VALID padding, there is no 'pads' attribute
                # simply skip it
                if node.attrs["auto_pad"] == "VALID":
                    continue
                k = node.attrs['kernel_shape']
                g = node.attrs['group']
                d = node.attrs['dilations']
                # always assume kernel shape is square
                effective_k = [1 + (k[ki] - 1) * d[ki] for ki in range(len(d))]
                # (pad_w // 2 , pad_h // 2) == (pad_left, pad_top)
                keras_paddings = tuple((ek - 1) // 2 for ek in effective_k)
                # (pad_left, pad_top, pad_right, pad_bottom)
                if g == 1:
                    # if it is not VALID, then it has to be NOTSET,
                    # to enable explicit paddings below
                    node.attrs["auto_pad"] = "NOTSET"
                    # only apply this patch for non-group convolutions
                    node.attrs['pads'] = keras_paddings * 2
            elif node.op in ["AveragePool", "MaxPool"]:
                # skip VALID padding case.
                if node.attrs["auto_pad"] == "VALID":
                    continue
                k = node.attrs['kernel_shape']
                # (pad_w // 2 , pad_h // 2) == (pad_left, pad_top)
                keras_paddings = tuple((ek - 1) // 2 for ek in k)
                # force it to be NOTSET to enable explicit paddings below
                node.attrs["auto_pad"] = "NOTSET"
                # (pad_left, pad_top, pad_right, pad_bottom)
                node.attrs['pads'] = keras_paddings * 2

    def _onnx_process_proposal(self, graph, spec):
        for node in graph.nodes:
            if node.name == "proposal_1/packed:0_shape":
                roi_shape_0_node = node
                continue
            if node.name.startswith("proposal_1") or node.name.startswith("_proposal_1"):
                node.outputs.clear()

        rpn_out_regress_node = self._get_node_by_name(
            graph, "rpn_out_regress"
        )
        # reconnect rois shape[0](batch size) to an exisiting node
        # e.g., here: rpn_out_regress_node
        if self.static_batch_size <= 0:
            roi_shape_0_node.inputs = [rpn_out_regress_node.outputs[0]]
        rpn_out_class_node = self._get_node_by_name(
            graph, "rpn_out_class"
        )
        proposal_out = onnx_gs.Variable(
            "proposal_out",
            dtype=np.float32
        )
        proposal_attrs = dict()
        proposal_attrs["input_height"] = int(spec.image_h)
        proposal_attrs["input_width"] = int(spec.image_w)
        proposal_attrs["rpn_stride"] = int(spec.rpn_stride)
        proposal_attrs["roi_min_size"] = 1.0
        proposal_attrs["nms_iou_threshold"] = spec.infer_rpn_nms_iou_thres
        proposal_attrs["pre_nms_top_n"] = int(spec.infer_rpn_pre_nms_top_N)
        proposal_attrs["post_nms_top_n"] = int(spec.infer_rpn_post_nms_top_N)
        proposal_attrs["anchor_sizes"] = spec.anchor_sizes
        proposal_attrs["anchor_ratios"] = spec.anchor_ratios
        Proposal_plugin = onnx_gs.Node(
            op="ProposalDynamic",
            name="proposal",
            inputs=[
                rpn_out_class_node.outputs[0],
                rpn_out_regress_node.outputs[0]
            ],
            outputs=[proposal_out],
            attrs=proposal_attrs
        )
        roi_reshape_node = self._get_node_by_name(
            graph,
            "nms_inputs_1/Reshape_reshape"
        )
        roi_reshape_node.inputs = [Proposal_plugin.outputs[0], roi_reshape_node.inputs[1]]
        graph.nodes.append(Proposal_plugin)
        graph.cleanup().toposort()
        # insert missing Sigmoid node for rpn_out_class
        sigmoid_output = onnx_gs.Variable(
            "sigmoid_output",
            dtype=np.float32
        )
        rpn_class_sigmoid_node = onnx_gs.Node(
            op="Sigmoid",
            name="rpn_out_class/Sigmoid",
            inputs=rpn_out_class_node.outputs,
            outputs=[sigmoid_output]
        )
        Proposal_plugin.inputs = [sigmoid_output, Proposal_plugin.inputs[1]]
        graph.nodes.append(rpn_class_sigmoid_node)
        graph.cleanup().toposort()

    def _onnx_process_crop_and_resize(self, graph, spec, base_feature_name):
        pool_size = spec.roi_pool_size
        if spec.roi_pool_2x:
            pool_size *= 2
        # crop_and_resize plugin
        base_feature_node = self._get_node_by_name(graph, base_feature_name)
        crop_and_resize_out = onnx_gs.Variable(
            "crop_and_resize_out",
            dtype=np.float32
        )
        crop_and_resize_attrs = dict()
        crop_and_resize_attrs["crop_height"] = pool_size
        crop_and_resize_attrs["crop_width"] = pool_size
        Proposal_plugin = self._get_node_by_op(graph, "ProposalDynamic")
        CropAndResize_plugin = onnx_gs.Node(
            op="CropAndResizeDynamic",
            name="CropAndResize_plugin",
            inputs=[
                base_feature_node.outputs[0],
                Proposal_plugin.outputs[0]
            ],
            outputs=[crop_and_resize_out],
            attrs=crop_and_resize_attrs
        )
        graph.nodes.append(CropAndResize_plugin)
        crop_and_resize_old_output_node = self._get_node_by_name(
            graph,
            "crop_and_resize_1/Reshape_1_reshape"
        )
        CropAndResize_plugin.outputs = crop_and_resize_old_output_node.outputs
        for node in graph.nodes:
            if (
                node.name.startswith("crop_and_resize_1") or
                node.name.startswith("_crop_and_resize_1")
            ):
                # fix corner case for googlenet where the next pooling
                # somehow has name with crop_and_resize_1 in it
                if "pooling" not in node.name:
                    node.outputs.clear()
        graph.cleanup().toposort()

    def _onnx_process_nms(self, graph, spec):
        prior_data_node = self._get_node_by_name(graph, "nms_inputs_1/prior_data_concat")
        loc_data_node = self._get_node_by_name(graph, "nms_inputs_1/loc_data_reshape")
        conf_data_node = self._get_node_by_name(graph, "nms_inputs_1/conf_data_reshape")
        nms_out = onnx_gs.Variable(
            "nms_out",
            dtype=np.float32
        )
        nms_out_1 = onnx_gs.Variable(
            "nms_out_1",
            dtype=np.float32
        )
        nms_attrs = dict()
        nms_attrs["shareLocation"] = 0
        nms_attrs["varianceEncodedInTarget"] = 1
        nms_attrs["backgroundLabelId"] = spec.num_classes - 1
        nms_attrs["confidenceThreshold"] = self.spec.infer_confidence_thres
        nms_attrs["nmsThreshold"] = spec.infer_rcnn_nms_iou_thres
        nms_attrs["topK"] = spec.infer_rpn_post_nms_top_N
        nms_attrs["codeType"] = 1
        nms_attrs["keepTopK"] = spec.infer_rcnn_post_nms_top_N
        nms_attrs["numClasses"] = spec.num_classes
        nms_attrs["inputOrder"] = [1, 2, 0]
        nms_attrs["confSigmoid"] = 0
        nms_attrs["isNormalized"] = 1
        nms_attrs["scoreBits"] = spec.infer_nms_score_bits
        NMS_plugin = onnx_gs.Node(
            op="NMSDynamic_TRT",
            name="NMS",
            inputs=[
                prior_data_node.outputs[0],
                loc_data_node.outputs[0],
                conf_data_node.outputs[0]
            ],
            outputs=[nms_out, nms_out_1],
            attrs=nms_attrs
        )
        graph.nodes.append(NMS_plugin)
        # delete reshape op in the TimeDistributed layers
        graph.outputs = NMS_plugin.outputs
        graph.cleanup().toposort()

    def node_process(self, dynamic_graph, base_feature_name):
        """Manipulating the dynamic graph to make it compatible with TRT."""
        spec = self.spec
        # create TRT plugin nodes
        pool_size = spec.roi_pool_size
        if spec.roi_pool_2x:
            pool_size *= 2
        CropAndResize_plugin = \
            gs.create_plugin_node(name='roi_pooling_conv_1/CropAndResize_new',
                                  op="CropAndResize",
                                  inputs=[base_feature_name,
                                          'proposal'],
                                  crop_height=pool_size,
                                  crop_width=pool_size)
        Proposal_plugin = \
            gs.create_plugin_node(name='proposal',
                                  op='Proposal',
                                  inputs=['rpn_out_class/Sigmoid',
                                          'rpn_out_regress/BiasAdd'],
                                  input_height=int(spec.image_h),
                                  input_width=int(spec.image_w),
                                  rpn_stride=int(spec.rpn_stride),
                                  roi_min_size=1.0,
                                  nms_iou_threshold=spec.infer_rpn_nms_iou_thres,
                                  pre_nms_top_n=int(spec.infer_rpn_pre_nms_top_N),
                                  post_nms_top_n=int(spec.infer_rpn_post_nms_top_N),
                                  anchor_sizes=spec.anchor_sizes,
                                  anchor_ratios=spec.anchor_ratios)
        # isNormalized is True because the Proposal plugin always normalizes the coordinates
        NMS = gs.create_plugin_node(name='NMS', op='NMS_TRT',
                                    inputs=["nms_inputs_1/prior_data",
                                            'nms_inputs_1/loc_data',
                                            'nms_inputs_1/conf_data'],
                                    shareLocation=0,
                                    varianceEncodedInTarget=1,
                                    backgroundLabelId=self.spec.num_classes - 1,
                                    confidenceThreshold=self.spec.infer_confidence_thres,
                                    nmsThreshold=self.spec.infer_rcnn_nms_iou_thres,
                                    topK=self.spec.infer_rpn_post_nms_top_N,  # topK as NMS input
                                    codeType=1,
                                    keepTopK=self.spec.infer_rcnn_post_nms_top_N,  # NMS output topK
                                    numClasses=self.spec.num_classes,
                                    inputOrder=[1, 2, 0],
                                    confSigmoid=0,
                                    isNormalized=1,
                                    scoreBits=spec.infer_nms_score_bits,
                                    # FasterRCNN takes RoI as inputs and they
                                    # differs per image, so we should set it to
                                    # False. By default, this parameter is True
                                    # It was introduced start from OSS 21.06
                                    # This issue should only impact UFF but not ONNX
                                    # see: commit: a2b3d3d5cc9cd79c84dffc1b82b5439442cde201
                                    isBatchAgnostic=False)
        namespace_plugin_map = {
            "crop_and_resize_1" : CropAndResize_plugin,
            "proposal_1": Proposal_plugin,
        }
        # replace Tensorflow op with plugin nodes
        dynamic_graph.collapse_namespaces(namespace_plugin_map)
        _remove_node_input(dynamic_graph, "roi_pooling_conv_1/CropAndResize_new", 2)
        _remove_node_input(dynamic_graph, "proposal", 2)
        # delete reshape op in the TimeDistributed layers
        _delete_td_reshapes(dynamic_graph)
        dynamic_graph.append(NMS)
        return dynamic_graph

    def set_data_preprocessing_parameters(self, input_dims, image_mean=None):
        """Simple function to set data preprocessing parameters."""
        # In FasterRCNN, we have configurable image scaling, means and channel order
        # setup image scaling factor and per-channel mean values
        image_config = self.spec.model_config.input_image_config
        scale = 1.0 / image_config.image_scaling_factor
        means = image_config.image_channel_mean
        _num_channels = 3 if (image_config.image_type == 0) else 1
        if _num_channels == 3:
            means = [means['r'], means['g'], means['b']]
        else:
            means = [means['l']]
        if image_config.image_channel_order == 'bgr':
            flip_channel = True
            means = means[::-1]
        else:
            flip_channel = False
        self.preprocessing_arguments = {"scale": scale,
                                        "means": means,
                                        "flip_channel": flip_channel}

    def get_calibrator(self,
                       calibration_cache,
                       data_file_name,
                       n_batches,
                       batch_size,
                       input_dims,
                       calibration_images_dir=None,
                       image_mean=None):
        """Simple function to get an int8 calibrator.

        Args:
            calibration_cache (str): Path to store the int8 calibration cache file.
            data_file_name (str): Path to the TensorFile. If the tensorfile doesn't exist
                at this path, then one is created with either n_batches of random tensors,
                images from the file in calibration_images_dir of dimensions
                (batch_size,) + (input_dims)
            n_batches (int): Number of batches to calibrate the model over.
            batch_size (int): Number of input tensors per batch.
            input_dims (tuple): Tuple of input tensor dimensions in CHW order.
            calibration_images_dir (str): Path to a directory of images to generate the
                data_file from.
            image_mean (list): Image mean values.

        Returns:
            calibrator(TensorfileCalibrator or FasterRCNNCalibrator):
                TRTEntropyCalibrator2 instance to calibrate the TensorRT engine.
        """
        if data_file_name and os.path.exists(data_file_name):
            logger.info("Using existing tensor file for INT8 calibration.")
            calibrator = TensorfileCalibrator(data_file_name,
                                              calibration_cache,
                                              n_batches,
                                              batch_size)
        elif data_file_name:
            if (calibration_images_dir and os.path.exists(calibration_images_dir)):
                logger.info("Generating tensor file from image directory and"
                            " then use the tensor file for INT8 calibration.")
            self.generate_tensor_file(data_file_name,
                                      calibration_images_dir,
                                      input_dims,
                                      n_batches=n_batches,
                                      batch_size=batch_size)
            calibrator = TensorfileCalibrator(data_file_name,
                                              calibration_cache,
                                              n_batches,
                                              batch_size)
        else:
            logger.info("Using data loader to generate the data for INT8 calibration.")
            # default to use data loader if neither tensorfile nor images is provided
            calibrator = FasterRCNNCalibrator(
                self.spec,
                calibration_cache,
                n_batches,
                batch_size)

        return calibrator

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        if self.spec is None:
            raise AttributeError(
                "Experiment spec wasn't loaded. To get class labels "
                "please provide the experiment spec file using the -e "
                "option.")
        target_classes = self.spec.class_mapping.values()
        target_classes = sorted(set(target_classes))
        target_classes.append('background')
        return target_classes
