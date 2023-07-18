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
"""MaskRCNN export model to UFF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import struct
import tempfile
from google.protobuf.json_format import MessageToDict
import graphsurgeon as gs
from numba import cuda
from pycocotools.coco import COCO
import six

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util
import uff

from nvidia_tao_tf1.cv.common.export.base_exporter import BaseExporter as Exporter
from nvidia_tao_tf1.cv.common.export.trt_utils import UFFEngineBuilder
from nvidia_tao_tf1.cv.common.types.base_ds_config import BaseDSConfig
from nvidia_tao_tf1.cv.mask_rcnn.dataloader import dataloader
from nvidia_tao_tf1.cv.mask_rcnn.executer.distributed_executer import BaseExecuter
from nvidia_tao_tf1.cv.mask_rcnn.export import converter_functions
from nvidia_tao_tf1.cv.mask_rcnn.export.DynamicGraph import DynamicGraph
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import mask_rcnn_params
from nvidia_tao_tf1.cv.mask_rcnn.hyperparameters import params_io
from nvidia_tao_tf1.cv.mask_rcnn.models import mask_rcnn_model
from nvidia_tao_tf1.cv.mask_rcnn.utils.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.mask_rcnn.utils.model_loader import load_mrcnn_tlt_model
from nvidia_tao_tf1.encoding import encoding

gs.DynamicGraph = DynamicGraph
uff.converters.tensorflow.converter_functions = converter_functions
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


class MaskRCNNExporter(Exporter):
    """Exporter class to export a trained MaskRCNN model."""

    def __init__(self,
                 model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 data_format="channels_first",
                 backend="uff",
                 **kwargs):
        """Instantiate the MaskRCNN exporter to export etlt model.

        Args:
            model_path(str): Path to the MaskRCNN model file.
            key (str): Key to decode the model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            experiment_spec_path (str): Path to MaskRCNN experiment spec file.
            backend (str): Type of intermediate backend parser to be instantiated.
        """
        super(MaskRCNNExporter, self).__init__(model_path=model_path,
                                               key=key,
                                               data_type=data_type,
                                               strict_type=strict_type,
                                               backend=backend)
        self.experiment_spec_path = experiment_spec_path
        assert os.path.isfile(self.experiment_spec_path), \
            "Experiment spec file not found at {}.".format(self.experiment_spec_path)
        self.experiment_spec = load_experiment_spec(experiment_spec_path)
        self.key = key
        self.backend = backend
        self.data_type = data_type
        self._temp_dir = tempfile.mkdtemp()
        self.data_format = data_format
        assert self.data_format in ["channels_first", "channels_last"], (
            "Invalid data format encountered: {}".format(self.data_format)
        )

    def load_model(self):
        """Load model."""
        raise NotImplementedError("No maskrcnn keras model provided")

    def _extract_ckpt(self, encoded_checkpoint, key):
        """Get unencrypted checkpoint from tlt file."""
        logging.info("Loading weights from {}".format(encoded_checkpoint))
        checkpoint_path = load_mrcnn_tlt_model(
            encoded_checkpoint,
            key
        )
        return checkpoint_path

    def _generate_maskrcnn_config(self):
        """Generate MaskRCNN config."""

        maskrcnn_param = mask_rcnn_params.default_config()
        temp_config = MessageToDict(self.experiment_spec,
                                    preserving_proto_field_name=True,
                                    including_default_value_fields=True)
        try:
            data_config = temp_config['data_config']
            maskrcnn_config = temp_config['maskrcnn_config']
        except ValueError:
            print("Make sure data_config and maskrcnn_config are configured properly.")
        finally:
            del temp_config['data_config']
            del temp_config['maskrcnn_config']
        temp_config.update(data_config)
        temp_config.update(maskrcnn_config)
        # eval some string type params
        if 'freeze_blocks' in temp_config:
            temp_config['freeze_blocks'] = eval_str(temp_config['freeze_blocks'])
        if 'image_size' in temp_config:
            temp_config['image_size'] = eval_str(temp_config['image_size'])
        else:
            raise ValueError("image_size is not set.")
        if 'learning_rate_steps' in temp_config:
            temp_config['learning_rate_steps'] = eval_str(temp_config['learning_rate_steps'])
        if 'learning_rate_decay_levels' in temp_config:
            temp_config['learning_rate_levels'] = \
                [decay * temp_config['init_learning_rate']
                    for decay in eval_str(temp_config['learning_rate_decay_levels'])]
        if 'bbox_reg_weights' in temp_config:
            temp_config['bbox_reg_weights'] = eval_str(temp_config['bbox_reg_weights'])
        if 'aspect_ratios' in temp_config:
            temp_config['aspect_ratios'] = eval_str(temp_config['aspect_ratios'])
        # force some params to default value
        temp_config['use_fake_data'] = False
        temp_config['allow_xla_at_inference'] = False
        # Force freeze_bn to True during export
        temp_config['freeze_bn'] = True
        if 'num_steps_per_eval' in temp_config:
            temp_config['save_checkpoints_steps'] = temp_config['num_steps_per_eval']
        else:
            raise ValueError("num_steps_per_eval is not set.")

        # load model from json graphs in the same dir as the checkpoint
        temp_config['pruned_model_path'] = os.path.dirname(self.model_path)
        # use experiment spec to overwrite default hparams
        maskrcnn_param = params_io.override_hparams(maskrcnn_param, temp_config)
        return maskrcnn_param

    def _generate_estimator_run_config(self, maskrcnn_param):
        """Estimator run_config."""

        run_config = tf.estimator.RunConfig(
            tf_random_seed=(
                maskrcnn_param.seed
            ),
            save_summary_steps=None,  # disabled
            save_checkpoints_steps=None,  # disabled
            save_checkpoints_secs=None,  # disabled
            keep_checkpoint_max=20,  # disabled
            keep_checkpoint_every_n_hours=None,  # disabled
            log_step_count_steps=None,  # disabled
            session_config=BaseExecuter._get_session_config(
                mode="eval",
                use_xla=maskrcnn_param.use_xla,
                use_amp=maskrcnn_param.use_amp,
                use_tf_distributed=False,
                # TODO: Remove when XLA at inference fixed
                allow_xla_at_inference=maskrcnn_param.allow_xla_at_inference
            ),
            protocol=None,
            device_fn=None,
            train_distribute=None,
            eval_distribute=None,
            experimental_distribute=None
        )

        return run_config

    def _train_ckpt_to_eval_ckpt(self, decoded_ckpt_path):
        """Convert train ckpt to eval ckpt."""

        tmp_eval_ckpt_path = tempfile.mkdtemp()
        maskrcnn_param = self._generate_maskrcnn_config()
        estimator_runconfig = self._generate_estimator_run_config(maskrcnn_param)
        tmp_input_fn = dataloader.InputReader(
            file_pattern=maskrcnn_param.validation_file_pattern,
            mode=tf.estimator.ModeKeys.PREDICT,
            num_examples=maskrcnn_param.eval_samples,
            use_fake_data=False,
            use_instance_mask=maskrcnn_param.include_mask,
            seed=maskrcnn_param.seed)
        maskrcnn_param = dict(
            maskrcnn_param.values(),
            mode='eval',
            batch_size=maskrcnn_param.eval_batch_size,
            augment_input_data=False)

        tmp_estimator = tf.estimator.Estimator(
            model_fn=mask_rcnn_model.mask_rcnn_model_fn,
            model_dir=self._temp_dir,
            config=estimator_runconfig,
            params=maskrcnn_param)
        saving_hook = [tf.estimator.CheckpointSaverHook(
            checkpoint_dir=tmp_eval_ckpt_path,
            save_steps=1)]
        tmp_predictor = tmp_estimator.predict(
            input_fn=tmp_input_fn,
            checkpoint_path=decoded_ckpt_path,
            yield_single_examples=False,
            hooks=saving_hook)

        tmp_predictions = six.next(tmp_predictor)

        del tmp_predictor
        del tmp_predictions

        tmp_meta_path = os.path.join(tmp_eval_ckpt_path,
                                     decoded_ckpt_path.split("/")[-1]+".meta")

        return tmp_meta_path, tmp_eval_ckpt_path

    def _ckpt_to_pb(self, meta_path, ckpt_path, tmp_pb_file):
        """Convert ckpt to pb."""

        with tf.Session() as sess:
            # Restore the graph
            saver = tf.compat.v1.train.import_meta_graph(meta_path)
            saver.restore(sess, tf.compat.v1.train.latest_checkpoint(ckpt_path))

            output_node_names = ["gpu_detections/generate_detections/denormalize_box/concat",
                                 "mask_fcn_logits/BiasAdd"]
            constant_graph = graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(),
                output_node_names)

            graph_io.write_graph(constant_graph, os.path.dirname(tmp_pb_file),
                                 os.path.basename(tmp_pb_file),
                                 as_text=False)

    def save_exported_file(self, output_file_name):
        """Save the exported model file.

        This routine converts a keras model to onnx/uff model
        based on the backend the exporter was initialized with.

        Args:
            output_file_name (str): Path to the output file.

        Returns:
            tmp_uff_file (str): Path to the temporary uff file.
        """
        os_handle, tmp_pb_file = tempfile.mkstemp(suffix=".pb")
        os.close(os_handle)

        if self.backend == "uff":
            # 1) extract train checkpoint
            decoded_ckpt_path = self._extract_ckpt(self.model_path, self.key)
            # print("tmp train ckpt path: {}".format(decoded_ckpt_path))
            # 2) load train checkpoint
            # 3) convert train checkpoint to eval checkpoint
            eval_meta_path, eval_ckpt_path = self._train_ckpt_to_eval_ckpt(decoded_ckpt_path)
            # print("tmp meta path: {}\ntmp_eval_ckpt_path: {}".format(eval_meta_path,
            #                                                          eval_ckpt_path))
            # 4) convert eval checkpoint to PB
            self._ckpt_to_pb(eval_meta_path, eval_ckpt_path, tmp_pb_file)

            tf.reset_default_graph()
            cuda.close()
            dynamic_graph = gs.DynamicGraph(tmp_pb_file)
            dynamic_graph = self.node_process(dynamic_graph)

            os.remove(tmp_pb_file)

            uff.from_tensorflow(dynamic_graph.as_graph_def(),
                                self.output_node_names,
                                output_filename=output_file_name,
                                text=False,
                                list_nodes=False,
                                quiet=True)
            logger.info("Converted model was saved into %s", output_file_name)
            return output_file_name
        raise NotImplementedError("Invalid backend provided. {}".format(self.backend))

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["generate_detections", "mask_fcn_logits/BiasAdd"]
        self.input_node_names = ["Input"]

    def set_keras_backend_dtype(self):
        """skip."""
        pass

    def set_session(self):
        """skip."""
        pass

    def node_process(self, maskrcnn_graph):
        """Manipulating the dynamic graph to make it compatible with TRT.

        Args:
            maskrcnn_graph (gs.DynamicGraph): Dynamic graph from the TF Proto file.

        Returns:
            maskrcnn_graph (gs.DymanicGraph): Post processed dynamic graph which is ready to be
                serialized as a uff file.
        """
        spec = self.experiment_spec
        height, width = eval(spec.data_config.image_size)
        height = int(height)
        width = int(width)

        Input = gs.create_node("Input",
                               op="Placeholder",
                               dtype=tf.float32,
                               shape=[-1, 3, height, width])
        box_head_softmax = gs.create_node("box_head_softmax",
                                          op="Softmax")

        # Plugin nodes initialization:
        nearest_upsampling = gs.create_plugin_node(
            "nearest_upsampling", op="ResizeNearest_TRT", dtype=tf.float32, scale=2.0)
        nearest_upsampling_1 = gs.create_plugin_node(
            "nearest_upsampling_1", op="ResizeNearest_TRT", dtype=tf.float32, scale=2.0)
        nearest_upsampling_2 = gs.create_plugin_node(
            "nearest_upsampling_2", op="ResizeNearest_TRT", dtype=tf.float32, scale=2.0)

        test_rpn_nms_thresh = spec.maskrcnn_config.test_rpn_nms_thresh
        test_rpn_post_nms_topn = spec.maskrcnn_config.test_rpn_post_nms_topn
        multilevel_propose_rois = gs.create_plugin_node("multilevel_propose_rois",
                                                        op="MultilevelProposeROI_TRT",
                                                        prenms_topk=4096,
                                                        keep_topk=test_rpn_post_nms_topn,
                                                        fg_threshold=0.0,
                                                        iou_threshold=test_rpn_nms_thresh,
                                                        image_size=[3, height, width])

        box_pool_size = int(spec.maskrcnn_config.mrcnn_resolution // 4)
        pyramid_crop_and_resize_box = gs.create_plugin_node("pyramid_crop_and_resize_box",
                                                            op="MultilevelCropAndResize_TRT",
                                                            pooled_size=box_pool_size,
                                                            image_size=[3, height, width])

        num_classes = spec.data_config.num_classes
        test_detections_per_image = spec.maskrcnn_config.test_detections_per_image
        test_nms = spec.maskrcnn_config.test_nms
        generate_detections = gs.create_plugin_node("generate_detections",
                                                    op="GenerateDetection_TRT",
                                                    num_classes=num_classes,
                                                    keep_topk=test_detections_per_image,
                                                    score_threshold=0.0,
                                                    iou_threshold=test_nms,
                                                    image_size=[3, height, width])

        mask_pool_size = int(spec.maskrcnn_config.mrcnn_resolution // 2)
        pyramid_crop_and_resize_mask = gs.create_plugin_node("pyramid_crop_and_resize_mask",
                                                             op="MultilevelCropAndResize_TRT",
                                                             pooled_size=mask_pool_size,
                                                             image_size=[3, height, width])

        mrcnn_detection_bboxes = gs.create_plugin_node("mrcnn_detection_bboxes",
                                                       op="SpecialSlice_TRT")

        # Create a mapping of namespace names -> plugin nodes.
        namespace_plugin_map = {"input_1": Input,  # for pruned model
                                "input_2": Input,  # for pruned model
                                "IteratorV2": Input,  # for unpruned model
                                "IteratorGetNext": Input,  # for unpruned model
                                "IteratorGetNext:1": Input,  # for unpruned model
                                "input_image_transpose": Input,  # for unpruned model
                                "FPN_up_2": nearest_upsampling,
                                "FPN_up_3": nearest_upsampling_1,
                                "FPN_up_4": nearest_upsampling_2,
                                "anchor_layer": multilevel_propose_rois,
                                "MLP/multilevel_propose_rois": multilevel_propose_rois,
                                "MLP/concat_scores": multilevel_propose_rois,
                                "MLP/concat_rois": multilevel_propose_rois,
                                "MLP/roi_post_nms_topk": multilevel_propose_rois,
                                "multilevel_crop_resize/multilevel_crop_and_resize":
                                    pyramid_crop_and_resize_box,
                                "multilevel_crop_resize/selective_crop_and_resize":
                                    pyramid_crop_and_resize_box,
                                "gpu_detections/generate_detections":
                                    generate_detections,
                                "multilevel_crop_resize_1/multilevel_crop_and_resize":
                                    pyramid_crop_and_resize_mask,
                                "multilevel_crop_resize_1/selective_crop_and_resize":
                                    pyramid_crop_and_resize_mask,
                                }

        # Keep sigmoid and reshape for rpn output
        excluded_nodes_names = \
            ["MLP/multilevel_propose_rois/level_{}/Reshape".format(i) for i in range(2, 7)]
        excluded_nodes_names.extend(
            ["MLP/multilevel_propose_rois/level_{}/Reshape/shape".format(i) for i in range(2, 7)])
        excluded_nodes_names.extend(
            ["MLP/multilevel_propose_rois/level_{}/Sigmoid".format(i) for i in range(2, 7)])
        excluded_nodes_names.extend(
            ["MLP/multilevel_propose_rois/level_{}/Reshape_1".format(i) for i in range(2, 7)])
        excluded_nodes_names.extend(
            ["MLP/multilevel_propose_rois/level_{}/Reshape_1/shape".format(i) for i in range(2, 7)])

        excluded_nodes = []
        for node_name in excluded_nodes_names:
            excluded_nodes.append(maskrcnn_graph.node_map[node_name])

        # Plugin can get input image info through header file
        input_disconnect_pairs = [("Input", "multilevel_propose_rois"),
                                  ("Input", "generate_detections")]

        # Insert slice node
        detection_disconnect_pairs = [("generate_detections", "pyramid_crop_and_resize_mask")]
        detection_connect_pairs = [("generate_detections", "mrcnn_detection_bboxes"),
                                   ("mrcnn_detection_bboxes", "pyramid_crop_and_resize_mask")]

        # to change the input data's order in graph
        box_head_disconnect_pairs = [("multilevel_propose_rois", "pyramid_crop_and_resize_box")]
        box_head_connect_pairs = [("multilevel_propose_rois", "pyramid_crop_and_resize_box")]

        # remove the reshape in box head since TRT FC layer can handle 5-D input
        box_head_remove_list = ["box_head_reshape1/box_head_reshape1",
                                "box_head_reshape2/box_head_reshape2",
                                "box_head_reshape3/box_head_reshape3"]
        box_head_connect_pairs.extend([("pyramid_crop_and_resize_box", "fc6/MatMul"),
                                       ("class-predict/BiasAdd", "box_head_softmax"),
                                       ("box_head_softmax", "generate_detections"),
                                       ("box-predict/BiasAdd", "generate_detections")])

        # remove the reshape before mask_head since TRT conv can handle 5-D input
        mask_head_remove_list = ["mask_head_reshape_1/mask_head_reshape_1"]
        mask_head_connect_pairs = [("pyramid_crop_and_resize_mask",
                                    "mask-conv-l0/Conv2D")]

        def connect(dynamic_graph, connections_list):

            for node_a_name, node_b_name in connections_list:
                if node_a_name not in dynamic_graph.node_map[node_b_name].input:
                    dynamic_graph.node_map[node_b_name].input.insert(0, node_a_name)

        def add(dynamic_graph, node):

            dynamic_graph._internal_graphdef.node.extend([node])
            # remap node:
            dynamic_graph.node_map = \
                {node.name: node for node in dynamic_graph._internal_graphdef.node}

        maskrcnn_graph.collapse_namespaces(
            namespace_plugin_map, exclude_nodes=excluded_nodes, unique_inputs=True)
        maskrcnn_graph.remove(mask_head_remove_list)
        maskrcnn_graph.remove(box_head_remove_list)
        add(maskrcnn_graph, mrcnn_detection_bboxes)
        add(maskrcnn_graph, box_head_softmax)

        for who, of_whom in input_disconnect_pairs:
            maskrcnn_graph.disconnect(who, of_whom)

        for who, of_whom in detection_disconnect_pairs:
            maskrcnn_graph.disconnect(who, of_whom)

        for who, of_whom in box_head_disconnect_pairs:
            maskrcnn_graph.disconnect(who, of_whom)

        connect(maskrcnn_graph, detection_connect_pairs)
        connect(maskrcnn_graph, mask_head_connect_pairs)
        connect(maskrcnn_graph, box_head_connect_pairs)

        return maskrcnn_graph

    def set_data_preprocessing_parameters(self, input_dims, image_mean=None):
        """Set data pre-processing parameters for the int8 calibration."""
        logger.debug("Input dimensions: {}".format(input_dims))
        num_channels = input_dims[0]
        if num_channels == 3:
            means = [123.675, 116.280, 103.53]
        else:
            raise NotImplementedError("Invalid number of dimensions {}.".format(num_channels))
        # ([R, G, B]/ 255 - [0.485, 0.456, 0.406]) / 0.224 -->
        # (R/G/B - mean) * ratio
        self.preprocessing_arguments = {"scale": 0.017507,
                                        "means": means,
                                        "flip_channel": False}

    def generate_ds_config(self, input_dims, num_classes=None):
        """Generate Deepstream config element for the exported model."""
        if num_classes:
            assert num_classes > 1, "Please verify num_classes in the spec file. \
                num_classes should be number of categories in json + 1."
        if input_dims[0] == 1:
            color_format = "l"
        else:
            color_format = "bgr" if self.preprocessing_arguments["flip_channel"] else "rgb"
        kwargs = {
            "data_format": self.data_format,
            "backend": self.backend,
            # Setting this to 3 by default since MaskRCNN
            # is an instance segmentation network.
            "network_type": 3
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

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        ann_file = self.experiment_spec.data_config.val_json_file
        if not os.path.isfile(ann_file):
            logger.warning("validation json file ({}) doesn't exist.\
                DS label file won't be generated.".format(ann_file))
            return None
        coco = COCO(ann_file)
        # Create an index for the category names
        cats = coco.loadCats(coco.getCatIds())
        assert cats, "Please verify the categories in the json annotation."
        cats_sorted = sorted(cats, key=lambda k: k['id'])
        cats_list = [c['name'] for c in cats_sorted]
        return cats_list

    def export(self, output_file_name, backend,
               calibration_cache="", data_file_name="",
               n_batches=1, batch_size=1, verbose=True,
               calibration_images_dir="", save_engine=False,
               engine_file_name="", max_workspace_size=1 << 30,
               max_batch_size=1, force_ptq=False, static_batch_size=None,
               gen_ds_config=False, min_batch_size=None,
               opt_batch_size=None, calib_json_file=None):
        """Export."""
        if force_ptq:
            print("MaskRCNN doesn't support QAT. Post training quantization is used by default.")
        # Set keras session.
        self.set_backend(backend)
        self.set_input_output_node_names()
        _ = self.save_exported_file(output_file_name)

        # Get int8 calibrator
        calibrator = None
        max_batch_size = max(batch_size, max_batch_size)
        spec = self.experiment_spec
        hh, ww = eval(spec.data_config.image_size)
        input_dims = (3, hh, ww)
        logger.debug("Input dims: {}".format(input_dims))

        if self.data_type == "int8":
            # no tensor scale, take traditional INT8 calibration approach
            # use calibrator to generate calibration cache
            calibrator = self.get_calibrator(calibration_cache=calibration_cache,
                                             data_file_name=data_file_name,
                                             n_batches=n_batches,
                                             batch_size=batch_size,
                                             input_dims=input_dims,
                                             calibration_images_dir=calibration_images_dir)
            logger.info("Calibration takes time especially if number of batches is large.")

        if gen_ds_config:
            self.set_data_preprocessing_parameters(input_dims)
            ds_config = self.generate_ds_config(input_dims, spec.data_config.num_classes)
            ds_labels = self.get_class_labels()
            output_root = os.path.dirname(output_file_name)
            if ds_labels:
                ds_labels = ['BG'] + ds_labels
                with open(os.path.join(output_root, "labels.txt"), "w") as lfile:
                    for label in ds_labels:
                        lfile.write("{}\n".format(label))
                assert lfile.closed, (
                    "Label file wasn't closed after saving."
                )
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            ds_file = os.path.join(output_root, "nvinfer_config.txt")
            with open(ds_file, "w") as dsf:
                dsf.write(str(ds_config))
            assert dsf.closed, (
                "Deepstream config file wasn't closed."
            )

        # Verify with engine generation / run calibration.
        if self.backend == "uff":
            # Assuming single input node graph for uff engine creation.
            in_tensor_name = self.input_node_names[0]
            if not isinstance(input_dims, dict):
                input_dims = {in_tensor_name: input_dims}
            engine_builder = UFFEngineBuilder(output_file_name,
                                              in_tensor_name,
                                              input_dims,
                                              self.output_node_names,
                                              max_batch_size=max_batch_size,
                                              max_workspace_size=max_workspace_size,
                                              dtype=self.data_type,
                                              strict_type=self.strict_type,
                                              verbose=verbose,
                                              calibrator=calibrator,
                                              tensor_scale_dict=self.tensor_scale_dict)
            trt_engine = engine_builder.get_engine()
            if save_engine:
                with open(engine_file_name, "wb") as outf:
                    outf.write(trt_engine.serialize())
            if trt_engine:
                del trt_engine
        else:
            raise NotImplementedError("Invalid backend.")
