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

import copy
import logging
import os
import random
import tempfile
import onnx
from six.moves import xrange
import tensorflow as tf
from tensorflow.compat.v1 import GraphDef
import tf2onnx
from tqdm import tqdm

from nvidia_tao_tf1.core.export._onnx import keras_to_onnx
from nvidia_tao_tf1.core.export._uff import keras_to_pb
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.common.export.tensorfile import TensorFile
from nvidia_tao_tf1.cv.unet.export.unet_ds_config import UnetDSConfig
from nvidia_tao_tf1.cv.unet.model.model_io import check_for_quantized_layers, load_keras_model
from nvidia_tao_tf1.cv.unet.model.utilities import build_target_class_list, get_train_class_mapping
from nvidia_tao_tf1.cv.unet.model.utilities import initialize, initialize_params
from nvidia_tao_tf1.cv.unet.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.unet.utils.data_loader import Dataset


logger = logging.getLogger(__name__)


class UNetExporter(Exporter):
    """Exporter class to export a trained UNet model."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 backend="onnx",
                 **kwargs):
        """Instantiate the UNet exporter to export a trained UNet .tlt model.

        Args:
            model_path(str): Path to the UNet model file.
            key (str): Key to decode the model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            experiment_spec_path (str): Path to UNet experiment spec file.
            backend (str): Type of intermediate backend parser to be instantiated.
        """
        super(UNetExporter, self).__init__(model_path=model_path,
                                           key=key,
                                           data_type=data_type,
                                           strict_type=strict_type,
                                           backend=backend,
                                           **kwargs)
        self.experiment_spec_path = experiment_spec_path
        assert os.path.isfile(self.experiment_spec_path), \
            "Experiment spec file not found at {}.".format(self.experiment_spec_path)
        self.experiment_spec = load_experiment_spec(self.experiment_spec_path,
                                                    merge_from_default=False)
        self.keras_model, self.custom_objs = load_keras_model(self.experiment_spec,
                                                              self.model_path,
                                                              export=True, key=self.key)

        self.enable_qat = self.experiment_spec.model_config.enable_qat
        self.model_arch = self.experiment_spec.model_config.arch

        self.enable_qat = self.experiment_spec.model_config.enable_qat
        self.model_arch = self.experiment_spec.model_config.arch
        self.export_route = "pb2onnx"
        initialize(self.experiment_spec)
        # Initialize Params
        self.params = initialize_params(self.experiment_spec)

    def load_model(self, backend="onnx"):
        """Simple function to load the UNet Keras model."""

        model = self.keras_model
        if check_for_quantized_layers(model):
            logger.info("INT8 quantization using QAT")
            model, self.tensor_scale_dict = self.extract_tensor_scale(model, backend)

        return model

    def set_keras_backend_dtype(self):
        """skip."""
        pass

    def pb_to_onnx(self, input_filename, output_filename, input_node_names,
                   output_node_names, target_opset=None):
        """Convert a TensorFlow model to ONNX.

        The input model needs to be passed as a frozen Protobuf file.
        The exported ONNX model may be parsed and optimized by TensorRT.

        Args:
            input_filename (str): path to protobuf file.
            output_filename (str): file to write exported model to.
            input_node_names (list of str): list of model input node names as
                returned by model.layers[some_idx].get_output_at(0).name.split(':')[0].
            output_node_names (list of str): list of model output node names as
                returned by model.layers[some_idx].get_output_at(0).name.split(':')[0].
            target_opset (int): Target opset version to use, default=<default opset for
                the current keras2onnx installation>
        Returns:
            tuple<in_tensor_name(s), out_tensor_name(s):
            in_tensor_name(s): The name(s) of the input nodes. If there is
            only one name, it will be returned as a single string, otherwise
            a list of strings.
            out_tensor_name(s): The name(s) of the output nodes. If there is only one name,
            it will be returned as a single string, otherwise a list of strings.
        """
        graphdef = GraphDef()
        with tf.gfile.GFile(input_filename, "rb") as frozen_pb:
            graphdef.ParseFromString(frozen_pb.read())

        if not isinstance(input_node_names, list):
            input_node_names = [input_node_names]
        if not isinstance(output_node_names, list):
            output_node_names = [output_node_names]

        # The ONNX parser requires tensors to be passed in the node_name:port_id format.
        # Since we reset the graph below, we assume input and output nodes have a single port.
        input_node_names = ["{}:0".format(node_name) for node_name in input_node_names]
        output_node_names = ["{}:0".format(node_name) for node_name in output_node_names]

        tf.reset_default_graph()
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graphdef, name="")

            onnx_graph = tf2onnx.tfonnx.process_tf_graph(
                tf_graph,
                input_names=input_node_names,
                output_names=output_node_names,
                continue_on_error=True,
                verbose=True,
                opset=target_opset,
            )
            onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)
            model_proto = onnx_graph.make_model("test")
            with open(output_filename, "wb") as f:
                f.write(model_proto.SerializeToString())

        # Reload and check ONNX model.
        onnx_model = onnx.load(output_filename)
        onnx.checker.check_model(onnx_model)

        # Return a string instead of a list if there is only one input or output.
        if len(input_node_names) == 1:
            input_node_names = input_node_names[0]

        if len(output_node_names) == 1:
            output_node_names = output_node_names[0]

        return input_node_names, output_node_names

    def optimize_onnx(self, onnx_path):
        """"Function to optimize the ONNX by removing softmax."""

        model = onnx.load(onnx_path)
        copied_model = copy.deepcopy(model)
        graph = copied_model.graph

        if self.params.activation == "sigmoid":
            pass
        else:
            softmax_1 = [node for node in graph.node if 'softmax' in node.name]
            # Record the input node names before removing the node
            softmax_1_inp_node = softmax_1[0].input
            graph.node.remove(softmax_1[0])
            graph.output.remove(graph.output[0])
            # Create ArgMax node
            # Input shape is Nxhxwx2. Output shape is Nxhxwx1.
            output_tensor = onnx.helper.make_tensor_value_info(
                'argmax_1', onnx.TensorProto.INT64,
                ('N', self.experiment_spec.model_config.model_input_height,
                 self.experiment_spec.model_config.model_input_width, 1))
            graph.output.append(output_tensor)
            # Last axis - input tensor shape is Nx544x960x2.
            argmax_node = onnx.helper.make_node(
                'ArgMax', softmax_1_inp_node, ['argmax_1'], axis=-1, keepdims=1)
            graph.node.append(argmax_node)
            onnx.checker.check_model(copied_model)
            logger.info('New input/output')
            logger.info(graph.input)
            logger.info(graph.output)
            onnx.save(copied_model, onnx_path)
        output_names = [node.name for node in copied_model.graph.output]

        return output_names

    def convert_to_onnx(self, model, export_route, tmp_onnx_file):
        """Function to model to ONNX based on the export_route."""

        if export_route == "keras2onnx":
            keras_to_onnx(model,
                          tmp_onnx_file,
                          custom_objects=self.custom_objs,
                          target_opset=self.target_opset)
        elif export_route == "pb2onnx":
            if self.target_opset != 11:
                logger.warning("UNet uses target opset of 11 by default."
                               "Overriding the provided opset {} to 11.".format(self.target_opset))
            target_opset = 11
            output_pb_filename = tmp_onnx_file.replace(".onnx", ".pb")
            in_tensor_names, out_tensor_names, __ = keras_to_pb(model,
                                                                output_pb_filename,
                                                                output_node_names=None,
                                                                custom_objects=self.custom_objs)
            (_, _) = self.pb_to_onnx(output_pb_filename,
                                     tmp_onnx_file,
                                     in_tensor_names,
                                     out_tensor_names,
                                     target_opset)
            os.remove(output_pb_filename)
            
    def save_exported_file(self, model, output_file_name):
        """save an ONNX file.

        Args:
            model (keras.model.Model): Decoded keras model to be exported.
            output_file_name (str): Path to the output file.

        Returns:
            tmp_uff_file (str): Path to the temporary uff file.
        """

        if self.backend == "onnx":
            # TO DO: Remove the below line. It is to experiment the trtexec.
            # generate encoded onnx model with empty string as input node name
            self.convert_to_onnx(model, self.export_route, output_file_name)
            output_names = self.optimize_onnx(output_file_name)
            # Update output names here for the modified graph
            self.output_node_names = output_names
            tf.reset_default_graph()
            logger.info("Converted model was saved into %s", output_file_name)
            return output_file_name
        raise NotImplementedError("Invalid backend provided. {}".format(self.backend))

    def set_input_output_node_names(self):
        """Set input output node names."""

        model = self.load_model()
        output_name = [node.op.name for node in model.outputs]
        input_name = [node.op.name for node in model.inputs]
        output_node_name = output_name[0].split("/")[0]
        input_node_name = input_name[0].split("/")[0]
        self.output_node_names = [output_node_name]
        self.input_node_names = [input_node_name]

    def set_data_preprocessing_parameters(self, input_dims, image_mean):
        """Set data pre-processing parameters for the int8 calibration."""
        num_channels = input_dims[0]
        if num_channels == 3:
            means = [127.5, 127.5, 127.5]
        else:
            means = [127.5]
        self.preprocessing_arguments = {"scale": 1.0 / 127.5,
                                        "means": means,
                                        "flip_channel": True}

    def generate_ds_config(self, input_dims, num_classes=None):
        """Generate Deepstream config element for the exported model."""
        if input_dims[0] == 1:
            color_format = "l"
        else:
            color_format = "bgr" if self.preprocessing_arguments["flip_channel"] else "rgb"

        kwargs = {
            "data_format": self.data_format,
            "backend": self.backend,
            # Setting this to 2 by default since it is semantic segmentation
        }
        if self.params.activation == "sigmoid":
            kwargs["network_type"] = 2
            kwargs["output_tensor_meta"] = 0
        else:
            kwargs["network_type"] = 100
            # This is set to output the fibal inference image
            kwargs["output_tensor_meta"] = 1

        if num_classes:
            kwargs["num_classes"] = num_classes
        ds_config = UnetDSConfig(
            self.preprocessing_arguments["scale"],
            self.preprocessing_arguments["means"],
            input_dims,
            color_format,
            self.key,
            segmentation_threshold=0.0,
            output_blob_names=self.output_node_names[0],
            segmentation_output_order=1,
            **kwargs
        )
        return ds_config

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        if self.experiment_spec is None:
            raise AttributeError(
                "Experiment spec wasn't loaded. To get class labels "
                "please provide the experiment spec file using the -e "
                "option.")
        target_labels = []
        target_classes = build_target_class_list(
            self.experiment_spec.dataset_config.data_class_config)

        train_id_name_mapping = get_train_class_mapping(target_classes)
        num_classes = len(train_id_name_mapping)
        for class_id in range(num_classes):
            target_labels.append(train_id_name_mapping[class_id][0])

        return target_labels

    def generate_tensor_file(self, data_file_name,
                             calibration_images_dir,
                             input_dims, n_batches=10,
                             batch_size=1, image_mean=None):
        """Generate calibration Tensorfile for int8 calibrator.

        This function generates a calibration tensorfile from a directory of images, or dumps
        n_batches of random numpy arrays of shape (batch_size,) + (input_dims).

        Args:
            data_file_name (str): Path to the output tensorfile to be saved.
            calibration_images_dir (str): Path to the images to generate a tensorfile from.
            input_dims (list): Input shape in CHW order.
            n_batches (int): Number of batches to be saved.
            batch_size (int): Number of images per batch.
            image_mean (list): Image mean per channel.

        Returns:
            No explicit returns.
        """

        # Initialize the environment
        initialize(self.experiment_spec)
        # Initialize Params
        params = initialize_params(self.experiment_spec)
        params["experiment_spec"] = self.experiment_spec
        target_classes = build_target_class_list(
            self.experiment_spec.dataset_config.data_class_config)
        dataset = Dataset(
                          batch_size=batch_size,
                          params=params,
                          target_classes=target_classes)

        # Preparing the list of images to be saved.
        num_images = n_batches * batch_size
        valid_image_ext = ['jpg', 'jpeg', 'png']
        image_list = dataset.image_names_list
        if not len(image_list) > 0:
            if os.path.exists(calibration_images_dir):
                image_list = [os.path.join(calibration_images_dir, image)
                              for image in os.listdir(calibration_images_dir)
                              if image.split('.')[-1] in valid_image_ext]
        if len(image_list) > 0:
            if len(image_list) < num_images:
                raise ValueError('Not enough number of images provided:'
                                 ' {} < {}'.format(len(image_list), num_images))
            image_idx = random.sample(xrange(len(image_list)), num_images)
            self.set_data_preprocessing_parameters(input_dims, image_mean)
            # Writing out processed dump.
            with TensorFile(data_file_name, 'w') as f:
                for chunk in tqdm(image_idx[x:x+batch_size] for x in xrange(0, len(image_idx),
                                                                            batch_size)):
                    dump_data = self.prepare_chunk(chunk, image_list,
                                                   image_width=input_dims[2],
                                                   image_height=input_dims[1],
                                                   channels=input_dims[0],
                                                   batch_size=batch_size,
                                                   **self.preprocessing_arguments)
                    f.write(dump_data)
            f.closed
        else:
            # Calibration images are not present in cal image dir or experiment spec
            logger.info("Generating a tensorfile with random tensor images. This may work well as "
                        "a profiling tool, however, it may result in inaccurate results at "
                        "inference. Please generate a tensorfile using the tlt-int8-tensorfile, "
                        "or provide a custom directory of images for best performance.")
            self.generate_random_tensorfile(data_file_name,
                                            input_dims,
                                            n_batches=n_batches,
                                            batch_size=batch_size)

    def get_input_dims_from_model(self, model=None):
        """Read input dimensions from the model.

        Args:
            model (keras.models.Model): Model to get input dimensions from.

        Returns:
            input_dims (tuple): Input dimensions.
        """
        if model is None:
            raise IOError("Invalid model object.")
        input_dims = model.layers[1].input_shape[1:]
        return input_dims
