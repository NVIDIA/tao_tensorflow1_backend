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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import sys
import tempfile

"""Root logger for export app."""
logger = logging.getLogger(__name__)  # noqa

import keras
import mock
import numpy as np

try:
    import pycuda.driver as cuda
except ImportError:
    logger.warning(
        "Failed to import CUDA package. TRT inference testing will not be available."
    )
    cuda = None

from nvidia_tao_tf1.core.export import (
    keras_to_caffe,
    keras_to_onnx,
    keras_to_tensorrt,
    keras_to_uff,
    TensorFile
)
from nvidia_tao_tf1.core.export._tensorrt import _set_excluded_layer_precision
from nvidia_tao_tf1.core.export.app import get_model_input_dtype
from nvidia_tao_tf1.core.models.templates.conv_gru_2d_export import ConvGRU2DExport
from nvidia_tao_tf1.core.templates.helnet import HelNet
import nvidia_tao_tf1.core.utils

import pytest
import tensorflow as tf

try:
    import tensorrt as trt
except ImportError:
    logger.warning(
        "Failed to import TRT package. TRT inference testing will not be available."
    )
    trt = None

import third_party.keras.tensorflow_backend

_onnx_supported = False
if sys.version_info >= (3, 0):
    _onnx_supported = True


class TestModelExport(object):
    """Main class for model export tests."""

    def common(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
        add_transpose_conv=False,
        add_reshape=False,
        add_dropout=False,
        add_dense=False,
        add_maxpool2D=False,
        add_concat_layer=False,
        model_in_model=False,
        multiple_outputs=False,
        add_unnecessary_outputs=False,
        intermediate_output=False,
        dilation_rate=None,
        add_conv_gru=False,
    ):
        inputs = keras.layers.Input(shape=input_shape)

        model = model(
            nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
        )

        # Custom keras layers to be exported.
        custom_keras_layers = dict()
        # Layers that have additional state input/output.
        layers_with_state_io = []

        if add_dropout:
            x = model.outputs[0]
            x = keras.layers.Dropout(rate=0.5)(x)
            x = keras.layers.Conv2D(32, (3, 3))(x)
            model = keras.models.Model(inputs=inputs, outputs=x)

        if add_transpose_conv:
            model = nvidia_tao_tf1.core.models.templates.utils.add_deconv_head(
                model, inputs, nmaps=3, upsampling=4, data_format=data_format
            )

        if add_maxpool2D:
            x = model.outputs[0]
            x = keras.layers.MaxPooling2D()(x)
            model = keras.models.Model(inputs=inputs, outputs=x)

        if add_reshape:
            x = model.outputs[0]
            x = keras.layers.Reshape((-1, 16))(x)
            model = keras.models.Model(inputs=inputs, outputs=x)

        if add_dense:
            x = model.outputs[0]
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(10, activation="tanh")(x)
            model = keras.models.Model(inputs=inputs, outputs=x)

        if add_concat_layer:
            x = model.outputs[0]
            # First branch.
            num_filters_x1 = 4
            x1 = keras.layers.Conv2D(
                filters=num_filters_x1,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                data_format=data_format,
                dilation_rate=(1, 1),
                activation="sigmoid",
                name="conv2d_x1",
            )(x)
            # Second branch.
            num_filters_x2 = 2
            x2 = keras.layers.Conv2D(
                filters=num_filters_x2,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                dilation_rate=(1, 1),
                activation="relu",
                name="conv2d_x2",
            )(x)
            # Merge branches.
            concat_axis = 1 if data_format == "channels_first" else -1
            x = keras.layers.Concatenate(axis=concat_axis, name="concat")([x1, x2])
            # Add extra layer on top.
            x = keras.layers.Conv2D(
                filters=num_filters_x1 + num_filters_x2,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                data_format=data_format,
                activation="sigmoid",
                name="conv2d_output",
            )(x)
            # One final layer.
            x = keras.layers.Conv2D(
                filters=1,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                data_format=data_format,
                name="net_output",
            )(x)
            model = keras.models.Model(inputs=inputs, outputs=x)

        if dilation_rate is not None:
            x = model.outputs[0]
            # Add a Conv2D layer with dilation.
            y = keras.layers.Conv2D(
                filters=1,
                kernel_size=[1, 1],
                strides=(1, 1),
                dilation_rate=dilation_rate,
                data_format=data_format,
            )(x)
            model = keras.models.Model(inputs=inputs, outputs=y)

        if model_in_model:
            outer_inputs = keras.layers.Input(shape=input_shape)
            inner_model = model
            model = keras.models.Model(
                inputs=outer_inputs, outputs=inner_model(outer_inputs)
            )

        if add_conv_gru:
            x = model.outputs[0]
            # Add the conv GRU layer.
            y = ConvGRU2DExport(
                model_sequence_length_in_frames=1,
                input_sequence_length_in_frames=1,
                state_scaling=0.9,
                input_shape=x.shape.as_list(),
                initial_state_shape=x.shape.as_list(),
                spatial_kernel_height=1,
                spatial_kernel_width=1,
                kernel_regularizer=None,
                bias_regularizer=None,
                is_stateful=True,
            )(x)
            model = keras.models.Model(inputs=inputs, outputs=y)
            # Update custom layers dictionary for passing to the export code.
            custom_keras_layers.update({"ConvGRU2DExport": ConvGRU2DExport})
            # Add this layer as a state io layer.
            layers_with_state_io.append(model.layers[-1])

        if multiple_outputs:
            y1 = model.outputs[0]
            y2 = keras.layers.Conv2D(32, (3, 3))(y1)

            outputs = [y1, y2]

            if add_unnecessary_outputs:
                # Add y3-y5 to the Keras model outputs. These should be ignored by the
                # exporters as only y1 and y2 are added to output_node_names.
                y3 = keras.layers.Conv2D(16, (3, 3))(y1)
                y4 = keras.layers.Conv2D(16, (3, 3))(y2)
                y5 = keras.layers.Conv2D(16, (3, 3))(y4)
                outputs += [y3, y4, y5]

            model = keras.models.Model(inputs=inputs, outputs=outputs)

        keras_model_file = os.path.join(str(tmpdir), "model.hdf5")
        with keras.utils.CustomObjectScope(custom_keras_layers):
            model.save(keras_model_file)
        keras.backend.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=config))
        model = keras.models.load_model(keras_model_file, custom_keras_layers)

        if intermediate_output:
            # Get a preceding layer's activation as the output.
            output_layers = [model.layers[-4]]
        else:
            # Use keras model outputs, but only get up to two outputs to test exporting a
            # model that has unnecessary extra outputs as well.
            output_layers = model._output_layers[:2]

        # Get output node names to be exported.
        output_node_names = []
        for layer in output_layers:
            if export_format == "uff":
                output_node_names.append(layer.get_output_at(0).name.split(":")[0])
            elif export_format == "onnx":
                # keras2onnx will drop the part after slash for output tensor names
                output_node_names.append(layer.get_output_at(0).name.split("/")[0])
            elif export_format == "caffe":
                # Convert Tensor names to Caffe layer names.
                if (
                    hasattr(layer, "activation")
                    and layer.activation.__name__ != "linear"
                ):
                    output_node_name = "%s/%s" % (
                        layer.name,
                        layer.activation.__name__.capitalize(),
                    )
                else:
                    output_node_name = layer.name
                output_node_names.append(output_node_name)

        test_input_shape = (None,) + input_shape

        # For the stateful layers only uff export is supported.
        if export_format in ["uff", "onnx"]:
            if layers_with_state_io:
                test_input_shape = [test_input_shape]
            for layer in layers_with_state_io:
                if export_format == "uff":
                    stateful_node_name = layer.get_output_at(0).name.split(":")[0]
                else:
                    stateful_node_name = layer.get_output_at(0).name

                if stateful_node_name not in output_node_names:
                    output_node_names.append(stateful_node_name)
                test_input_shape.append(layer.input_shape)

        # Only specify output node names to exporter if we need to deviate from keras model outputs.
        output_node_names_to_export = (
            output_node_names
            if add_unnecessary_outputs or intermediate_output
            else None
        )

        if export_format == "uff":
            uff_filename = os.path.join(str(tmpdir), "model.uff")
            in_tensor_name, out_tensor_name, export_input_dims = keras_to_uff(
                model,
                uff_filename,
                output_node_names=output_node_names_to_export,
                custom_objects=custom_keras_layers,
            )
            assert os.path.isfile(uff_filename)
            assert test_input_shape == export_input_dims

        elif export_format == "onnx":
            onnx_filename = os.path.join(str(tmpdir), "model.onnx")
            in_tensor_name, out_tensor_name, export_input_dims = keras_to_onnx(
                model, onnx_filename, custom_objects=custom_keras_layers
            )
            assert os.path.isfile(onnx_filename)
            # onnx model has explicit batch size, hence dim[0] cannot match that in Keras
            assert list(test_input_shape)[1:] == list(export_input_dims)[1:]

        elif export_format == "caffe":
            proto_filename = os.path.join(str(tmpdir), "model.proto")
            snapshot_filename = os.path.join(str(tmpdir), "model.caffemodel")
            in_tensor_name, out_tensor_name = keras_to_caffe(
                model,
                proto_filename,
                snapshot_filename,
                output_node_names=output_node_names_to_export,
            )
            assert os.path.isfile(proto_filename)
            assert os.path.isfile(snapshot_filename)

            # TensorRT requires all input_layers to be 4-dimensional.
            # This check ensures that the caffe model can be converted to TensorRT.
            assert all(
                [
                    len(input_layer.input_shape) == 4
                    for input_layer in model._input_layers
                ]
            )
        else:
            raise ValueError("Unknown format: %s" % export_format)

        # For the stateful layers only uff export is supported.
        if layers_with_state_io and export_format == "uff":
            assert in_tensor_name == [
                model.layers[0].get_output_at(0).name.split(":")[0]
            ] + [layer.state_input_name for layer in layers_with_state_io]
        elif export_format == "onnx":
            assert in_tensor_name == model.layers[0].get_output_at(0).name.split(":")[0]
        else:
            # Make sure input/output tensor names were returned correctly.
            assert in_tensor_name == model.layers[0].get_output_at(0).name.split(":")[0]

        # Exporter gives a list of output names only if there are multiple outputs.
        if len(output_node_names) == 1:
            output_node_names = output_node_names[0]

        if model_in_model and export_format in ["uff"]:
            # In the case of a model-in-model architecture, output tensor names
            # are registered in the namespace of the inner model.
            output_node_names = "{}/{}".format(inner_model.name, output_node_names)
        elif model_in_model and export_format in ["onnx"]:
            output_node_names = inner_model.name

        if isinstance(out_tensor_name, list):
            for idx in range(len(out_tensor_name)):
                assert out_tensor_name[idx] in output_node_names[idx]
        else:
            assert out_tensor_name in output_node_names

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 256, 256), "uff"),
            (HelNet, 6, "channels_first", False, (3, 256, 256), "uff"),
            (HelNet, 10, "channels_last", True, (64, 64, 3), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 128), "caffe"),
            (HelNet, 10, "channels_first", False, (3, 256, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 128), "onnx"),
            (HelNet, 10, "channels_first", False, (3, 256, 256), "onnx"),
        ],
    )
    def test_export(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that our model exports to the destination format."""
        if export_format == "onnx" and not _onnx_supported:
            return

        keras.layers.Input(shape=input_shape)

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_with_conv_transpose_head(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that our model exports to PB and UFF."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            add_transpose_conv=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_with_reshape(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that our model exports to PB and UFF."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            add_reshape=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_with_dropout(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that our model with dropout exports to PB and UFF."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            add_dropout=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_with_dense(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that our model with dense layer exports to PB and UFF."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            add_dense=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_with_branches(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that our model with branches and a concatenation layer exports to Caffe and UFF."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            add_concat_layer=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_with_maxpool2D(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that our model with 2D max pooling exports to PB and UFF."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            add_maxpool2D=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape, "
        "dilation_rate, export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), (2, 2), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), (2, 2), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), (2, 2), "onnx"),
        ],
    )
    def test_with_dilation(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        dilation_rate,
        export_format,
    ):
        """Test that our model with dilation exports to PB and UFF."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            dilation_rate=dilation_rate,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_model_in_model(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that we can export a model-in-model type of architecture."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            model_in_model=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_multiple_outputs(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that we can export a model with multiple outputs."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            multiple_outputs=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            # TODO(bhupinders): Right now ONNX cannot pick and chose output nodes to
            # export, it has to export the entire keras model.
            # (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_unnecessary_outputs(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that we can export only given output nodes, ignoring other outputs."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            multiple_outputs=True,
            add_unnecessary_outputs=True,
        )

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), "caffe"),
            # TODO(bhupinders): ONNX converter can't manipulate/extract intermediate
            # outputs.
            # (HelNet, 6, "channels_first", True, (3, 128, 256), "onnx"),
        ],
    )
    def test_intermediate_output(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that we can export subgraphs of models."""
        if export_format == "onnx" and not _onnx_supported:
            return

        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            intermediate_output=True,
        )

    # @pytest.mark.parametrize(
    #     "output_fn, model, nlayers, data_format, use_batch_norm, input_shape,"
    #     "output_format",
    #     [
    #         ("default", HelNet, 6, "channels_first", False, (3, 256, 256), "uff"),
    #         ("default", HelNet, 6, "channels_first", False, (3, 256, 256), "onnx"),
    #     ],
    # )
    # @pytest.mark.script_launch_mode("subprocess")
    # def test_export_app(
    #     self,
    #     script_runner,
    #     tmpdir,
    #     output_fn,
    #     model,
    #     nlayers,
    #     data_format,
    #     use_batch_norm,
    #     input_shape,
    #     output_format,
    # ):
    #     """Test the export application.

    #     Just make sure a model file is generated.
    #     """
    #     if output_format == "onnx" and not _onnx_supported:
    #         return

    #     model_filename = os.path.join(str(tmpdir), "model.h5")

    #     if output_fn == "default":
    #         extra_args = []
    #         suffix = ".%s" % output_format
    #         output_filename = os.path.join(str(tmpdir), "model.h5" + suffix)
    #     else:
    #         output_filename = os.path.join(str(tmpdir), output_fn)
    #         extra_args = ["--output", output_filename]

    #     extra_args.extend(["--format", output_format])

    #     inputs = keras.layers.Input(shape=input_shape)
    #     model = model(
    #         nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
    #     )
    #     model.save(model_filename)

    #     env = os.environ.copy()
    #     # If empty, then "import pycuda.autoinit" will not work saying
    #     # "pycuda._driver.RuntimeError: cuInit failed: no CUDA-capable device is detected"
    #     # This is inside a protected "try" in tensorrt/lite/engine.py, so you'll only see an error
    #     # saying to make sure pycuda is installed, which is wrong.
    #     # TODO(xiangbok): restore to no devices
    #     env["CUDA_VISIBLE_DEVICES"] = "0"
    #     script = "app.py"
    #     # Path adjustment for bazel tests
    #     if os.path.exists(os.path.join("nvidia_tao_tf1/core/export", script)):
    #         script = os.path.join("nvidia_tao_tf1/core/export", script)
    #     ret = script_runner.run(script, model_filename, env=env, *extra_args)

    #     assert ret.success, "Process returned error: %s error trace: %s" % (
    #         ret.success,
    #         ret.stderr,
    #     )
    #     assert os.path.isfile(output_filename)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape," "export_format",
        [(HelNet, 6, "channels_first", True, (3, 128, 256), "uff")],
    )
    def test_with_conv_gru_head(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        export_format,
    ):
        """Test that our model exports to PB and UFF."""
        self.common(
            tmpdir,
            model,
            nlayers,
            data_format,
            use_batch_norm,
            input_shape,
            export_format,
            add_conv_gru=True,
        )


class TestDataExport(object):
    """A class to test exporting tensors to a file."""

    @pytest.mark.parametrize(
        "nbatches, batch_size, input_shape, dtype, bit_depth",
        [(10, 6, (3, 64, 64), np.float32, 4)],
    )
    def test(self, tmpdir, nbatches, batch_size, input_shape, dtype, bit_depth):
        """Main test.

        Args:
            tmpdir (dir object): temp dir provided by pytest fixture.
            nbatches (int): number of batches to use.
            batch_size (int): number of samples per batch.
            input_shape (list): shape of each sample.
            dtype (dtype): data type to use.
            bit_depth (int): size (in bytes) of each element.
        """
        filename = os.path.join(str(tmpdir), "tensors.h5")
        batch_shape = (batch_size,) + input_shape
        dataset_shape = (nbatches,) + batch_shape

        dataset = np.random.random_sample(dataset_shape).astype(dtype)

        # Fill some batches with constants to test compression.
        dataset[0].fill(0)
        dataset[-1].fill(-1)

        # Dump data to file.
        with TensorFile(filename, "w") as f:
            for i in range(nbatches):
                f.write(dataset[i])

        assert os.path.isfile(filename)

        # Make sure some sort of compression happened.
        file_size = os.path.getsize(filename)
        assert file_size < dataset.size * bit_depth

        # Read back through sequential accesses.
        with TensorFile(filename, "r") as f:
            for batch_index, batch in enumerate(f):
                assert batch.dtype == dtype
                assert np.allclose(batch, dataset[batch_index])

        # Read back through random accesses.
        with TensorFile(filename, "r") as f:
            indices = list(range(nbatches))
            random.shuffle(indices)
            for idx in indices:
                f.seek(idx)
                batch = f.read()
                assert batch.dtype == dtype
                assert np.allclose(batch, dataset[idx])

    @pytest.mark.parametrize(
        "nbatches, batch_size, keys, input_shapes, dtype",
        [(4, 3, ["o1", "o2"], [(1, 24, 48), (4, 24, 48)], np.float32)],
    )
    def test_dict(self, tmpdir, nbatches, batch_size, keys, input_shapes, dtype):
        """Test that we can read/write dictionaries of numpy tensors.

        Args:
            tmpdir (dir object): temp dir provided by pytest fixture.
            nbatches (int): number of batches to use.
            batch_size (int): number of samples per batch.
            keys (list): list of dictionary keys.
            input_shapes (list): list of shapes.
            dtype (dtype): data type to use.
            bit_depth (int): size (in bytes) of each element.
        """
        filename = os.path.join(str(tmpdir), "tensors.h5")
        batch_shapes = [(batch_size,) + input_shape for input_shape in input_shapes]
        data_shapes = [(nbatches,) + batch_shape for batch_shape in batch_shapes]
        datasets = [
            np.random.random_sample(shape).astype(dtype) for shape in data_shapes
        ]

        # Dump data to file.
        with TensorFile(filename, "w", enforce_same_shape=False) as f:
            for i in range(nbatches):
                batch = {}
                for idx, key in enumerate(keys):
                    batch[key] = datasets[idx][i]
                f.write(batch)

        assert os.path.isfile(filename)

        # Read back through sequential accesses.
        with TensorFile(filename, "r", enforce_same_shape=False) as f:
            for batch_index, batch in enumerate(f):
                assert isinstance(batch, dict)
                for key_index, key in enumerate(keys):
                    assert batch[key].dtype == dtype
                    assert np.allclose(batch[key], datasets[key_index][batch_index])

        # Read back through random accesses.
        with TensorFile(filename, "r", enforce_same_shape=False) as f:
            indices = list(range(nbatches))
            random.shuffle(indices)
            for batch_index in indices:
                f.seek(batch_index)
                batch = f.read()
                assert isinstance(batch, dict)
                for key_index, key in enumerate(keys):
                    assert batch[key].dtype == dtype
                    assert np.allclose(batch[key], datasets[key_index][batch_index])

    @pytest.mark.parametrize("shape, dtype", [((1, 24, 48), np.float32)])
    def test_nested_dict(self, tmpdir, shape, dtype):
        """Test that we can read/write nested dictionaries of numpy tensors.

        Args:
            tmpdir (dir object): temp dir provided by pytest fixture.
            shape (list): tensor shape.
            dtype (dtype): data type to use.
            bit_depth (int): size (in bytes) of each element.
        """
        filename = os.path.join(str(tmpdir), "tensors.h5")

        batch = {
            "c1": {
                "o1": np.random.random_sample(shape).astype(dtype),
                "o2": np.random.random_sample(shape).astype(dtype),
            },
            "o1": np.random.random_sample(shape).astype(dtype),
        }

        # Dump data to file.
        with TensorFile(filename, "w") as f:
            f.write(batch)

        assert os.path.isfile(filename)

        # Read back through sequential accesses.
        with TensorFile(filename, "r") as f:
            read_batch = f.read()
            assert "o1" in read_batch
            assert np.allclose(batch["o1"], read_batch["o1"])
            assert "c1" in read_batch
            assert "o1" in read_batch["c1"]
            assert np.allclose(batch["c1"]["o1"], read_batch["c1"]["o1"])
            assert "o2" in read_batch["c1"]
            assert np.allclose(batch["c1"]["o2"], read_batch["c1"]["o2"])

    @pytest.mark.parametrize(
        "batch_sizes, input_shape, dtype, enforce_same_shape",
        [
            ([8, 8, 16, 8], (1, 28, 28), np.float32, True),
            ([8, 8, 16, 8], (1, 28, 28), np.float32, False),
        ],
    )
    def test_enforce_same_shape(
        self, tmpdir, batch_sizes, input_shape, dtype, enforce_same_shape
    ):
        """Test shape enforcement.

        Args:
            tmpdir (dir object): temp dir provided by pytest fixture.
            batch_sizes (list): list of batch sizes.
            input_shape (list): shape of each sample.
            dtype (dtype): data type to use.
            enforce_same_shape (bool): whether to enforce same shape.
        """
        filename = os.path.join(str(tmpdir), "tensors.h5")

        # Dump data to file.
        with TensorFile(
            filename, "w", enforce_same_shape=enforce_same_shape
        ) as f:
            prev_batch_size = batch_sizes[0]
            for batch_size in batch_sizes:
                batch_shape = (batch_size,) + input_shape
                batch = np.random.random_sample(batch_shape).astype(dtype)
                if enforce_same_shape and batch_size != prev_batch_size:
                    with pytest.raises(ValueError):
                        f.write(batch)
                else:
                    f.write(batch)

    def test_read_inexistant_file(self, tmpdir):
        """Test that an error is dropped when trying to open an inexistant file."""
        filename = os.path.join(str(tmpdir), "tensors.h5")

        with pytest.raises(IOError):
            with TensorFile(filename, "r"):
                pass

    def test_read_error(self, tmpdir):
        """Test that an error is dropped when trying to read from a write-only file."""
        filename = os.path.join(str(tmpdir), "tensors.h5")
        # Create the file.
        with TensorFile(filename, "w"):
            pass
        # Open for writing and try to read.
        with pytest.raises(IOError):
            with TensorFile(filename, "w") as f:
                f.read()

    def test_write_error(self, tmpdir):
        """Test that an error is dropped when trying to write to a read-only file."""
        filename = os.path.join(str(tmpdir), "tensors.h5")
        # Create the file.
        with TensorFile(filename, "w"):
            pass
        # Open for reading and try to write.
        with pytest.raises(IOError):
            with TensorFile(filename, "r") as f:
                f.write(np.zeros(10))


def keras_classification_model(num_samples=1000, nepochs=5, batch_size=10):
    # Create a dummy Keras classification model.
    # Define model.
    inputs = keras.layers.Input((1,))
    x = keras.layers.Dense(1, activation="linear")(inputs)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    m = keras.models.Model(inputs, outputs)

    # Compile model.
    optimizer = keras.optimizers.Adam(lr=0.01)
    m.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

    # Generate training data.
    x_train = np.linspace(0, 1, num=num_samples)
    np.random.shuffle(x_train)
    y_train = np.zeros((num_samples, 2))
    y_train[x_train > 0.5, 1] = 1
    y_train[x_train <= 0.5, 0] = 1

    # Train model and verify accuracy.
    res = m.fit(x_train, y_train, batch_size=batch_size, epochs=nepochs)
    logger.info("Keras Model average accuracy: %.3f", res.history["acc"][-1])
    assert res.history["acc"][-1] > 0.95

    n_batches = num_samples // batch_size
    x_batches = [
        x_train[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)
    ]
    y_batches = [
        y_train[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)
    ]

    return n_batches, batch_size, x_batches, y_batches, m


@pytest.fixture(scope="function")
def classification_model(num_samples=1000, nepochs=5, batch_size=10):
    # Test fixture to create a TensorRT engine.
    n_batches, batch_size, x_batches, y_batches, m = keras_classification_model()

    MAX_WORKSPACE = 1 << 28
    MAX_BATCHSIZE = 16

    _, out_tensor_name, engine = keras_to_tensorrt(
        m,
        input_dims=(1, 1, 1),
        max_workspace_size=MAX_WORKSPACE,
        max_batch_size=MAX_BATCHSIZE,
    )

    return n_batches, batch_size, x_batches, y_batches, out_tensor_name, engine


class TestTensorRTInference(object):
    """Test TensorRT inference."""

    def test_classify_iterator(self, tmpdir, classification_model):
        """Test TRT classification on trained model using iterator API."""
        assert cuda is not None, "CUDA not imported."
        assert trt is not None, "TRT not imported."

        n_batches, batch_size, x_batches, y_batches, out_tensor_name, engine = (
            classification_model
        )

        # Verify accuracy of TensorRT engine using the iterator API.
        acc_accuracy = 0
        for i, output in enumerate(engine.infer_iterator(x_batches)):
            labels = np.argmax(y_batches[i], axis=1)
            predictions = np.argmax(output[out_tensor_name].squeeze(), axis=1)
            accuracy = np.sum(predictions == labels) / float(batch_size)
            acc_accuracy += accuracy
        avg_accuracy = acc_accuracy / n_batches
        logger.info(
            "TensorRT Model average accuracy (iterator API): %.3f", avg_accuracy
        )
        assert avg_accuracy > 0.95
        # Very loose check that the reported forward time is sensible.
        forward_time = engine.get_forward_time()
        assert 0 < forward_time < 1

    def test_classify(self, tmpdir, classification_model):
        """Test TRT classification on trained model using single batch inference API."""
        n_batches, batch_size, x_batches, y_batches, out_tensor_name, engine = (
            classification_model
        )

        # Verify accuracy using the single array inference API.
        acc_accuracy = 0
        for i in range(n_batches):
            output = engine.infer(x_batches[i])
            labels = np.argmax(y_batches[i], axis=1)
            predictions = np.argmax(output[out_tensor_name].squeeze(), axis=1)
            accuracy = np.sum(predictions == labels) / float(batch_size)
            acc_accuracy += accuracy
        avg_accuracy = acc_accuracy / n_batches
        logger.info("TensorRT Model average accuracy: %.3f", avg_accuracy)
        assert avg_accuracy > 0.95

    def test_serialize_classify(self, tmpdir, classification_model):
        """Test TRT classification after serializing and reloading from file."""

        n_batches, batch_size, x_batches, y_batches, out_tensor_name, engine = (
            classification_model
        )

        # Serialize TensorRT engine to a file.
        trt_filename = os.path.join(str(tmpdir), "model.trt")
        engine.save(trt_filename)

        # Delete the engine, load the engine again from its serialized
        # representation and verify accuracy.
        del engine
        engine = nvidia_tao_tf1.core.export.load_tensorrt_engine(trt_filename)
        acc_accuracy = 0
        for i, output in enumerate(engine.infer_iterator(x_batches)):
            labels = np.argmax(y_batches[i], axis=1)
            predictions = np.argmax(output[out_tensor_name].squeeze(), axis=1)
            accuracy = np.sum(predictions == labels) / float(batch_size)
            acc_accuracy += accuracy
        avg_accuracy = acc_accuracy / n_batches
        logger.info("TensorRT Model average accuracy: %.3f", avg_accuracy)
        assert avg_accuracy > 0.95
        logger.info("Serialized TensorRT Model average accuracy: %.3f", avg_accuracy)

    def test_classification_int8(self, tmpdir):
        """Test TRT classification in reduce precision."""

        third_party.keras.tensorflow_backend.limit_tensorflow_GPU_mem(gpu_fraction=0.9)

        n_batches, batch_size, x_batches, y_batches, m = keras_classification_model()

        # Serialize input batches to file.
        tensor_filename = os.path.join(str(tmpdir), "tensor.dump")
        with TensorFile(tensor_filename, "w") as f:
            for x_batch in x_batches:
                f.write(x_batch)

        try:
            cal_cache_filename = os.path.join(str(tmpdir), "cal.bin")
            _, out_tensor_name, engine = keras_to_tensorrt(
                m,
                input_dims=(1, 1, 1),
                dtype="int8",
                calibration_data_filename=tensor_filename,
                calibration_cache_filename=cal_cache_filename,
                calibration_n_batches=n_batches,
                calibration_batch_size=batch_size,
            )
        except AttributeError as e:
            logger.warning(str(e))
            pytest.skip(str(e))

        # Verify accuracy of TensorRT engine using the iterator API.
        acc_accuracy = 0
        for i, output in enumerate(engine.infer_iterator(x_batches)):
            labels = np.argmax(y_batches[i], axis=1)
            predictions = np.argmax(output[out_tensor_name].squeeze(), axis=1)
            accuracy = np.sum(predictions == labels) / float(batch_size)
            acc_accuracy += accuracy
        avg_accuracy = acc_accuracy / n_batches
        logger.info(
            "TensorRT INT8 Model average accuracy (iterator API): %.3f", avg_accuracy
        )
        assert avg_accuracy > 0.95

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "batch_size, parser",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), 2, "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), 2, "caffe"),
            # TODO(MOD-435)
        ],
    )
    def test_net_output(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        batch_size,
        parser,
    ):
        """Create a model with random weights and match Keras and TensorRT output."""
        if parser == "onnx" and not _onnx_supported:
            return

        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            keras_model = model(
                nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
            )
            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "input_shape," "batch_size, concat_axis, parser",
        [((3, 48, 16), 2, 1, "caffe"), ((3, 64, 32), 4, 2, "caffe")],
    )
    def test_concat(self, tmpdir, input_shape, batch_size, concat_axis, parser):
        """Create a model with branches and a concat layer and match Keras and TensorRT output."""
        if concat_axis != 1:
            pytest.skip("TensorRT does not support concatenation on axis!=1.")

        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            inputs_relu = keras.layers.Activation("relu")(inputs)
            inputs_sigmoid = keras.layers.Activation("sigmoid")(inputs)
            inputs_softmax_with_axis = keras.layers.Softmax(axis=1)(inputs)
            net_output = keras.layers.Concatenate(axis=concat_axis)(
                [inputs_relu, inputs_sigmoid, inputs_softmax_with_axis]
            )
            keras_model = keras.models.Model(inputs=inputs, outputs=net_output)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "model, nlayers, input_shape, batch_size, use_batch_norm, data_format,"
        "padding, kernel_size, strides, parser",
        [
            (
                HelNet,
                6,
                (3, 96, 64),
                2,
                True,
                "channels_first",
                "valid",
                [1, 1],
                (1, 1),
                "caffe",
            ),
            (
                HelNet,
                6,
                (3, 96, 64),
                2,
                True,
                "channels_first",
                "valid",
                [1, 1],
                (1, 1),
                "uff",
            ),
            (
                HelNet,
                6,
                (3, 96, 64),
                2,
                True,
                "channels_first",
                "same",
                [1, 1],
                (1, 1),
                "caffe",
            ),
            (
                HelNet,
                6,
                (3, 96, 64),
                2,
                True,
                "channels_first",
                "same",
                [1, 1],
                (1, 1),
                "uff",
            ),
            (
                HelNet,
                6,
                (3, 96, 64),
                2,
                True,
                "channels_first",
                "valid",
                [3, 3],
                (1, 1),
                "caffe",
            ),
            (
                HelNet,
                6,
                (3, 96, 64),
                2,
                True,
                "channels_first",
                "valid",
                [3, 3],
                (1, 1),
                "uff",
            ),
            (
                HelNet,
                6,
                (3, 96, 64),
                2,
                True,
                "channels_first",
                "same",
                [3, 3],
                (1, 1),
                "caffe",
            ),
            (
                HelNet,
                6,
                (3, 96, 64),
                2,
                True,
                "channels_first",
                "same",
                [3, 3],
                (1, 1),
                "uff",
            ),
            # TODO(MOD-435)
        ],
    )
    def test_conv2dtranspose_layers(
        self,
        tmpdir,
        model,
        nlayers,
        input_shape,
        batch_size,
        use_batch_norm,
        data_format,
        padding,
        kernel_size,
        strides,
        parser,
    ):
        """
        Test that models with Conv2DTranspose layers convert to TensorRT correctly.

        Includes tests where kernel_size and strides are not equal, for both same and valid padding.
        """
        if parser == "onnx" and not _onnx_supported:
            return

        keras.backend.clear_session()
        third_party.keras.tensorflow_backend.limit_tensorflow_GPU_mem(gpu_fraction=0.9)
        with tf.device("cpu:0"):
            nvidia_tao_tf1.core.utils.set_random_seed(1)
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            keras_model = model(
                nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
            )
            x = keras_model.outputs[0]
            # Add Conv2DTranspose layer.
            # comment out this to walk around a bug in TRT 7.0/7.1.
            # see: https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=200603939&cmtNo=
            # num_filters = 4
            # x = keras.layers.Conv2DTranspose(
            #     filters=num_filters,
            #     kernel_size=kernel_size,
            #     strides=strides,
            #     padding=padding,
            #     activation="relu",
            #     name="conv2DTranspose",
            # )(x)
            keras_model = keras.models.Model(inputs=inputs, outputs=x)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "input_shape, batch_size, parser, use_gru",
        [
            ((3, 4, 6), 2, "caffe", False),
            ((3, 6, 4), 4, "uff", False),
            ((3, 6, 6), 1, "uff", True),
            ((3, 6, 4), 4, "onnx", False),
        ],
    )
    def test_multiple_outputs(self, tmpdir, input_shape, batch_size, parser, use_gru):
        """Create a model with multiple outputs and match and TensorRT output."""
        if parser == "onnx" and not _onnx_supported:
            return

        tf.reset_default_graph()
        keras.backend.clear_session()
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            if parser == 'onnx':
                inputs = keras.layers.Input(
                    name="input_1",
                    batch_shape=(batch_size,) + input_shape)
            else:
                inputs = keras.layers.Input(name="input_1", shape=input_shape)
            conv = keras.layers.Conv2D(16, (3, 3))(inputs)
            relu = keras.layers.Activation("relu", name="relu0")(conv)
            sigmoid = keras.layers.Activation("sigmoid", name="sigmoid0")(conv)
            keras_model = keras.models.Model(inputs=inputs, outputs=[relu, sigmoid])

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_outputs = keras_model.predict(data)

            if use_gru:
                assert (
                    parser == "uff"
                ), "Only UFF parser is supported for exporting GRU models."
                # An RNN/GRU TRT model needs extra input, that is the state
                # at last time step. This is needed as TRT cannot handle state internally.
                # State input shape that is aligned with the input shapes.
                state_input_shape = tuple(relu.shape.as_list()[1:])
                state_data_shape = (batch_size,) + state_input_shape
                state_data = np.random.random_sample(state_data_shape)
                # Create an export compatible convolutional GRU layer.
                convgru2d_export_layer = ConvGRU2DExport(
                    model_sequence_length_in_frames=1,
                    input_sequence_length_in_frames=1,
                    state_scaling=0.9,
                    input_shape=relu.shape.as_list(),
                    initial_state_shape=[None] + list(state_input_shape),
                    spatial_kernel_height=1,
                    spatial_kernel_width=1,
                    kernel_regularizer=None,
                    bias_regularizer=None,
                )

                gru_output = convgru2d_export_layer(relu)
                keras_model = keras.models.Model(
                    inputs=inputs, outputs=[relu, sigmoid, gru_output]
                )
                # Test the output of the end-to-end keraas model.
                # A session is used instead of keras.predict since a feed_dict is needed.
                with tf.compat.v1.Session() as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    keras_gru_output = sess.run(
                        gru_output,
                        feed_dict={
                            convgru2d_export_layer._initial_state: state_data,
                            inputs: data,
                        },
                    )

                # Add the state input for GRU input to input data and dimensions.
                input_dims = {
                    "input_1": input_shape,
                    "conv_gru_2d_export/state_placeholder": state_input_shape,
                }
                input_data = {
                    "input_1": data,
                    "conv_gru_2d_export/state_placeholder": state_data,
                }
                keras_outputs.append(keras_gru_output)
            else:
                # Single input (no sate input).
                input_dims = input_shape
                input_data = data

            output_layer_names = ["relu0", "sigmoid0"]
            output_layers = [keras_model.get_layer(name) for name in output_layer_names]

            if parser == "uff":
                # Provide TensorFlow tensor names.
                output_node_names = [
                    l.get_output_at(0).name.split(":")[0] for l in output_layers
                ]
            elif parser == "onnx":
                output_node_names = [l.get_output_at(0).name.split("/")[0] for l in output_layers]
            elif parser == "caffe":
                output_node_names = []
                for l in output_layers:
                    if hasattr(l, "activation") and l.activation.__name__ != "linear":
                        output_node_names.append(
                            "%s/%s" % (l.name, l.activation.__name__.capitalize())
                        )
                    else:
                        output_node_names.append(l.name)
            else:
                raise ValueError("Unknown parser: %s" % parser)

            # Pass ConvGRU2DExport as the custom object to the exporter if gru is being used.
            custom_objects = {"ConvGRU2DExport": ConvGRU2DExport} if use_gru else None
            if use_gru:
                output_node_names += ["conv_gru_2d_export/state_output"]

            _, _, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_dims,
                max_batch_size=batch_size,
                parser=parser,
                output_node_names=output_node_names,
                custom_objects=custom_objects,
            )
            inferred_data = engine.infer(input_data)
            print("INFER: {}".format(list(inferred_data.keys())))
            tensorrt_outputs = [inferred_data[name] for name in output_node_names]
            # Check that keras and TRT model outputs are aligned and match correspondingly.
            assert len(keras_outputs) == len(tensorrt_outputs)
            assert all(
                [a.shape == b.shape for (a, b) in zip(keras_outputs, tensorrt_outputs)]
            )
            assert all(
                [
                    np.allclose(a, b, atol=1e-2)
                    for (a, b) in zip(keras_outputs, tensorrt_outputs)
                ]
            )

    @pytest.mark.parametrize(
        "batch_size, parser, use_gru",
        [(2, "caffe", False), (4, "uff", False), (1, "uff", True), (4, "onnx", False)],
    )
    def test_multiple_inputs(self, tmpdir, batch_size, parser, use_gru):
        """
        Create a model with multiple inputs and match and TensorRT output.

        Also create caffe model and make sure input layer names match keras inputs.
        """
        if parser == "onnx" and not _onnx_supported:
            return

        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.

            # Creating a model that takes two inputs where the first input
            # is twice as large as the second input in each dimension.
            # The first input goes through a conv layer with stride 2.
            # The second input goes through a conv layer with stride 1.
            # Outputs of the conv layers are then concatenated before going through
            # a final convolution.
            if parser == 'onnx':
                input_0_shape = (batch_size, 3, 32, 64)
                input_1_shape = (batch_size, 3, 16, 32)
                input_0 = keras.layers.Input(batch_shape=input_0_shape, name="input_0")
                input_1 = keras.layers.Input(batch_shape=input_1_shape, name="input_1")
            else:
                input_0_shape = (3, 32, 64)
                input_1_shape = (3, 16, 32)
                input_0 = keras.layers.Input(shape=input_0_shape, name="input_0")
                input_1 = keras.layers.Input(shape=input_1_shape, name="input_1")

            conv_0 = keras.layers.Conv2D(16, (1, 1), strides=(2, 2))(input_0)
            conv_1 = keras.layers.Conv2D(16, (1, 1), strides=(1, 1))(input_1)
            # add = keras.layers.Add()([conv_0, conv_1])
            merged = keras.layers.Concatenate(axis=1, name="concat")([conv_0, conv_1])
            conv = keras.layers.Conv2D(8, (3, 3))(merged)
            net_output = keras.layers.Activation("relu", name="relu0")(conv)
            keras_model = keras.models.Model(
                inputs=[input_0, input_1], outputs=net_output
            )
            if parser != 'onnx':
                data_0_shape = (batch_size,) + input_0_shape
                data_1_shape = (batch_size,) + input_1_shape
            else:
                data_0_shape = input_0_shape
                data_1_shape = input_1_shape
            data_0 = np.random.random_sample(data_0_shape)
            data_1 = np.random.random_sample(data_1_shape)

            if use_gru:
                assert (
                    parser == "uff"
                ), "Only UFF parser is supported for exporting GRU models."
                # An RNN/GRU TRT model the state at last time step. as extra input.
                # (As TRT cannot handle state internally).
                # State input shape that is aligned with the input shapes.
                state_input_shape = (8, 14, 30)
                state_data_shape = (batch_size,) + state_input_shape
                state_data = np.random.random_sample(state_data_shape)
                # Create an export compatible convolutional GRU layer.
                convgru2d_export_layer = ConvGRU2DExport(
                    model_sequence_length_in_frames=1,
                    input_sequence_length_in_frames=1,
                    state_scaling=0.9,
                    input_shape=net_output.shape.as_list(),
                    initial_state_shape=[None] + list(state_input_shape),
                    spatial_kernel_height=1,
                    spatial_kernel_width=1,
                    kernel_regularizer=None,
                    bias_regularizer=None,
                )
                gru_output = convgru2d_export_layer(net_output)
                keras_model = keras.models.Model(
                    inputs=[input_0, input_1], outputs=gru_output
                )
                # Test the output of the end-to-end keraas model.
                # A session is used instead of keras.predict since a feed_dict is needed.
                with tf.compat.v1.Session() as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    keras_output = sess.run(
                        gru_output,
                        feed_dict={
                            convgru2d_export_layer._initial_state: state_data,
                            input_0: data_0,
                            input_1: data_1,
                        },
                    )
                # Add the state input for GRU input to input data and dimensions.
                input_dims = {
                    "input_0": input_0_shape,
                    "input_1": input_1_shape,
                    "conv_gru_2d_export/state_placeholder": state_input_shape,
                }
                input_data = {
                    "input_0": data_0,
                    "input_1": data_1,
                    "conv_gru_2d_export/state_placeholder": state_data,
                }
            else:
                keras_output = keras_model.predict([data_0, data_1])
                input_dims = {"input_0": input_0_shape, "input_1": input_1_shape}
                input_data = {"input_0": data_0, "input_1": data_1}

            # If the parser is caffe, check that caffe and keras input layer names match.
            if parser == "caffe":
                caffe_in_names, _ = nvidia_tao_tf1.core.export.keras_to_caffe(
                    keras_model,
                    os.path.join(str(tmpdir) + "/model.prototxt"),
                    os.path.join(str(tmpdir) + "/model.caffemodel"),
                )
                assert all(
                    [caffe_name in input_dims.keys() for caffe_name in caffe_in_names]
                )

            custom_objects = {"ConvGRU2DExport": ConvGRU2DExport} if use_gru else None
            # Check that keras and TRT model outputs match after inference.
            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_dims,
                max_batch_size=batch_size,
                parser=parser,
                custom_objects=custom_objects,
            )

            inferred_data = engine.infer(input_data)
            tensorrt_output = inferred_data[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape, "
        "batch_size, parser",
        [(HelNet, 6, "channels_first", True, (3, 128, 256), 1, "uff")],
    )
    def test_net_output_with_gru(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        batch_size,
        parser,
    ):
        """
        Create a model with random weights and match Keras and TensorRT output.

        The model includes a GRU layer and hence has an external state input."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape, name="input_1")
            feature_model = model(
                nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
            )

            # Get the feature outputs (Input to the GRU).
            feature_out = feature_model.outputs[0]
            feature_out_shape = feature_out.shape.as_list()
            gru_input_shape = tuple(feature_out_shape[1:])

            # Shape and random input data for the GRU/RNN state.
            state_data_shape = (batch_size,) + tuple(feature_out.shape.as_list()[1:])
            state_data = np.float32(np.random.random_sample(state_data_shape))
            # Shape and random input data for non-state input-
            data_shape = (batch_size,) + input_shape
            data = np.float32(np.random.random_sample(data_shape))

            # Add the convolutinal GRU export layer.
            convgru2d_export_layer = ConvGRU2DExport(
                model_sequence_length_in_frames=1,
                input_sequence_length_in_frames=1,
                state_scaling=0.9,
                input_shape=feature_out_shape,
                initial_state_shape=feature_out_shape,
                spatial_kernel_height=1,
                spatial_kernel_width=1,
                kernel_regularizer=None,
                bias_regularizer=None,
                is_stateful=True,
            )
            net_output = convgru2d_export_layer(feature_out)
            # Construct end-to-end keras model.
            keras_model = keras.models.Model(inputs=inputs, outputs=net_output)

            # Test the output of the end-to-end keraas model.
            # A session is used instead of keras.predict since a feed_dict is needed.
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                keras_output_value = sess.run(
                    net_output,
                    feed_dict={
                        convgru2d_export_layer._initial_state: state_data,
                        inputs: data,
                    },
                )
            # Input dims is now a dictionary, just as in the case of multiple inputs.
            input_dims = {
                "input_1": input_shape,
                "conv_gru_2d_export/state_placeholder": gru_input_shape,
            }
            print(f"Input dimensions: {input_dims}")
            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_dims,
                max_batch_size=batch_size,
                parser=parser,
                custom_objects={"ConvGRU2DExport": ConvGRU2DExport},
            )
            # Input data for TRT inference.
            input_data = {
                "conv_gru_2d_export/state_placeholder": state_data,
                "input_1": data,
            }

            tensorrt_output = engine.infer(input_data)[out_tensor_name]
            assert keras_output_value.shape == tensorrt_output.shape
            assert np.allclose(keras_output_value, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "batch_size, dropout_type, parser",
        [
            (HelNet, 6, "channels_first", True, (3, 128, 256), 2, "dropout", "uff"),
            (HelNet, 6, "channels_first", True, (3, 128, 256), 2, "dropout", "caffe"),
            (
                HelNet,
                6,
                "channels_first",
                True,
                (3, 128, 256),
                2,
                "spatial_dropout_2d",
                "uff",
            ),
            (
                HelNet,
                6,
                "channels_first",
                True,
                (3, 128, 256),
                2,
                "spatial_dropout_2d",
                "caffe",
            ),
        ],
    )
    def test_dropout(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        batch_size,
        dropout_type,
        parser,
    ):
        """Test the models with dropout convert to TensorRT correctly."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            keras_model = model(
                nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
            )
            x = keras_model.outputs[0]

            assert dropout_type in ["dropout", "spatial_dropout_2d"]
            if dropout_type == "dropout":
                x = keras.layers.Dropout(rate=0.5)(x)
            elif dropout_type == "spatial_dropout_2d":
                x = keras.layers.SpatialDropout2D(rate=0.5)(x)

            x = keras.layers.Conv2D(32, (3, 3), name="conv2d_output", padding="same")(x)
            keras_model = keras.models.Model(inputs=inputs, outputs=x)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.skipif(
        os.getenv("RUN_ON_CI", "0") == "1",
        reason="Cannot be run on CI"
    )
    @pytest.mark.parametrize(
        "input_shape, batch_size, parser",
        [((3, 15, 30), 2, "caffe"), ((3, 15, 30), 2, "uff")],
    )
    def test_dense_dropout(self, tmpdir, input_shape, batch_size, parser):
        """Test the models with dropout after a dense layer convert to TensorRT correctly."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            x = keras.layers.Flatten()(inputs)
            x = keras.layers.Dense(128, activation="tanh")(x)
            x = keras.layers.Dropout(rate=0.5)(x)
            x = keras.layers.Dense(128, activation="tanh")(x)
            keras_model = keras.models.Model(inputs=inputs, outputs=x)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name].squeeze()
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "padding, batch_size, parser",
        [
            (HelNet, 6, "channels_first", True, (3, 96, 64), "valid", 2, "uff"),
            (HelNet, 6, "channels_first", True, (3, 160, 64), "valid", 5, "caffe"),
        ],
    )
    def test_eltwise_op_layers(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        padding,
        batch_size,
        parser,
    ):
        """Test the models with max pooling convert to TensorRT correctly."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            keras_model = model(
                nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
            )
            x = keras_model.outputs[0]
            # Branches for distinct add, subtract, and multiply layers.
            # First branch.
            num_filters = 4
            x1 = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                activation="sigmoid",
                name="conv2d_x1",
            )(x)
            # Second branch.
            x2 = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                activation="relu",
                name="conv2d_x2",
            )(x)

            # Third branch.
            x3 = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                activation="linear",
                name="conv2d_x3",
            )(x)
            # Fourth branch.
            x4 = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                activation="tanh",
                name="conv2d_x4",
            )(x)

            # Add, Subtract, and Multiply
            x = keras.layers.Add()([x1, x2])
            x = keras.layers.Subtract()([x, x3])
            x = keras.layers.Multiply()([x, x4])
            # walk around of a a bug in TRT7.0 by setting the branches as output nodes
            # see https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=200602766&cmtNo=
            # this bug is fixed in TRT 7.1, so remove this trick once we upgrade to TRT 7.1
            keras_model = keras.models.Model(inputs=inputs, outputs=[x, x1, x2, x3, x4])

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name[0]]
            assert keras_output[0].shape == tensorrt_output.shape
            assert np.allclose(keras_output[0], tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "batch_size, parser",
        [
            (HelNet, 6, "channels_first", True, (3, 96, 64), 4, "uff"),
            (HelNet, 6, "channels_first", True, (3, 160, 64), 2, "caffe"),
        ],
    )
    def test_eltwise_op_with_broadcast(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        batch_size,
        parser,
    ):
        """Test the models with broadcast element-wise op converts to TensorRT."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            keras_model = model(
                nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
            )
            x = keras_model.outputs[0]

            # Project to one channel.
            x_single_channel = keras.layers.Conv2D(
                filters=1,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                activation="sigmoid",
            )(x)

            # Branches for distinct add, subtract, and multiply layers.
            # First branch.
            num_filters = 4
            x1 = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                activation="sigmoid",
                name="conv2d_x1",
            )(x)
            x1 = keras.layers.Add()([x1, x_single_channel])

            # Second branch.
            x2 = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                activation="relu",
                name="conv2d_x2",
            )(x)
            x2 = keras.layers.Subtract()([x2, x_single_channel])

            # Third branch.
            x3 = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[1, 1],
                strides=(1, 1),
                padding="same",
                activation="linear",
                name="conv2d_x3",
            )(x)
            x3 = keras.layers.Multiply()([x3, x_single_channel])

            # Add them all together
            x = keras.layers.Add()([x1, x2])
            x = keras.layers.Add()([x, x3])
            keras_model = keras.models.Model(inputs=inputs, outputs=x)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "data_format, input_shape, num_outbound_nodes_after_pad2d,"
        "conv2d_pad_type, conv1_strides, conv2_strides, zeropad2d_padding,"
        " batch_size, parser",
        [
            (
                "channels_first",
                (3, 96, 64),
                0,
                "valid",
                (1, 1),
                (1, 1),
                (2, 2),
                2,
                "caffe",
            ),
            (
                "channels_first",
                (3, 32, 32),
                1,
                "same",
                (2, 2),
                (2, 2),
                (1, 1),
                2,
                "caffe",
            ),
            (
                "channels_first",
                (3, 96, 64),
                2,
                "valid",
                (2, 2),
                (2, 2),
                (1, 1),
                2,
                "caffe",
            ),
            (
                "channels_first",
                (3, 48, 48),
                0,
                "same",
                (1, 1),
                (1, 1),
                (2, 2),
                2,
                "uff",
            ),
            (
                "channels_first",
                (3, 128, 64),
                1,
                "valid",
                (2, 2),
                (2, 2),
                (1, 1),
                2,
                "uff",
            ),
            (
                "channels_first",
                (3, 64, 128),
                2,
                "same",
                (2, 2),
                (2, 2),
                (1, 1),
                2,
                "uff",
            ),
            (
                "channels_first",
                (3, 96, 64),
                0,
                "valid",
                (2, 2),
                (1, 1),
                (2, 2),
                2,
                "caffe",
            ),
            (
                "channels_first",
                (3, 32, 32),
                1,
                "same",
                (1, 1),
                (2, 2),
                (1, 1),
                2,
                "caffe",
            ),
            (
                "channels_first",
                (3, 96, 64),
                2,
                "valid",
                (1, 1),
                (2, 2),
                (1, 1),
                2,
                "caffe",
            ),
            (
                "channels_first",
                (3, 48, 48),
                0,
                "same",
                (2, 2),
                (1, 1),
                (2, 2),
                2,
                "uff",
            ),
            (
                "channels_first",
                (3, 128, 64),
                1,
                "valid",
                (1, 1),
                (2, 2),
                (1, 1),
                2,
                "uff",
            ),
            (
                "channels_first",
                (3, 64, 128),
                2,
                "same",
                (1, 1),
                (2, 2),
                (1, 1),
                2,
                "uff",
            ),
        ],
    )
    def test_zeropad2d_after_conv2d(
        self,
        tmpdir,
        data_format,
        input_shape,
        num_outbound_nodes_after_pad2d,
        conv2d_pad_type,
        conv1_strides,
        conv2_strides,
        zeropad2d_padding,
        batch_size,
        parser,
    ):
        """Test the models with ZeroPadding2D after conv2d."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)

            x = keras.layers.Conv2D(32, (3, 3), padding="same", strides=conv1_strides)(
                inputs
            )

            x = keras.layers.convolutional.ZeroPadding2D(
                padding=zeropad2d_padding, data_format=data_format
            )(x)

            x = keras.layers.Conv2D(
                32, (3, 3), padding=conv2d_pad_type, strides=conv2_strides
            )(x)

            if num_outbound_nodes_after_pad2d == 1:
                x = keras.layers.Activation("relu")(x)
            elif num_outbound_nodes_after_pad2d == 2:
                x1 = keras.layers.Activation("relu")(x)
                x2 = keras.layers.Activation("relu")(x)
                x = keras.layers.Add()([x1, x2])
            elif num_outbound_nodes_after_pad2d != 0:
                raise ValueError(
                    "Unhandled num_outbound_nodes_after_pad2d: %d"
                    % num_outbound_nodes_after_pad2d
                )

            keras_model = keras.models.Model(inputs=inputs, outputs=x)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "model, nlayers, data_format, use_batch_norm, input_shape,"
        "padding, pooling_type, batch_size, parser",
        [
            (HelNet, 6, "channels_first", True, (3, 96, 64), "valid", "AVE", 2, "uff"),
            (HelNet, 6, "channels_first", True, (3, 64, 96), "same", "AVE", 3, "uff"),
            (
                HelNet,
                6,
                "channels_first",
                True,
                (3, 128, 128),
                "same",
                "AVE",
                4,
                "caffe",
            ),
            (
                HelNet,
                6,
                "channels_first",
                True,
                (3, 160, 64),
                "valid",
                "AVE",
                5,
                "caffe",
            ),
            (HelNet, 6, "channels_first", True, (3, 96, 64), "valid", "MAX", 2, "uff"),
            (HelNet, 6, "channels_first", True, (3, 64, 96), "same", "MAX", 3, "uff"),
            (
                HelNet,
                6,
                "channels_first",
                True,
                (3, 128, 128),
                "same",
                "MAX",
                4,
                "caffe",
            ),
            (
                HelNet,
                6,
                "channels_first",
                True,
                (3, 160, 64),
                "valid",
                "MAX",
                5,
                "caffe",
            ),
        ],
    )
    def test_pooling(
        self,
        tmpdir,
        model,
        nlayers,
        data_format,
        use_batch_norm,
        input_shape,
        padding,
        pooling_type,
        batch_size,
        parser,
    ):
        """Test the models with average and max pooling convert to TensorRT correctly."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            keras_model = model(
                nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
            )
            x = keras_model.outputs[0]

            assert pooling_type in ["AVE", "MAX"]
            if pooling_type == "AVE":
                x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding=padding)(x)
            elif pooling_type == "MAX":
                x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding=padding)(x)

            x = keras.layers.Conv2D(32, (3, 3), name="conv2d_output", padding="same")(x)
            keras_model = keras.models.Model(inputs=inputs, outputs=x)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize("model", [HelNet])
    @pytest.mark.parametrize("nlayers", [6])
    # NOTE: input_shape (3, 64, 64) currently fails, likely due to some error while parsing the UFF
    # or Caffe model back into TRT, as Tensorflow, TRT, Keras **all** have outputs that match before
    # export.
    @pytest.mark.parametrize("input_shape", [(3, 96, 96)])
    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize("use_batch_norm", [True, False])
    @pytest.mark.parametrize("data_format", ["channels_first"])
    @pytest.mark.parametrize("parser", ["caffe", "uff"])
    @pytest.mark.parametrize(
        "kernel_size,dilation_rate",
        [((1, 1), (1, 1)), ((3, 3), (1, 1)), ((3, 3), (2, 2))],
    )
    def test_conv2d_dilation_layers(
        self,
        tmpdir,
        model,
        nlayers,
        input_shape,
        batch_size,
        use_batch_norm,
        data_format,
        kernel_size,
        parser,
        dilation_rate,
    ):
        """
        Test that models with Conv2D layers with dilation convert to TensorRT correctly.
        """
        keras.backend.clear_session()
        third_party.keras.tensorflow_backend.limit_tensorflow_GPU_mem(gpu_fraction=0.9)
        with tf.device("cpu:0"):
            nvidia_tao_tf1.core.utils.set_random_seed(1)
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            keras_model = model(
                nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
            )
            x = keras_model.outputs[0]
            # Add Conv2D layer with dilation.
            num_filters = 4
            y = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                data_format=data_format,
                activation="relu",
                padding="same",
                name="conv2D_dilated",
            )(x)
            keras_model = keras.models.Model(inputs=inputs, outputs=y)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    # @pytest.mark.parametrize(
    #     "model, nlayers, data_format, use_batch_norm, input_shape,"
    #     "padding, pooling_type, batch_size, parser",
    #     [
    #         (HelNet, 6, "channels_first", True, (3, 96, 64), "valid", "AVE", 2, "uff"),
    #         (
    #             HelNet,
    #             6,
    #             "channels_first",
    #             True,
    #             (3, 128, 128),
    #             "same",
    #             "AVE",
    #             4,
    #             "caffe",
    #         ),
    #     ],
    # )
    # @pytest.mark.script_launch_mode("subprocess")
    # def test_app(
    #     self,
    #     script_runner,
    #     tmpdir,
    #     model,
    #     nlayers,
    #     data_format,
    #     use_batch_norm,
    #     input_shape,
    #     padding,
    #     pooling_type,
    #     batch_size,
    #     parser,
    # ):
    #     """Test TRT export from app."""
    #     with tf.device("cpu:0"):
    #         # Creating graph on CPU to leave GPU memory to TensorRT.
    #         keras_filename = os.path.join(str(tmpdir), "model.h5")
    #         trt_filename = os.path.join(str(tmpdir), "model.trt")
    #         inputs = keras.layers.Input(shape=input_shape)
    #         m = model(
    #             nlayers, inputs, use_batch_norm=use_batch_norm, data_format=data_format
    #         )
    #         m.save(keras_filename)

    #         extra_args = []
    #         extra_args.extend(["--input_dims", ",".join([str(i) for i in input_shape])])
    #         extra_args.extend(["--parser", parser])
    #         extra_args.extend(["--format", "tensorrt"])
    #         extra_args.extend(["--output_file", trt_filename])
    #         extra_args.extend(["--random_data"])
    #         extra_args.extend(["-v"])

    #         env = os.environ.copy()
    #         env["CUDA_VISIBLE_DEVICES"] = "0"
    #         script = "app.py"
    #         # Path adjustment for bazel tests
    #         if os.path.exists(os.path.join("nvidia_tao_tf1/core/export", script)):
    #             script = os.path.join("nvidia_tao_tf1/core/export", script)
    #         ret = script_runner.run(script, keras_filename, env=env, *extra_args)

    #         assert ret.success, "Process returned error: %s error trace: %s" % (
    #             ret.success,
    #             ret.stderr,
    #         )
    #         assert os.path.isfile(trt_filename)

    #         assert "Elapsed time" in ret.stderr, "Read: %s" % ret.stderr

    @pytest.mark.parametrize(
        "model, nlayers, input_shape, batch_size, parser",
        [(HelNet, 6, (3, 96, 64), 2, "uff"), (HelNet, 6, (3, 128, 128), 4, "caffe")],
    )
    def test_3d_softmax(self, tmpdir, model, nlayers, input_shape, batch_size, parser):
        """Test the models with average and max pooling convert to TensorRT correctly."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            keras_model = model(nlayers, inputs, data_format="channels_first")
            x = keras_model.outputs[0]
            # walk around of a bug in TRT7.0/7.1 at this moment
            # see https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=200603619&cmtNo=
            # remove this trick once this bug is fixed in a future version of TRT
            # x = keras.layers.Softmax(axis=1)(x)

            keras_model = keras.models.Model(inputs=inputs, outputs=x)

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = keras_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                keras_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name]
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)

    @pytest.mark.parametrize(
        "fp32_layer_names, fp16_layer_names, \
                              expected_fp32_layer_names, expected_fp16_layer_names",
        [
            (
                [],
                ["softmax_1"],
                [],
                ["softmax_1/transpose", "softmax_1/Softmax", "softmax_1/transpose_1"],
            ),
            (
                ["softmax_1"],
                [],
                ["softmax_1/transpose", "softmax_1/Softmax", "softmax_1/transpose_1"],
                [],
            ),
        ],
    )
    def test_mixed_precision(
        self,
        tmpdir,
        fp32_layer_names,
        fp16_layer_names,
        expected_fp32_layer_names,
        expected_fp16_layer_names,
    ):
        """Test INT8 based mixed precision inference."""

        input_shape = (3, 96, 64)
        layer_num = 6
        batch_size = 16
        nbatches = 1

        # Define the model.
        inputs = keras.layers.Input(shape=input_shape)
        keras_model = HelNet(layer_num, inputs, data_format="channels_first")
        x = keras_model.outputs[0]
        x = keras.layers.Softmax(axis=1)(x)
        keras_model = keras.models.Model(inputs=inputs, outputs=x)

        # Prepare calibration data and save to a file.
        tensor_filename = os.path.join(str(tmpdir), "tensor.dump")
        with TensorFile(tensor_filename, "w") as f:
            cali_data = np.random.randn(
                batch_size, input_shape[0], input_shape[1], input_shape[2]
            )
            f.write(cali_data)

        # Prepare a dummy calibration table file.
        cal_cache_filename = os.path.join(str(tmpdir), "cal.bin")

        # Export TensorRT mixed precision model, and check the correctness.
        try:
            with mock.patch(
                "nvidia_tao_tf1.core.export._tensorrt._set_layer_precision",
                side_effect=_set_excluded_layer_precision,
            ) as spied_set_excluded_layer_precision:
                _, _, _ = keras_to_tensorrt(
                    keras_model,
                    dtype="int8",
                    input_dims=input_shape,
                    max_batch_size=2,
                    calibration_data_filename=tensor_filename,
                    calibration_cache_filename=cal_cache_filename,
                    calibration_n_batches=nbatches,
                    calibration_batch_size=batch_size,
                    fp32_layer_names=fp32_layer_names,
                    fp16_layer_names=fp16_layer_names,
                    parser="uff",
                )

                arg_fp32_layer_names = spied_set_excluded_layer_precision.call_args[1][
                    "fp32_layer_names"
                ]
                arg_fp16_layer_names = spied_set_excluded_layer_precision.call_args[1][
                    "fp16_layer_names"
                ]
                arg_network = spied_set_excluded_layer_precision.call_args[1]["network"]

                res_fp32_layer_names = []
                res_fp16_layer_names = []
                for layer in arg_network:
                    if layer.precision == trt.float32:
                        res_fp32_layer_names.append(layer.name)
                    elif layer.precision == trt.float16:
                        res_fp16_layer_names.append(layer.name)

                assert arg_fp32_layer_names == fp32_layer_names
                assert arg_fp16_layer_names == fp16_layer_names
                assert res_fp32_layer_names == expected_fp32_layer_names
                assert res_fp16_layer_names == expected_fp16_layer_names
        except AttributeError as e:
            logger.warning(str(e))
            pytest.skip(str(e))

    @pytest.mark.parametrize(
        "input_shape, batch_size, parser", [((3, 15, 30), 2, "uff")]
    )
    def test_model_in_model(self, tmpdir, input_shape, batch_size, parser):
        """Test the model-in-model converts to TensorRT correctly."""
        with tf.device("cpu:0"):
            # Creating graph on CPU to leave GPU memory to TensorRT.
            inputs = keras.layers.Input(shape=input_shape)
            x = keras.layers.Flatten()(inputs)
            x = keras.layers.Dense(128, activation="tanh")(x)
            inner_model = keras.models.Model(inputs=inputs, outputs=x)

            outer_inputs = keras.layers.Input(shape=input_shape)
            outer_model = keras.models.Model(
                inputs=outer_inputs, outputs=inner_model(outer_inputs)
            )

            data_shape = (batch_size,) + input_shape
            data = np.random.random_sample(data_shape)
            keras_output = outer_model.predict(data)

            _, out_tensor_name, engine = keras_to_tensorrt(
                outer_model,
                input_dims=input_shape,
                max_batch_size=batch_size,
                parser=parser,
            )
            tensorrt_output = engine.infer(data)[out_tensor_name].squeeze()
            assert keras_output.shape == tensorrt_output.shape
            assert np.allclose(keras_output, tensorrt_output, atol=1e-2)


@pytest.mark.parametrize("floatx", ["float32", "float16"])
def test_get_model_input_dtype(tmpdir, floatx):
    """Test that get_model_input_dtype function returns the correct dtype."""
    try:
        model_filename = os.path.join(str(tmpdir), "model.h5")

        keras.backend.set_floatx(floatx)
        inputs = keras.layers.Input(shape=(4,))
        x = keras.layers.Dense(4)(inputs)
        y = keras.layers.Dense(4)(x)
        model = keras.models.Model(inputs=inputs, outputs=y)
        model.save(model_filename)

        dtype = get_model_input_dtype(model_filename)

        assert dtype == floatx
    finally:
        # Set Keras float type to the default float32, so that other tests are not affected.
        keras.backend.set_floatx("float32")
