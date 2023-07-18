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

"""DriveIX common utils used across all apps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import logging
import os
import struct
import sys
import tempfile
from zipfile import BadZipFile, ZipFile
import keras
import onnx
import tensorflow as tf
from tensorflow.compat.v1 import GraphDef
import tf2onnx

from nvidia_tao_tf1.core.export import (
    keras_to_onnx,
    keras_to_pb,
    keras_to_uff
)
from nvidia_tao_tf1.encoding import encoding


ENCRYPTION_OFF = False

# logger = logging.getLogger(__name__)


def encode_from_keras(
    keras_model,
    output_filename,
    enc_key,
    only_weights=False,
    custom_objects=None
):
    """A simple function to encode a keras model into magnet export format.

    Args:
        keras_model (keras.models.Model object): The input keras model to be encoded.
        output_filename (str): The name of the encoded output file.
        enc_key (bytes): Byte text to encode the model.
        custom_objects(dict): Custom objects for serialization and deserialization.

    Returns:
        None
    """
    # Make sure that input model is a keras model object.
    if not isinstance(keras_model, keras.models.Model):
        raise TypeError("The model should be a keras.models.Model object")

    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)

    # Create a temporary model file for the keras model.
    if only_weights:
        keras_model.save_weights(temp_file_name)
    else:
        keras_model.save(temp_file_name)

    # Encode the keras model file.
    with open(output_filename, 'wb') as outfile, open(temp_file_name, 'rb') as infile:
        encoding.encode(infile, outfile, enc_key)
    infile.closed
    outfile.closed
    # Remove the temporary keras file.
    os.remove(temp_file_name)


def get_decoded_filename(input_file_name, enc_key, custom_objects=None):
    """Extract keras model file and get model dtype.

    Args:
        input_file_name (str): Path to input model file.
        enc_key (bytes): Byte text to decode model.
        custom_objects(dict): Custom objects for serialization and deserialization.

    Returns:
        model_dtype: Return the decoded model filename.
    """
    if input_file_name.endswith(".hdf5"):
        return input_file_name

    # Check if input file exists.
    if not os.path.isfile(input_file_name):
        raise ValueError("Cannot find input file name.")

    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)

    with open(temp_file_name, 'wb') as temp_file, open(input_file_name, 'rb') as encoded_file:
        encoding.decode(encoded_file, temp_file, enc_key)
    encoded_file.closed
    temp_file.closed

    # Check if the model is valid hdf5
    try:
        keras.models.load_model(temp_file_name, compile=False, custom_objects=custom_objects)
    except IOError:
        raise IOError("Invalid decryption. {}. The key used to load the model "
                      "is incorrect.".format(sys.exc_info()[1]))
    except ValueError:
        raise ValueError("Invalid decryption. {}. The key used to load the model "
                         "is incorrect.".format(sys.exc_info()[1]))

    return temp_file_name


def decode_to_keras(input_file_name, enc_key,
                    input_model=None, compile_model=True, by_name=True,
                    custom_objects=None):
    """A simple function to decode an encrypted file to a keras model.

    Args:
        input_file_name (str): Path to encoded input file.
        enc_key (bytes): Byte text to decode the model.
        custom_objects(dict): Custom objects for serialization and deserialization.

    Returns:
        decrypted_model (keras.models.Model): Returns a decrypted keras model.
    """
    # Check if input file exists.
    if not os.path.isfile(input_file_name):
        raise ValueError("Cannot find input file name.")

    if input_file_name.endswith("hdf5"):
        if input_model is None:
            return keras.models.load_model(input_file_name,
                                           compile=compile_model,
                                           custom_objects=custom_objects)
        assert isinstance(input_model, keras.models.Model), (
            "Input model not a valid Keras model."
        )
        input_model.load_weights(input_file_name, by_name=by_name, custom_objects=custom_objects)
        return input_model

    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)

    with open(temp_file_name, 'wb') as temp_file, open(input_file_name, 'rb') as encoded_file:
        encoding.decode(encoded_file, temp_file, enc_key)
    encoded_file.closed
    temp_file.closed

    if input_model is None:
        try:
            decrypted_model = keras.models.load_model(temp_file_name,
                                                      compile=compile_model,
                                                      custom_objects=custom_objects)
        except IOError:
            raise IOError("Invalid decryption. {}. The key used to load the model "
                          "is incorrect.".format(sys.exc_info()[1]))
        except ValueError:
            raise ValueError("Invalid decryption. {}".format(sys.exc_info()[1]))

        os.remove(temp_file_name)
        return decrypted_model
    assert isinstance(input_model, keras.models.Model), 'Input model not a valid Keras moodel.'
    try:
        input_model.load_weights(temp_file_name, by_name=by_name, custom_objects=custom_objects)
    except IOError:
        raise IOError("Invalid decryption. {}. The key used to load the model "
                      "is incorrect.".format(sys.exc_info()[1]))
    except ValueError:
        raise ValueError("Invalid decryption. {}. The key used to load the model "
                         "is incorrect.".format(sys.exc_info()[1]))

    os.remove(temp_file_name)
    return input_model


def model_io(model_path, enc_key=None, custom_objects=None):
    """Simple utility to handle model file based on file extensions.

    Args:
        pretrained_model_file (str): Path to the model file.
        enc_key (str): Key to load tlt file.
        custom_objects(dict): Custom objects for serialization and deserialization.

    Returns:
        model (keras.models.Model): Loaded keras model.
    """
    assert os.path.exists(
        model_path), "Model not found at {}".format(model_path)
    if model_path.endswith('.tlt'):
        assert enc_key is not None, "Key must be provided to load the model."
        return decode_to_keras(str(model_path),
                               enc_key,
                               custom_objects=custom_objects)
    if model_path.endswith('.hdf5'):
        return keras.models.load_model(str(model_path),
                                       compile=False,
                                       custom_objects=custom_objects)

    raise NotImplementedError(
        "Invalid model file extension. {}".format(model_path))


def get_step_from_ckzip(path):
    """Gets the step number from a ckzip checkpoint.

    Args:
        path (str): path to the checkpoint.
    Returns:
        int: the step number.
    """
    return int(os.path.basename(path).split('.')[1].split('-')[1])


def extract_checkpoint_file(tmp_zip_file):
    """Simple function to extract a checkpoint file.

    Args:
        tmp_zip_file (str): Path to the extracted zip file.

    Returns:
        tmp_checkpoint_path (str): Path to the extracted checkpoint.
    """
    # Set-up the temporary directory.
    temp_checkpoint_path = tempfile.mkdtemp()
    try:
        with ZipFile(tmp_zip_file, 'r') as zip_object:
            for member in zip_object.namelist():
                zip_object.extract(member, path=temp_checkpoint_path)
    except BadZipFile:
        raise ValueError(
            "The zipfile extracted was corrupt. Please check your key "
            "or delete the latest `*.ckzip` and re-run the command."
        )
    except Exception:
        raise IOError(
            "The last checkpoint file is not saved properly. "
            "Please delete it and rerun the script."
        )
    return temp_checkpoint_path


def get_tf_ckpt(ckzip_path, enc_key, latest_step):
    """Simple function to extract and get a trainable checkpoint.

    Args:
        ckzip_path (str): Path to the encrypted checkpoint.

    Returns:
        tf_ckpt_path (str): Path to the decrypted tf checkpoint
    """

    # Set-up the temporary directory.
    os_handle, temp_zip_path = tempfile.mkstemp()
    os.close(os_handle)

    # Decrypt the checkpoint file.
    try:
        # Try reading a checkpoint file directly.
        temp_checkpoint_path = extract_checkpoint_file(ckzip_path)
    except ValueError:
        # Decrypt and load checkpoints for TAO < 5.0
        with open(ckzip_path, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zip_file:
            encoding.decode(encoded_file, tmp_zip_file, bytes(enc_key, 'utf-8'))
        encoded_file.closed
        tmp_zip_file.closed
        # Load zip file and extract members to a tmp_directory.
        temp_checkpoint_path = extract_checkpoint_file(temp_zip_path)
        # Removing the temporary zip path.
        os.remove(temp_zip_path)

    return os.path.join(temp_checkpoint_path,
                        "model.ckpt-{}".format(latest_step))


def get_latest_checkpoint(results_dir, key):
    """Get the latest checkpoint path from a given results directory.

    Parses through the directory to look for the latest checkpoint file
    and returns the path to this file.

    Args:
        results_dir (str): Path to the results directory.
        key (str): key to decode/encode the model
    Returns:
        ckpt_path (str): Path to the latest checkpoint.
    """
    trainable_ckpts = [get_step_from_ckzip(item)
                       for item in os.listdir(results_dir) if item.endswith(".ckzip")]
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = os.path.join(results_dir, "model.epoch-{}.ckzip".format(latest_step))
    return get_tf_ckpt(latest_checkpoint, key, latest_step)


def get_latest_tlt_model(results_dir, extension=".hdf5"):
    """Get the latest checkpoint path from a given results directory.

    Parses through the directory to look for the latest tlt file
    and returns the path to this file.

    Args:
        results_dir (str): Path to the results directory.

    Returns:
        latest_checkpoint (str): Path to the latest checkpoint.
    """
    trainable_ckpts = []
    for item in os.listdir(results_dir):
        if item.endswith(extension):
            try:
                step_num = get_step_from_ckzip(item)
                trainable_ckpts.append(step_num)
            except IndexError:
                continue
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = os.path.join(results_dir, "model.epoch-{}{}".format(latest_step, extension))
    return latest_checkpoint


def load_model(model_path, key=None, custom_objects=None):
    """
    Load a model either in .h5 format, .tlt format or .hdf5 format.

    Args:
        custom_objects(dict): Custom objects for serialization and deserialization.

    Returns:
        model(keras.models.Model): Returns a keras model.
    """
    _, ext = os.path.splitext(model_path)

    if ext == '.hdf5':
        # directly load model, add dummy loss since loss is never required.
        model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        model.load_weights(model_path)

    elif ext == '.tlt':
        os_handle, temp_file_name = tempfile.mkstemp(suffix='.hdf5')
        os.close(os_handle)

        with open(temp_file_name, 'wb') as temp_file, open(model_path, 'rb') as encoded_file:
            encoding.decode(encoded_file, temp_file, str.encode(key))
            encoded_file.close()
            temp_file.close()

        # recursive call
        model = load_model(temp_file_name, None, custom_objects)
        os.remove(temp_file_name)

    else:
        raise NotImplementedError("{0} file is not supported!".format(ext))
    return model


def load_pretrained_weights(model, pretrained_model_path, key, logger=None):
    """Loads pretrained weights from another model into the specified model.

    Args:
        model (KerasModel): Model to load the pretrained weights into.
        pretrained_model_path (str): Path to the pretrained weights for the model.
        key (str): Key to decode/encode the model
        logger (obj): object for loggings
    """
    loaded_model = model_io(pretrained_model_path, enc_key=key)
    loaded_model_layers = [layer.name for layer in loaded_model.layers]
    if logger:
        logger.info("pretrained_model_path: {}".format(pretrained_model_path))
        logger.info("loaded_model_layers: {}".format(loaded_model_layers))
    for layer in model.layers:
        if layer.name in loaded_model_layers:
            pretrained_layer = loaded_model.get_layer(layer.name)
            weights_pretrained = pretrained_layer.get_weights()
            model_layer = model.get_layer(layer.name)
            try:
                model_layer.set_weights(weights_pretrained)
            except ValueError:
                continue
    del loaded_model
    # Trigger garbage collector to clear memory of the deleted loaded model
    gc.collect()


def save_exported_file(model, output_file_name, key, backend='onnx',
                       output_node_names=None, custom_objects=None,
                       target_opset=10, logger=None, delete_tmp_file=True):
    """Save the exported model file.

        This routine converts a keras model to onnx/uff model
        based on the backend the exporter was initialized with.

    Args:
        model (keras.model.Model): Decoded keras model to be exported.
        output_file_name (str): Path to the output file.
        key (str): key to decode/encode the model
        backend (str): backend engine
        output_node_names (str): name of the output node
        target_opset (int): target opset version
    """

    if backend == "onnx":
        in_tensor_names, out_tensor_names, in_tensor_shape = keras_to_onnx(
            model, output_file_name, custom_objects=custom_objects, target_opset=target_opset)

    elif backend == 'tfonnx':
        # Create froxen graph as .pb file.
        os_handle_tf, tmp_tf_file = tempfile.mkstemp(suffix=".pb")
        os.close(os_handle_tf)

        in_tensor_names, out_tensor_names, in_tensor_shape = keras_to_pb(
            model,
            tmp_tf_file,
            output_node_names,
            custom_objects=custom_objects)

        if output_node_names is None:
            output_node_names = out_tensor_names
        in_tensor_names, out_tensor_names = pb_to_onnx(
            tmp_tf_file,
            output_file_name,
            in_tensor_names,
            output_node_names,
            target_opset,
            verbose=False)

    elif backend == 'uff':
        os_handle, tmp_file_name = tempfile.mkstemp(suffix=".uff")
        os.close(os_handle)
        in_tensor_names, out_tensor_names, in_tensor_shape = keras_to_uff(
            model, output_file_name, None, custom_objects=custom_objects)
    else:
        raise NotImplementedError("Invalid backend provided. {}".format(backend))

    if logger:
        logger.info('Output Tensors: {}'.format(out_tensor_names))
        logger.info('Input Tensors: {} of shape: {}'.format(in_tensor_names, in_tensor_shape))

    return output_file_name, in_tensor_names, out_tensor_names


def change_model_batch_size(model, input_dims, logger=None, custom_objects=None):
    """Change batch size of a model.

    Args:
        model: input keras model
        input_dims (dict): model input name and shape.
        logger (obj): object for loggings
        custom_objects(dict): Custom objects for model conversion
    """
    # replace input shape of first layer
    layer_names_list = [layer.name for layer in model.layers]
    for layer_name in input_dims.keys():
        layer_idx = layer_names_list.index(layer_name)
        model._layers[layer_idx].batch_input_shape = input_dims[layer_name]

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)
    # new_model.summary() # Disable for TLT release

    # # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except layer is None:
            logger.warning("Could not transfer weights for layer {}".format(layer.name))
    return new_model


def pb_to_onnx(
    input_filename,
    output_filename,
    input_node_names,
    output_node_names,
    target_opset=None,
    verbose=False,
):
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
        in_tensor_name(s): The name(s) of the input nodes. If there is only one name, it will be
                           returned as a single string, otherwise a list of strings.
        out_tensor_name(s): The name(s) of the output nodes. If there is only one name, it will be
                            returned as a single string, otherwise a list of strings.
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
    # `tf2onnx.tfonnx.process_tf_graph` prints out layer names when
    # folding the layers. Disabling INFO logging for TLT branch.
    logging.getLogger("tf2onnx.tfonnx").setLevel(logging.WARNING)

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graphdef, name="")

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            tf_graph,
            input_names=input_node_names,
            output_names=output_node_names,
            continue_on_error=True,
            verbose=verbose,
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


def convertKeras2TFONNX(
    model,
    model_name,
    output_node_names=None,
    target_opset=10,
    custom_objects=None,
    logger=None
):
    """Convert keras model to onnx via frozen tensorflow graph.

    Args:
        model (keras.model.Model): Decoded keras model to be exported.
        model_name (str): name of the model file
        output_node_names (str): name of the output node
        target_opset (int): target opset version
    """
    # replace input shape of first layer
    output_pb_filename = model_name + '.pb'
    in_tensor_names, out_tensor_names, __ = keras_to_pb(
        model,
        output_pb_filename,
        output_node_names,
        custom_objects=custom_objects)

    if logger:
        logger.info('Output Tensors: {}'.format(out_tensor_names))
        logger.info('Input Tensors: {}'.format(in_tensor_names))

    output_onnx_filename = model_name + '.onnx'
    (_, _) = pb_to_onnx(output_pb_filename,
                        output_onnx_filename,
                        in_tensor_names,
                        out_tensor_names,
                        target_opset)
