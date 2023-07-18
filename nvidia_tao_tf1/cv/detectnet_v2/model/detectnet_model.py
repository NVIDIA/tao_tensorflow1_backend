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

"""GridboxModel class that takes care of constructing, training and validating a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import logging
from math import ceil

import keras
from keras.layers import Input
from keras.models import Model

import tensorflow as tf

import nvidia_tao_tf1.core
from nvidia_tao_tf1.core.export._quantized import check_for_quantized_layers
from nvidia_tao_tf1.core.models.quantize_keras_model import create_quantized_keras_model
from nvidia_tao_tf1.core.templates.darknet import DarkNet
from nvidia_tao_tf1.core.templates.efficientnet import EfficientNetB0
from nvidia_tao_tf1.core.templates.googlenet import GoogLeNet
from nvidia_tao_tf1.core.templates.mobilenet import MobileNet, MobileNetV2
from nvidia_tao_tf1.core.templates.resnet import ResNet
from nvidia_tao_tf1.core.templates.squeezenet import SqueezeNet
from nvidia_tao_tf1.core.templates.vgg import VggNet
from nvidia_tao_tf1.cv.common.utils import (
    encode_from_keras,
    get_num_params
)
from nvidia_tao_tf1.cv.detectnet_v2.model.utilities import get_class_predictions
from nvidia_tao_tf1.cv.detectnet_v2.model.utilities import inference_learning_phase
from nvidia_tao_tf1.cv.detectnet_v2.model.utilities import model_io
from nvidia_tao_tf1.cv.detectnet_v2.objectives.objective_set import build_objective_set
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer

logger = logging.getLogger(__name__)

# Setting up supported feature extractor templates.
SUPPORTED_TEMPLATES = ["resnet", "darknet", "mobilenet_v1", "mobilenet_v2",
                       "squeezenet", "googlenet", "vgg", "helnet",
                       "efficientnet_b0"]


class GridboxModel(object):
    """GridboxModel class.

    GridboxModel contains functionality for constructing and manipulating a Keras based models
    with gridbox head, building training and validation graphs for the model, and visualizing
    predictions.
    """

    def __init__(self, num_layers, use_pooling, use_batch_norm, dropout_rate,
                 objective_set_config, activation_config, target_class_names,
                 freeze_pretrained_layers, allow_loaded_model_modification,
                 template='resnet', all_projections=True, freeze_blocks=None,
                 freeze_bn=None, enable_qat=False):
        """Init function.

        Args:
            num_layers (int): Number of layers for scalable feature extractors.
            use_pooling (bool): Whether to add pooling layers to the feature extractor.
            use_batch_norm (bool): Whether to add batch norm layers.
            dropout_rate (float): Fraction of the input units to drop. 0.0 means dropout is
                not used.
            objective_set_config (ObjectiveSet proto): The ObjectiveSet configuration proto.
            target_class_names (list): A list of target class names.
            freeze_pretrained_layers (bool): Prevent updates to pretrained layers' parameters.
            allow_loaded_model_modification (bool): Allow loaded model modification.
            template (str): Model template to use for feature extractor.
        """
        self.num_layers = num_layers
        self.use_pooling = use_pooling
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.template = template
        self.enable_qat = enable_qat
        self.max_batch_size = None

        # Note: the order of target_class_names determines the order of classes in network output.
        self.target_class_names = target_class_names
        self.objective_set_config = objective_set_config
        self.activation_config = activation_config
        self.freeze_pretrained_layers = freeze_pretrained_layers
        self.freeze_blocks = freeze_blocks
        self.freeze_bn = freeze_bn
        self.allow_loaded_model_modification = allow_loaded_model_modification
        self.constructed = False
        self.all_projections = all_projections

    def construct_model(self, input_shape, kernel_regularizer=None, bias_regularizer=None,
                        pretrained_weights_file=None, enc_key=None):
        """Create a Keras model with gridbox head.

        Args:
            input_shape (tuple / list / TensorShape):
                model input shape without batch dimension (C, H, W).
            kernel_regularizer (keras.regularizers.Regularizer instance):
                Regularizer to be applied to convolution kernels.
            bias_regularizer (keras.regularizers.Regularizer instance):
                Regularizer to be applied to biases.
            pretrained_weights_file (str): An optional model weights file to be loaded.
        Raises:
            NotImplementedError: If pretrained_weights_file is not None.
        """
        data_format = 'channels_first'
        model = self._construct_feature_extractor(input_shape=input_shape,
                                                  data_format=data_format,
                                                  kernel_regularizer=kernel_regularizer,
                                                  bias_regularizer=bias_regularizer)
        # If you have weights you've trained previously, you can load them into this model.
        if pretrained_weights_file is not None:
            if pretrained_weights_file.endswith(".h5"):
                model.load_weights(str(pretrained_weights_file), by_name=True)
            else:
                loaded_model = model_io(pretrained_weights_file, enc_key=enc_key)
                loaded_model_layers = [layer.name for layer in loaded_model.layers]
                logger.info("Loading weights from pretrained "
                            "model file. {}".format(pretrained_weights_file))
                for layer in model.layers:
                    if layer.name in loaded_model_layers:
                        pretrained_layer = loaded_model.get_layer(layer.name)
                        weights_pretrained = pretrained_layer.get_weights()
                        model_layer = model.get_layer(layer.name)
                        try:
                            model_layer.set_weights(weights_pretrained)
                            logger.info(
                                "Layer {} weights set from pre-trained model.".format(
                                    model_layer.name
                                )
                            )
                        except ValueError:
                            logger.info("Layer {} weights skipped from pre-trained model.".format(
                                    model_layer.name
                                )
                            )
                            continue
                del loaded_model
                gc.collect()

        model = self._construct_objectives_head(model=model,
                                                data_format=data_format,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer)

        valid_qat_template = self.template not in [
            "mobilenet_v1", "mobilenet_v2"]
        no_quantized_layers_in_model = not(check_for_quantized_layers(model))
        if self.enable_qat and valid_qat_template:
            assert no_quantized_layers_in_model, (
                "Model already has quantized layers. Please consider using a non QAT model "
                "or set the enable_qat flag in the training config to false."
            )
            logger.info("Converting the keras model to quantize keras model.")
            model = create_quantized_keras_model(model)

        self.keras_model = model
        self.constructed = True

    def _construct_feature_extractor(self, input_shape, data_format, kernel_regularizer=None,
                                     bias_regularizer=None):
        """Generate a keras stride 16 feature extractor model.

        Args:
            input_shape: model input shape (N,C,H,W). N is ignored.
            data_format: Order of the dimensions (C, H, W).
            kernel_regularizer: Keras regularizer to be applied to convolution kernels.
            bias_regularizer: Keras regularizer to be applied to biases.
            pretrained_weights_file: An optional model weights file to be loaded.
        Raises:
            AssertionError: If the model is already constructed.
        Returns:
            model (keras.model): The model for feature extraction.
        """
        assert not self.constructed, "Model already constructed."
        # Define entry points to the model.
        assert len(input_shape) == 3
        self.input_num_channels = int(input_shape[0])
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        inputs = Input(shape=(self.input_num_channels, self.input_height, self.input_width))

        # Set up positional arguments and key word arguments to instantiate feature extractor
        # templates.
        args = [self.num_layers, inputs]
        kwargs = {'use_batch_norm': self.use_batch_norm,
                  'kernel_regularizer': kernel_regularizer,
                  'bias_regularizer': bias_regularizer,
                  'freeze_blocks': self.freeze_blocks,
                  'freeze_bn': self.freeze_bn}

        # Decide feature extractor architecture.
        if self.template == "resnet":
            model_class = ResNet
            kwargs['all_projections'] = self.all_projections
            kwargs['use_pooling'] = self.use_pooling
        elif self.template == "vgg":
            model_class = VggNet
            kwargs['use_pooling'] = self.use_pooling
        elif self.template == "googlenet":
            model_class = GoogLeNet
            # Remove nlayers as positional arguments to the googlenet template.
            args.pop(0)
        elif self.template == "mobilenet_v1":
            model_class = MobileNet
            kwargs['alpha'] = 1.0
            kwargs['depth_multiplier'] = 1
            kwargs['dropout'] = self.dropout_rate
            kwargs['stride'] = 16
            kwargs['add_head'] = False
            # Remove nlayers as positional arguments to the googlenet template.
            args.pop(0)
        elif self.template == "mobilenet_v2":
            model_class = MobileNetV2
            kwargs['alpha'] = 1.0
            kwargs['depth_multiplier'] = 1
            kwargs['stride'] = 16
            kwargs['add_head'] = False
            # Remove nlayers as positional arguments to the googlenet template.
            args.pop(0)
        elif self.template == "darknet":
            model_class = DarkNet
            args.pop(1)
            kwargs["input_tensor"] = inputs
            kwargs['alpha'] = 0.1
        elif self.template == "efficientnet_b0":
            model_class = EfficientNetB0
            kwargs["add_head"] = False
            kwargs["input_tensor"] = inputs
            kwargs["stride16"] = True
            # No positional args are required to generate the
            # efficientnet template
            while args:
                args.pop()
        elif self.template == "squeezenet":
            model_class = SqueezeNet
            kwargs.pop("freeze_bn", None)
            kwargs.pop("use_batch_norm", None)
            args.pop(0)
        else:
            error_string = "Unsupported model template: {}.\nPlease choose one" \
                "from the following: {}".format(self.template, SUPPORTED_TEMPLATES)
            raise NotImplementedError(error_string)

        model = model_class(*args, **kwargs)
        # Feature extractor output shape.
        self.output_height = model.output_shape[2]
        self.output_width = model.output_shape[3]

        return model

    def _construct_objectives_head(self, model, data_format, kernel_regularizer, bias_regularizer):
        """Construct the detector head on top of a feature extractor.

        Args:
            data_format (str): Order of the dimensions. Set to 'channels_first'.
            model (keras.model): Keras model that performs the feature extraction.
            kernel_regularizer (keras.regularizers.Regularizer instance):
                Regularizer to be applied to convolution kernels.
            bias_regularizer (keras.regularizers.Regularizer instance):
                Regularizer to be applied to biases.
        Returns:
            model (keras.model): An end to end keras model, where the gridbox head is
                attached to the feature extractor.
        """
        # Build the set of objectives (cov, bbox, ...).
        self.objective_set = build_objective_set(self.objective_set_config,
                                                 self.output_height,
                                                 self.output_width,
                                                 self.input_height,
                                                 self.input_width)

        # Construct DNN heads and get their output tensors for predicting the objectives.
        num_classes = len(self.target_class_names)
        outputs = self.objective_set.construct_outputs(model, num_classes, data_format,
                                                       kernel_regularizer, bias_regularizer)

        # Construct the complete model.
        return Model(inputs=model.inputs, outputs=outputs, name='%s_detectnet_v2' % (model.name))

    def predictions_to_dict(self, predictions):
        """Helper for converting Model predictions into a dictionary for easy parsing.

        Slices per class predictions to their own dimension.

        Args:
            predictions: Model predictions list.
        Returns:
            Dictionary of model predictions indexed objective name.
        """
        pred_dict = {}
        for objective in self.objective_set.learnable_objectives:
            matching_preds = [pred for pred in predictions if 'output_'+objective.name in pred.name]
            assert len(matching_preds) < 2, "Ambigous model predictions %s for objective %s" % \
                (matching_preds, objective.name)
            assert matching_preds, "Model predictions not found for objective %s" % \
                objective.name

            # Reshape such that class has its own dimension.
            pred = objective.reshape_output(matching_preds[0],
                                            num_classes=len(self.target_class_names))

            pred_dict[objective.name] = pred

        return pred_dict

    def save_model(self, file_name, enc_key=None):
        """Save the model to disk.

        Args:
            file_name (str): Model file name.
            enc_key (str): Key string for encryption.
        Raises:
            ValueError if postprocessing_config is None but save_metadata is True.
        """
        self.keras_model.save(file_name, overwrite=True, include_optimizer=False)

    def load_model_weights(self, model_file,
                           custom_objects=None,
                           enc_key=None,
                           input_num_channels=None,
                           input_height=None,
                           input_width=None):
        """Load a previously saved TLT model.

        Args:
            model_file (str): Model file name.
            custom_objects (dict): Dictionary for the custom Keras layers in the model.
            enc_key (str): Key for decryption.
            input_num_channels (int): Number of channels in the input to the model.
            input_height (int): Height of the input to the model.
            input_width (int): Width of the input to the model.
        """
        input_overrides = {input_num_channels, input_height, input_width}
        if input_overrides != {None}:
            if None in input_overrides:
                raise ValueError('HelnetGridbox.load_model_weights expects either no input / '
                                 'output shape overrides, or all of them to be overridden.')

        if model_file.endswith('.h5'):
            raise NotImplementedError("Cannot load just weights for a pruned model.")
        else:
            model = model_io(model_file, enc_key=enc_key)

        assert model, "Couldn't load model."
        if self.enable_qat:
            # Convert loaded gridbox model to a QAT enabled model with
            # QuantizedConv2D and QDQ nodes.
            assert not(check_for_quantized_layers(model)), (
                "The model provided already seems to have quantized layers. Please consider "
                "using a non QAT trained model as pretrained_model_file or set the enable_qat "
                "flag in training_config to false."
            )
            model = create_quantized_keras_model(model)
        self.keras_model = model

        # Set input and output size variables.
        default_output_shape = self.keras_model.get_layer(
            "output_cov").output_shape
        model_stride = max(self.keras_model.input_shape[2] // default_output_shape[-2],
                           self.keras_model.input_shape[3] // default_output_shape[-1])
        if input_overrides == {None}:
            # Retrieve them from saved model, and assume the user wants to infer using the same
            # shapes as those.
            self.input_num_channels = self.keras_model.input_shape[1]
            self.input_height = self.keras_model.input_shape[2]
            self.input_width = self.keras_model.input_shape[3]
            # The last two dimensions are height and width.
            self.output_height = default_output_shape[-2]
            self.output_width = default_output_shape[-1]
        else:
            self.input_num_channels = input_num_channels
            self.input_height = input_height
            self.input_width = input_width
            self.output_height = int(ceil(input_height / model_stride))
            self.output_width = int(ceil(input_width / model_stride))

        self.objective_set = build_objective_set(self.objective_set_config,
                                                 self.output_height,
                                                 self.output_width,
                                                 self.input_height,
                                                 self.input_width)

        self.constructed = True

    def update_regularizers(self, kernel_regularizer=None,
                            bias_regularizer=None):
        """Update regularizers for models that are being loaded."""
        model_config = self.keras_model.get_config()
        for layer, layer_config in zip(self.keras_model.layers, model_config['layers']):
            # Updating regularizer parameters for conv2d, depthwise_conv2d and dense layers.
            if type(layer) in [keras.layers.convolutional.Conv2D,
                               keras.layers.core.Dense,
                               keras.layers.DepthwiseConv2D]:
                if hasattr(layer, 'kernel_regularizer'):
                    layer_config['config']['kernel_regularizer'] = kernel_regularizer
                if hasattr(layer, 'bias_regularizer'):
                    layer_config['config']['bias_regularizer'] = bias_regularizer
        prev_model = self.keras_model
        self.keras_model = keras.models.Model.from_config(model_config)
        self.keras_model.set_weights(prev_model.get_weights())

    @classmethod
    def load_model(cls, model_file, objective_set_config, target_class_names):
        """Create a new GridboxModel instance with model metadata.

        Args:
            model_file: Model file name.
            objective_set_config: Loaded objective set config from model metadata.
            target_class_names (list): Loaded target class names from model metadata.
        Returns:
            GridboxModel object.
        """
        gridbox_model = cls(num_layers=None,
                            template=None,
                            use_pooling=None,
                            use_batch_norm=None,
                            dropout_rate=None,
                            objective_set_config=None,
                            activation_config=None,
                            target_class_names=None,
                            freeze_pretrained_layers=None,
                            allow_loaded_model_modification=None)
        if objective_set_config is not None:
            assert gridbox_model.objective_set_config is None, \
                "Loaded config would override spec."
            gridbox_model.objective_set_config = objective_set_config
        if target_class_names is not None:
            assert gridbox_model.target_class_names is None, \
                "Loaded config would override spec."
            gridbox_model.target_class_names = target_class_names

        gridbox_model.load_model_weights(model_file)

        return gridbox_model

    @property
    def objective_names(self):
        """Return the objective names this model is outputting.

        Returns:
            objective_names (set): Set of objective names, each of them a str.

        Raises:
            RuntimeError: If the model has not been constructed yet (in which case it does not make
                sense to ask for objectives).
        """
        if not self.constructed:
            raise RuntimeError("Objective names cannot be determined before the model has been"
                               "constructed.")
        return set(obj.name for obj in self.objective_set.learnable_objectives)

    @property
    def output_layer_names(self):
        """Return the model output layer names.

        Returns:
            output_layer_names (list): List of output layer names, each of them a str.

        Raises:
            RuntimeError: If the model has not been constructed yet (in which case it does not make
                sense to ask for outputs).
        """
        if not self.constructed:
            raise RuntimeError("Output layer names cannot be determined before the model has been"
                               "constructed.")
        return ['output_' + o.name for o in self.objective_set.learnable_objectives]

    def add_missing_outputs(self, kernel_regularizer=None, bias_regularizer=None):
        """Add missing outputs to a loaded model.

        Args:
            kernel_regularizer: Keras regularizer to be applied to convolution kernels.
            bias_regularizer: Keras regularizer to be applied to biases.

        Raises:
            AssertionError: if the model modification is not allowed and the model does
                not contain heads for all learnable objectives.
        """
        # If model modification is not allowed, return.
        if not self.allow_loaded_model_modification:
            # Model should be good to go as is. Assert that the model heads are in place.
            for objective in self.objective_set.learnable_objectives:
                assert any([objective.name in o for o in self.keras_model.output_names]), \
                    "Objective head is missing from model, and model modification is not allowed."
            return

        # Construct outputs. In case the loaded model does not contain outputs for all
        # objectives, we need to construct the associated model heads.
        outputs = self.objective_set.construct_outputs(model=self.keras_model,
                                                       num_classes=len(self.target_class_names),
                                                       data_format='channels_first',
                                                       kernel_regularizer=kernel_regularizer,
                                                       bias_regularizer=bias_regularizer)

        self.keras_model = Model(inputs=self.keras_model.inputs,
                                 outputs=outputs,
                                 name=self.keras_model.name)

    def _print_model_summary_recurse(self, model):
        """Print model summary recursively.

        Helper function for printing nested models (ie. models that have models as layers).

        Args:
            model: Keras model to print.
        """
        model.summary()
        for l in model.layers:
            if isinstance(l, keras.engine.training.Model):
                print('where %s is' % l.name)
                self._print_model_summary_recurse(l)

    def print_model_summary(self):
        """Print model summary."""
        self._print_model_summary_recurse(self.keras_model)

    def get_model_name(self):
        """Return model name."""
        return self.keras_model.name

    def _cost_func(self, target_classes, cost_combiner_func, ground_truth_tensors_dict,
                   pred_tensors_dict, loss_masks=None):
        """Model cost function.

        Args:
            target_classes (list): A list of TargetClass instances.
            cost_combiner_func: A function that takes in a dictionary of objective costs,
                and total cost by computing a weighted sum of the objective costs.
            ground_truth_tensors_dict (dict): Maps from [target_class_name][objective_name] to
                rasterized ground truth tensors.
            pred_tensors_dict (dict): Maps fro [target_class_name][objective_name] to dnn
                prediction tensors.
            loss_masks (nested dict): [target_class_name][objective_name]. The leaf values are the
                corresponding loss masks (tf.Tensor) for a batch of frames.

        Returns:
            total_cost: Scalar cost.
        """
        # Compute per target class per objective costs.
        component_costs = self.objective_set.compute_component_costs(ground_truth_tensors_dict,
                                                                     pred_tensors_dict,
                                                                     target_classes, loss_masks)

        # Use external cost_combiner_func to compute total cost.
        return cost_combiner_func(component_costs)

    def build_training_graph(self, inputs, ground_truth_tensors, optimizer, target_classes,
                             cost_combiner_func, train_op_generator, loss_masks=None):
        """Build a training graph.

        Args:
            inputs: Dataset input tensors to be used for training.
            ground_truth_tensors (dict): [target_class_name][objective_name] -> tf.Tensor in model
                output space.
            optimizer: Optimizer to be used for updating model weights.
            target_classes (list): A list of TargetClass instances.
            cost_combiner_func: A function that takes in a dictionary of objective costs,
                and returns the total cost by computing a weighted sum of the objective costs.
            train_op_generator: Object that creates TF op for one training step.
            loss_masks (nested dict): [target_class_name][objective_name]. The leaf values are the
                corresponding loss masks (tf.Tensor) for a batch of frames.

        Raises:
            AssertionError: If the model hasn't been constructed yet.
        """
        assert self.constructed, "Construct the model before build_training_graph."

        model_name = self.get_model_name()
        inputs = Input(tensor=inputs, name="input_images")
        predictions = self.keras_model(inputs)

        # Build a training model that connects input tensors to model.
        self.keras_training_model = keras.models.Model(inputs=inputs,
                                                       outputs=predictions,
                                                       name=model_name)

        # Convert the network predictions to float32 for accuracte cost computation
        predictions = self._convert_to_fp32(predictions)

        output_dict = self.predictions_to_dict(predictions)
        pred_dict = get_class_predictions(output_dict, self.target_class_names)
        # Compute task cost.
        task_cost = self._cost_func(target_classes, cost_combiner_func, ground_truth_tensors,
                                    pred_dict, loss_masks)
        tf.summary.scalar('task_cost', task_cost)
        # Compute regularization cost.
        regularization_cost = tf.reduce_sum(self.keras_training_model.losses)
        tf.summary.scalar('regularization_cost', regularization_cost)
        # Compute total cost.
        self.total_cost = task_cost + regularization_cost
        tf.summary.scalar('total_cost', self.total_cost)

        # Create training op and apply dynamic cost scaling if enabled in spec.
        self.train_op = train_op_generator.get_train_op(optimizer=optimizer,
                                                        total_cost=self.total_cost,
                                                        var_list=self.keras_model.trainable_weights)

        if Visualizer.enabled:
            # Set histogram plot collection.
            histogram_collections = [nvidia_tao_tf1.core.hooks.utils.INFREQUENT_SUMMARY_KEY]
            # Add weight histogram to tf summary.
            Visualizer.keras_model_weight_histogram(
                self.keras_training_model,
                collections=histogram_collections
            )

    @staticmethod
    def _convert_to_fp32(tensor_list):
        """Convert a list of TF tensors to float32 TF tensors.

        Args:
            tensor_list: A list of TF tensors of any numeric data type.
        Returns:
            A list of float32 TF tensors.
        """
        # Cast operation must maintain the name of the input tensor
        return [tf.cast(tensor, dtype=tf.float32, name=tensor.name.split(':')[0] + '/cast_to_32')
                for tensor in tensor_list]

    def get_total_cost(self):
        """Return total cost."""
        return self.total_cost

    def get_train_op(self):
        """Return train op."""
        return self.train_op

    def get_keras_training_model(self):
        """Return Keras training model."""
        return self.keras_training_model

    def get_ground_truth_labels(self, ground_truth_labels):
        """Get ground truth labels.

        For the base GridboxModel class, this is a pass-through.

        Args:
            ground_truth_labels (list): Each element is a dict of target features.

        Returns:
            ground_truth_labels (list): Unchanged.
        """
        return ground_truth_labels

    def generate_ground_truth_tensors(self, bbox_rasterizer, batch_labels):
        """Generate ground truth tensors.

        Args:
            bbox_rasterizer (BboxRasterizer): Instance of the BboxRasterizer class that will handle
                label-to-rasterizer-arg translation and provide the target_gradient() methods with
                the necessary inputs, as well as perform the final call to the SDK's rasterizer.
            batch_labels (list): Each element is a dict of target features (each a tf.Tensor).

        Returns:
            target_tensors (dict): [target_class_name][objective_name] rasterizer ground truth
                tensor.
        """
        target_tensors = \
            self.objective_set.generate_ground_truth_tensors(bbox_rasterizer, batch_labels)

        return target_tensors

    @inference_learning_phase
    def build_inference_graph(self, inputs):
        """Set up the model for pure inference.

        Args:
            inputs: Input tensors of shape (N, 3, H, W). Can come from keras.layers.Input or be
                some tf.Tensor / placeholder.

        Returns:
            raw_predictions: pure output from the keras model. This is a list of output tensors for
                each objective.
            class_predictions: (dict) [target_class_name][output_name] = tensor of shape
                [N, obj_depth, output_height, output_width]
        """
        assert self.constructed, "Construct the model before build_inference_graph."

        raw_predictions = self.keras_model(inputs)

        # Convert the network predictions to float32 for accuracte cost computation
        raw_predictions = self._convert_to_fp32(raw_predictions)

        output_dict = self.predictions_to_dict(raw_predictions)
        raw_predictions = get_class_predictions(output_dict, self.target_class_names)

        input_space_predictions = {}
        # Convert predictions to input image space (e.g. to absolute bbox coordinates)
        absolute_predictions = self.objective_set.predictions_to_absolute(output_dict)
        input_space_predictions = self.objective_set.transform_predictions(absolute_predictions)

        # Get the predictions per class.
        input_space_predictions = self.get_class_predictions(input_space_predictions)

        # Return inference outputs
        return raw_predictions, input_space_predictions

    def get_class_predictions(self, predictions):
        """Converting predictions dictionary to be indexed by class names.

        Args:
            predictions (dict): Dictionary of model predictions indexed by objective name.
        Returns:
            pred_dict: Dictionary of model predictions indexed by
                target class name and objective name.
        """
        pred_dict = get_class_predictions(predictions, self.target_class_names)
        return pred_dict

    @inference_learning_phase
    def build_validation_graph(self, inputs, ground_truth_tensors, target_classes,
                               cost_combiner_func, loss_masks=None):
        """Set up the model for validation.

        Args:
            inputs: Dataset input tensors to be used for validation.
            ground_truth_tensors (dict): [target_class_name][objective_name] -> tf.Tensor.
            target_classes (list): A list of TargetClass instances.
            cost_combiner_func: A function that takes in a dictionary of objective costs,
                and returns the total cost by computing a weighted sum of the objective costs.
            loss_masks (nested dict): [target_class_name][objective_name]. The leaf values are the
                corresponding loss masks (tf.Tensor) for a batch of frames.

        Raises:
            AssertionError: If the model hasn't been constructed yet.
        """
        assert self.constructed, "Construct the model before build_validation_graph."

        class_predictions, self.validation_predictions = \
            self.build_inference_graph(inputs)

        # Compute validation cost using model raw predictions.
        # Disable visualization during validation cost computation to avoid Tensorboard clutter.
        with Visualizer.disable():
            self.validation_cost = self._cost_func(target_classes, cost_combiner_func,
                                                   ground_truth_tensors,
                                                   class_predictions, loss_masks)

    def get_validation_tensors(self):
        """Get a list of tensors for validating/evaluating the model."""
        return [self.validation_predictions, self.validation_cost]

    def get_model_weights(self):
        """Return model weights as numpy arrays."""
        return self.keras_model.get_weights()

    @property
    def num_params(self):
        """Get the number of parameters in the keras model."""
        if not self.constructed:
            raise RuntimeError(
                "Model parameter count cannot be derived unless the"
                "GridBox Model class sets self.constructed to True"
            )
        return get_num_params(self.keras_model)

    def set_model_weights(self, weights):
        """Set model weights from numpy arrays.

        Args:
            weights: Model weights as numpy arrays.
        """
        return self.keras_model.set_weights(weights)

    def get_target_class_names(self):
        """Return a list of model target classes."""
        return self.target_class_names

    @staticmethod
    def get_session_config():
        """Retrieve a TensorFlow session config.

        Returns:
            config (tf.compat.v1.ConfigProto): Retrive tensorflow config
                with GPU options set.
        """
        gpu_options = tf.compat.v1.GPUOptions(
            allow_growth=True
        )
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        return config

    def visualize_predictions(self):
        """Visualize bboxes predicted by the model."""
        if Visualizer.enabled is False or Visualizer.num_images <= 0:
            return

        # Compute the number of images to visualize as the minimum of the user
        # parameter and the actual minibatch size.
        batch_size = self.keras_training_model.inputs[0].shape[0]
        batch_size = min(Visualizer.num_images, batch_size)

        # We're visualizing only a part of the minibatch.
        inputs = self.keras_training_model.inputs[0][0:batch_size, :3]
        raw_outputs = self.keras_training_model.outputs

        # For visualization, float32 input and output is required
        inputs = tf.cast(inputs, dtype=tf.float32)
        raw_outputs = self._convert_to_fp32(raw_outputs)

        predictions = self.predictions_to_dict(raw_outputs)
        predictions = {output_name: tensor[0:batch_size] for output_name, tensor in
                       predictions.items()}

        abs_predictions = self.objective_set.predictions_to_absolute(predictions)

        Visualizer.visualize_elliptical_bboxes(self.target_class_names, inputs,
                                               abs_predictions['cov'],
                                               abs_predictions['bbox'])
        Visualizer.visualize_rectangular_bboxes(self.target_class_names, inputs,
                                                abs_predictions['cov'],
                                                abs_predictions['bbox'])
