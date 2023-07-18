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

"""A builder for DetectNet V2 Evaluator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import keras
import tensorflow as tf

from nvidia_tao_tf1.core.utils import set_random_seed
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import get_model_file_size
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_auto_weight_hook import (
    build_cost_auto_weight_hook
)
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_function_parameters import (
    build_target_class_list,
    get_target_class_names
)
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import build_dataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import select_dataset_proto
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.evaluation import Evaluator
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.evaluation_config import build_evaluation_config
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.tensorrt_evaluator import TensorRTEvaluator
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import build_model
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import select_model_proto
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing_config import (
    build_postprocessing_config
)
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer import BboxRasterizer
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.build_bbox_rasterizer_config import (
    build_bbox_rasterizer_config
)
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer

logger = logging.getLogger(__name__)

EVALUATOR_CLASS = {
    "tlt": Evaluator,
    "tensorrt": TensorRTEvaluator
}


def build_evaluator_for_trained_gridbox(experiment_spec,
                                        model_path,
                                        use_training_set,
                                        use_confidence_models,
                                        key=None,
                                        framework="tlt"):
    """Load a trained DetectNet V2 model and data for evaluation.

    Args:
        experiment_spec: experiment_pb2.Experiment object.
        model_path (str): Absolute path to a model file.
        model_type (str): Model type: 'keras' or 'tensorrt'.
        use_training_set (bool): If True, evaluate training set, else evaluate validation set.
        use_confidence_models (bool): If True, load confidence models.
        key (str): Key to load tlt model file.
        framework (str): Backend framework for the evaluator.
            Choices: ['tlt', 'tensorrt'].

    Returns:
        Evaluator instance.

    Raises:
        ValueError: if the model type is unsupported for evaluation.
    """
    # Set testing phase.
    keras.backend.set_learning_phase(0)

    # Load the trained model.
    Visualizer.build_from_config(experiment_spec.training_config.visualizer)

    target_class_names = get_target_class_names(
        experiment_spec.cost_function_config)
    target_classes = build_target_class_list(
        experiment_spec.cost_function_config)

    # Select the model config, which might have ModelConfig / TemporalModelConfig type.
    model_config = select_model_proto(experiment_spec)
    gridbox_model = build_model(m_config=model_config,
                                target_class_names=target_class_names,
                                framework=framework)
    config = gridbox_model.get_session_config()
    keras.backend.set_session(tf.Session(config=config))

    constructor_kwargs = {}
    if framework == "tlt":
        assert key is not None, (
            "The key to load the model must be provided when using the tlt framework "
            "to evaluate."
        )
        constructor_kwargs['enc_key'] = key
    logging.info("Loading model weights.")
    gridbox_model.load_model_weights(model_path, **constructor_kwargs)
    if framework == "tensorrt":
        gridbox_model.print_model_summary()

    # Set Maglev random seed.
    set_random_seed(experiment_spec.random_seed)

    # For now, use e.g. batch_size from the training parameters.
    if gridbox_model.max_batch_size:
        batch_size = gridbox_model.max_batch_size
    else:
        batch_size = experiment_spec.training_config.batch_size_per_gpu

    dataset_proto = select_dataset_proto(experiment_spec)
    target_class_mapping = dict(dataset_proto.target_class_mapping)
    # Build a dataloader.
    logging.info("Building dataloader.")
    dataloader = build_dataloader(
        dataset_proto=dataset_proto,
        augmentation_proto=experiment_spec.augmentation_config)

    # Note that repeat is set to true, or we will not be able to get records into a
    # fixed-shaped list.
    images, ground_truth_labels, num_samples = dataloader.get_dataset_tensors(
        batch_size, training=use_training_set, enable_augmentation=False, repeat=True)

    logger.info("Found %d samples in validation set", num_samples)

    # Note: this rounds up. If num_samples is not a multiple of batch_size, the last batch will
    # be duplicate work. However, this is masked from metrics and is correct.
    steps = (num_samples + batch_size - 1) // batch_size

    postprocessing_config = build_postprocessing_config(
        experiment_spec.postprocessing_config)
    evaluation_config = build_evaluation_config(
        experiment_spec.evaluation_config, target_class_names)

    confidence_models = None

    evaluator = EVALUATOR_CLASS[framework](
        postprocessing_config, evaluation_config, gridbox_model, images,
        ground_truth_labels, steps, confidence_models
    )

    # Setup the cost function.
    cost_auto_weight_hook =\
        build_cost_auto_weight_hook(
            experiment_spec.cost_function_config, steps)

    # Get a BboxRasterizer.
    bbox_rasterizer_config = \
        build_bbox_rasterizer_config(experiment_spec.bbox_rasterizer_config)
    bbox_rasterizer = BboxRasterizer(input_width=gridbox_model.input_width,
                                     input_height=gridbox_model.input_height,
                                     output_width=gridbox_model.output_width,
                                     output_height=gridbox_model.output_height,
                                     target_class_names=target_class_names,
                                     bbox_rasterizer_config=bbox_rasterizer_config,
                                     target_class_mapping=target_class_mapping)

    # Build ops for doing validation.
    # NOTE: because of potential specifity in how labels are fed to the DetectNet V2 (or child
    # class) object, we use the ones the Evaluator object's wrapper around it.
    ground_truth_tensors = gridbox_model.generate_ground_truth_tensors(
        bbox_rasterizer=bbox_rasterizer,
        batch_labels=evaluator.ground_truth_labels)

    gridbox_model.build_validation_graph(images, ground_truth_tensors, target_classes,
                                         cost_auto_weight_hook.cost_combiner_func)

    gridbox_model.print_model_summary()
    model_metadata = {
        "size": get_model_file_size(model_path),
        "param_count": gridbox_model.num_params
    }
    status_logging.get_status_logger().write(
        data=model_metadata,
        message="Model constructed."
    )

    return evaluator
