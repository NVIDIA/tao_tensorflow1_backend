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

"""
Perform Evaluation for the Unet Segmentation.

This code does the evaluation. Given the paths of th evalidation images and the
masks , it prints the miou, f1 score, recall, avg score metrics.

Short code breakdown:
(1) Creates the Runtime_config and creates the estimator
(2) Hook up the data pipe and estimator to unet model with backbones such as
Resnet, vanilla Unet
(3) Retrieves/ Encrypts the trained checkpoint.
(4) Performs Evaluation and prints the semantic segmentation metric and dumps
it a json file.
"""

import argparse
import collections
import json
import logging
import math
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import get_model_file_size
from nvidia_tao_tf1.cv.unet.dllogger.logger import JSONStreamBackend, Logger, StdOutBackend, \
    Verbosity
from nvidia_tao_tf1.cv.unet.hooks.profiling_hook import ProfilingHook
from nvidia_tao_tf1.cv.unet.model.build_unet_model import build_model
from nvidia_tao_tf1.cv.unet.model.build_unet_model import select_model_proto
from nvidia_tao_tf1.cv.unet.model.model_io import _extract_ckpt
from nvidia_tao_tf1.cv.unet.model.utilities import build_target_class_list, get_train_class_mapping
from nvidia_tao_tf1.cv.unet.model.utilities import get_custom_objs, get_pretrained_ckpt, \
    update_model_params
from nvidia_tao_tf1.cv.unet.model.utilities import initialize, initialize_params, save_tmp_json
from nvidia_tao_tf1.cv.unet.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.unet.utils.data_loader import Dataset
from nvidia_tao_tf1.cv.unet.utils.evaluate_trt import Evaluator
from nvidia_tao_tf1.cv.unet.utils.model_fn import unet_fn

logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.INFO)


def getScoreAverage(scoreList):
    """Compute the average score of all classes."""

    validScores = 0
    scoreSum = 0.0
    for score in scoreList:
        if not math.isnan(score):
            validScores += 1
            scoreSum += score
    if validScores == 0:
        return float('nan')
    avg_score = scoreSum / validScores
    return avg_score


def compute_metrics_masks(predictions_it, dataset, target_classes, params):
    """Compute metrics for semantic segmentation.

    Args:
        predictions(list): List of prediction numpy arrays.
        img_names(list): The img_name of the test images
        dataset(class object):  Dataset object from the dataloader utility.

    """
    num_classes = params.num_conf_mat_classes
    conf_mat = np.zeros([num_classes, num_classes], dtype=np.float32)

    for p in tqdm(predictions_it):
        pred = p["conf_matrix"]
        conf_mat += np.matrix(pred)

    metrices = {}
    perclass_tp = np.diagonal(conf_mat).astype(np.float32)
    perclass_fp = conf_mat.sum(axis=0) - perclass_tp
    perclass_fn = conf_mat.sum(axis=1) - perclass_tp
    iou_per_class = perclass_tp/(perclass_fp+perclass_tp+perclass_fn)
    precision_per_class = perclass_tp/(perclass_fp+perclass_tp)
    recall_per_class = perclass_tp/(perclass_tp+perclass_fn)
    train_id_name_mapping = get_train_class_mapping(target_classes)
    f1_per_class = []
    final_results_dic = {}
    for num_class in range(num_classes):
        name_class = "/".join(train_id_name_mapping[num_class])
        per_class_metric = {}
        prec = precision_per_class[num_class]
        rec = recall_per_class[num_class]
        iou = iou_per_class[num_class]
        f1 = (2 * prec * rec)/float((prec + rec))
        f1_per_class.append(f1)
        per_class_metric["precision"] = prec
        per_class_metric["Recall"] = rec
        per_class_metric["F1 Score"] = f1
        per_class_metric["iou"] = iou

        final_results_dic[name_class] = per_class_metric

    mean_iou_index = getScoreAverage(iou_per_class)
    mean_rec = getScoreAverage(recall_per_class)
    mean_precision = getScoreAverage(precision_per_class)
    mean_f1_score = getScoreAverage(f1_per_class)

    metrices["rec"] = mean_rec
    metrices["prec"] = mean_precision
    metrices["fmes"] = mean_f1_score
    metrices["mean_iou_index"] = mean_iou_index
    metrices["results_dic"] = final_results_dic

    return metrices


def print_compute_metrics(dataset, predictions_it, output_dir, target_classes, params,
                          mode="tlt"):
    """Run the prediction followed by evaluation using the estimator.

    Args:
        estimator: estimator object wrapped with run config parameters.
        dataset: Dataset object fro the dataloader utility.
        params: Parameters to feed to Estimator.
        unet_model: Keras Unet Model.
        profile_logger: Logging the Evaluation updates.
        key: The key to encrypt the model.
        output_dir: The directory where the results file is saved.

    """
    metrices = compute_metrics_masks(predictions_it, dataset, target_classes, params)
    recall_str = "Recall : " + str(metrices["rec"])
    precision_str = "Precision: " + str(metrices["prec"])
    f1_score_str = "F1 score: " + str(metrices["fmes"])
    mean_iou_str = "Mean IOU: " + str(metrices["mean_iou_index"])
    results_str = [recall_str, precision_str, f1_score_str, mean_iou_str]
    results_file = os.path.join(output_dir, "results_"+mode+".json")

    metrices_str_categorical = {}
    metrices_str = collections.defaultdict(dict)
    for k, v in metrices["results_dic"].items():
        class_name = str(k)
        for metric_type, val in v.items():
            metrices_str[str(metric_type)][class_name] = str(val)
    metrices_str_categorical["categorical"] = metrices_str

    # writing the results to a file
    with open(results_file, 'w') as fp:
        json.dump(str(metrices["results_dic"]), fp)
    s_logger = status_logging.get_status_logger()
    s_logger.kpi = {
        "Mean IOU": metrices["mean_iou_index"],
        "Average precision": metrices["prec"],
        "Average recall": metrices["rec"],
        "F1 score": metrices["fmes"],
        "model size": params["model_size"],
    }
    s_logger.write(
        data=metrices_str_categorical,
        status_level=status_logging.Status.RUNNING)
    for result in results_str:
        # This will print the results to the stdout
        print(result+"\n")


def run_evaluate_tlt(dataset, params, unet_model, load_graph, key,
                     output_dir, model_path, target_classes):
    """Run the prediction followed by evaluation using the estimator.

    Args:
        estimator: estimator object wrapped with run config parameters.
        dataset: Dataset object fro the dataloader utility.
        params: Parameters to feed to Estimator.
        unet_model: Keras Unet Model.
        profile_logger: Logging the Evaluation updates.
        key: The key to encrypt the model.
        output_dir: The directory where the results file is saved.

    """

    backends = [StdOutBackend(Verbosity.VERBOSE)]
    backends.append(JSONStreamBackend(Verbosity.VERBOSE, output_dir+"/log.txt"))
    profile_logger = Logger(backends)

    gpu_options = tf.compat.v1.GPUOptions()
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        save_summary_steps=1,
        tf_random_seed=None,
        session_config=config)
    if load_graph:
        checkpoint_path, model_json, _ = get_pretrained_ckpt(model_path, key)
    else:
        checkpoint_path, model_json = _extract_ckpt(model_path, key)
    if params.load_graph:
        # Need to set it True if directly loading pruned tlt model
        assert model_json, \
            "Load graph should be set only when inferring a pruned/re-trained model."
        params["model_json"] = model_json
    estimator = tf.estimator.Estimator(
        model_fn=unet_fn,
        model_dir=params.model_dir,
        config=run_config,
        params=params)

    hooks = None
    if params.benchmark:
        hooks = [ProfilingHook(profile_logger,
                               batch_size=params.batch_size,
                               log_every=params.log_every,
                               warmup_steps=params.warmup_steps,
                               mode="test")]
    predictions = estimator.predict(
        input_fn=lambda: dataset.eval_fn(
            count=1), hooks=hooks, checkpoint_path=checkpoint_path,
        )

    logger.info("Starting Evaluation.")
    print_compute_metrics(dataset, predictions, output_dir, target_classes, params,
                          mode="tlt")


def run_evaluate_trt(model_path, experiment_spec, output_dir, dataset,
                     target_classes, params, key=None):
    """Run the evaluate loop using the estimator.

    Args:
        model_path: The path string where the trained model needs to be saved.
        experiment_spec: Experiment spec proto.
        output_dir: Folder to save the results text file.
        dataset: Dataset object.
        key: Key to encrypt the model.

    """
    num_conf_mat_classes = params.num_conf_mat_classes
    activation = params.activation
    evaluator = Evaluator(keras_model=None, trt_engine_path=model_path,
                          dataset=dataset, batch_size=dataset._batch_size,
                          activation=activation, num_conf_mat_classes=num_conf_mat_classes)
    predictions = evaluator.evaluate(dataset.image_names_list, dataset.masks_names_list)

    print_compute_metrics(dataset, predictions, output_dir, target_classes,
                          params, mode="trt")


def evaluate_unet(model_path, experiment_spec, output_dir, key=None):
    """Run the evaluate loop using the estimator.

    Args:
        model_dir: The path string where the trained model needs to be saved.
        experiment_spec: Experiment spec proto.
        output_dir: Folder to save the results text file.
        key: Key to encrypt the model.

    """

    # Initialize the environment
    initialize(experiment_spec)
    # Initialize Params
    params = initialize_params(experiment_spec)
    target_classes = build_target_class_list(
        experiment_spec.dataset_config.data_class_config)
    # Build run config
    model_config = select_model_proto(experiment_spec)
    unet_model = build_model(m_config=model_config,
                             target_class_names=target_classes)
    model_dir = os.path.abspath(os.path.join(model_path, os.pardir))
    model_ext = os.path.splitext(model_path)[1]

    custom_objs = None
    model_json = None
    # Update custom_objs with Internal TAO custom layers
    custom_objs = get_custom_objs(model_arch=model_config.arch)
    params = update_model_params(params=params, unet_model=unet_model,
                                 experiment_spec=experiment_spec,
                                 key=key, target_classes=target_classes,
                                 results_dir=model_dir,
                                 phase="val",
                                 model_json=model_json,
                                 custom_objs=custom_objs)
    params["model_size"] = get_model_file_size(model_path)

    if params.enable_qat and not params.load_graph:
        # Eval is done by using model_json of pruned ckpt if load_graph is set
        # QAT nodes are added only for non-pruned graph
        # We add QDQ nodes before session is formed
        img_height, img_width, img_channels = \
            experiment_spec.model_config.model_input_height, \
            experiment_spec.model_config.model_input_width, \
            experiment_spec.model_config.model_input_channels
        model_qat_json = unet_model.construct_model(
            input_shape=(img_channels, img_height, img_width),
            pretrained_weights_file=params.pretrained_weights_file,
            enc_key=params.key, model_json=params.model_json,
            features=None, construct_qat=True)
        model_qat_json = save_tmp_json(model_qat_json)
        params.model_json = model_qat_json

    dataset = Dataset(
                      batch_size=params.batch_size,
                      fold=params.crossvalidation_idx,
                      augment=params.augment,
                      params=params,
                      phase="val",
                      target_classes=target_classes)

    if model_ext in ['.tlt', '']:
        run_evaluate_tlt(dataset, params, unet_model, params.load_graph,
                         key, output_dir, model_path, target_classes)
    elif model_ext in ['.engine', '.trt']:
        run_evaluate_trt(model_path, experiment_spec, output_dir, dataset,
                         target_classes, params, key=key)
    else:
        raise ValueError("Model extension needs to be either .engine or .trt.")


def run_experiment(model_path, config_path, output_dir,
                   override_spec_path=None, key=None):
    """
    Launch experiment that evaluates the model.

    NOTE: Do not change the argument names without verifying that cluster submission works.

    Args:
        model_dir (str): The model path that contains the latest checkpoint for evaluating.
        config_path (list): List containing path to a text file containing a complete experiment
            configuration and possibly a path to a .yml file containing override parameter values.
        output_dir (str): Path to a folder where the output of the evaluation .
            If the folder does not already exist, it will be created.
        override_spec_path (str): Absolute path to yaml file which is used to overwrite some of the
            experiment spec parameters.
        key (str): Key to save and load models from tlt.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    status_file = os.path.join(output_dir, "status.json")
    status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=True,
                append=False,
                verbosity=logger.getEffectiveLevel()
            )
        )
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.STARTED,
        message="Starting UNet Evaluation"
    )
    logger.debug("Starting experiment.")

    # Load experiment spec.
    if config_path is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", config_path)

        # The spec in experiment_spec_path has to be complete.
        # Default spec is not merged into experiment_spec.
        experiment_spec = load_experiment_spec(
            config_path, merge_from_default=False)
    else:
        logger.info("Loading default ISBI single class experiment spec.")
        experiment_spec = load_experiment_spec()

    evaluate_unet(model_path, experiment_spec, output_dir, key=key)

    logger.debug("Experiment complete.")


def build_command_line_parser(parser=None):
    """
    Parse command-line flags passed to the evaluation script.

    Returns:
      Namespace with all parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='Evaluate',
                                         description='Evaluate the segmentation model.')

    default_experiment_path = os.path.join(os.path.expanduser('~'), 'experiments')

    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        default=None,
        help='Path to spec file. Absolute path or relative to working directory. \
            If not specified, default spec from spec_loader.py is used.'
    )
    parser.add_argument(
        '-o',
        '--results_dir',
        type=str,
        default=default_experiment_path,
        help='Path to a folder where experiment outputs metrics json should be \
              written.'
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        default=default_experiment_path,
        help='Path to a folder from where the model should be taken for evaluation.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Set verbosity level for the logger.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        default="",
        required=False,
        help='The key to load model provided for evaluation.'
    )
    # Dummy arguments for Deploy
    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=False,
        default=None,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '-l',
        '--label_dir',
        type=str,
        required=False,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=1,
        help=argparse.SUPPRESS
    )
    return parser


def parse_command_line_args(cl_args=None):
    """Parser command line arguments to the trainer.

    Args:
        cl_args(sys.argv[1:]): Arg from the command line.

    Returns:
        args: Parsed arguments using argparse.
    """
    parser = build_command_line_parser(parser=None)
    args = parser.parse_args(cl_args)
    return args


def main(args=None):
    """Run the evaluation process."""
    args = parse_command_line_args(args)

    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'

    # Configure the logger.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=verbosity)

    # Configure tf logger verbosity.
    tf.logging.set_verbosity(tf.logging.INFO)
    run_experiment(config_path=args.experiment_spec,
                   model_path=args.model_path,
                   output_dir=args.results_dir,
                   key=args.key)


if __name__ == "__main__":
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Evaluation finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
