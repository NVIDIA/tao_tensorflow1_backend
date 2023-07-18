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
"""A class to evaluate an FpeNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import logging
import os
from time import gmtime, strftime

import keras
from keras import backend as K
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import TAOObject
from nvidia_tao_tf1.core.coreobject import save_args
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utilities.model_file_processing import save_best_model
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import model_io
from nvidia_tao_tf1.cv.fpenet.models.custom.softargmax import Softargmax

logger = logging.getLogger(__name__)

EXTENSIONS = ["hdf5", "tlt"]


class FpeNetEvaluator(TAOObject):
    """Class for running evaluation on a trained FpeNet model."""

    @save_args
    def __init__(self,
                 model,
                 dataloader,
                 save_dir,
                 mode='validation',
                 visualizer=None,
                 enable_viz=False,
                 num_keypoints=80,
                 loss=None,
                 model_path=None,
                 key=None,
                 steps_per_epoch=None,
                 **kwargs):
        """Initialization for evaluator.

        Args:
            model (nvidia_tao_tf1.blocks.Model): A trained FpeNet model for evaluation.
            dataloader (nvidia_tao_tf1.blocks.Dataloader): Instance dataloader to load
                                                    evaluation images and masks.
            save_dir (str): The full path where all the model files are saved.
            mode (str): 'validation' or 'kpi_testing'.
            visualizer (driveix.fpenet.FpeNetVisualizer): Instance of Fpenet visualizer.
            enable_viz (bool): Flag to enable evaluation visuzlization.
            num_keypoints (int): Number of facial keypoints.
            loss (driveix.fpenet.FpeLoss): Instance of Fpenet Loss.
            model_path (str): The model path to be evaluated.
            key (str): Key to load tlt file.
        """
        super(FpeNetEvaluator, self).__init__(**kwargs)
        self.model = model
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.model_path = model_path
        self.visualizer = visualizer
        self.enable_viz = enable_viz
        self.loss = loss
        self.mode = mode
        self.num_keypoints = num_keypoints
        self.batch_size = self._steps = self.evaluation_tensors = None
        self.crop_size = self.dataloader.image_width
        self._key = key
        self._steps_per_epoch = steps_per_epoch

    def build(self):
        """Load a trained FpeNet model and data for evaluation."""
        self.batch_size = self.dataloader.batch_size

        if self.mode == 'kpi_testing':
            images, ground_truth_labels, num_samples, occ_masking_info, \
                    face_bbox, image_names = \
                    self.dataloader(repeat=True,
                                    phase=self.mode)
        else:
            images, ground_truth_labels, num_samples, occ_masking_info = \
                    self.dataloader(repeat=True,
                                    phase=self.mode)

        self._steps = num_samples // self.batch_size

        if not self.model_path:
            for extension in EXTENSIONS:
                model_file = os.path.join(self.save_dir, f'model.{extension}')
                self.model_path = model_file if self.mode != 'validation' \
                    and os.path.exists(model_file) else None
                break

        # Get input tensors.
        input_tensor = keras.layers.Input(
            tensor=images, name='input_face_images')

        # Set testing phase.
        keras.backend.set_learning_phase(0)

        if self.mode == 'validation':
            if not hasattr(self.model, 'keras_model'):
                raise ValueError(
                    "The model graph should be built before calling build() \
                                  in validation mode.")
            predictions = self.model.keras_model([input_tensor])

        elif self.mode == 'kpi_testing':
            logger.info('model_path: {}'.format(self.model_path))
            if self.model_path is None or not os.path.isfile(self.model_path):
                raise ValueError(
                    "Please provide a valid fpenet file path for evaluation.")
            self.model.keras_model = model_io(self.model_path,
                                              enc_key=self._key,
                                              custom_objects={"Softargmax": Softargmax})
            predictions = self.model.keras_model([input_tensor])

        else:
            raise ValueError("Evaluation mode not supported.")

        # extract keypoints and confidence from model output
        predictions_coord = K.reshape(predictions[0], (self.batch_size, self.num_keypoints, 2))
        predictions_conf = K.reshape(predictions[1], (self.batch_size, self.num_keypoints))

        # vizualization of images with predicted keypoints overlay
        if (self.enable_viz):
            self.visualizer.visualize_images(
                images, predictions_coord, viz_phase=self.mode)

        # compute loss for validation/testing data
        evaluation_cost, mouth_cost, eyelid_cost = self.loss(ground_truth_labels[0],
                                                             predictions_coord,
                                                             ground_truth_labels[1],
                                                             occ_masking_info,
                                                             num_keypoints=self.num_keypoints)

        if self.mode == 'kpi_testing':
            self.evaluation_tensors = [predictions_coord,
                                       predictions_conf,
                                       ground_truth_labels[0],
                                       evaluation_cost,
                                       mouth_cost,
                                       eyelid_cost,
                                       ground_truth_labels[1],
                                       face_bbox,
                                       image_names]
        else:
            self.evaluation_tensors = [predictions_coord,
                                       predictions_conf,
                                       ground_truth_labels[0],
                                       evaluation_cost,
                                       mouth_cost,
                                       eyelid_cost]

        # Restores learning phase.
        keras.backend.set_learning_phase(1)

    def evaluate(self, sess=None, global_step=None):
        """Evaluate a loaded FpeNet model.

        Args:
            sess (tf.Session): User defined session if exists.
            global_step (int): Global step in the graph if doing validation.
        Returns:
            evaluation_cost (float): An average loss on the evaluation dataset.
        """
        if self.evaluation_tensors is None:
            raise ValueError("Evaluator must be built before evaluation!")

        if sess is None:
            sess = keras.backend.get_session()
            sess.run(
                tf.group(tf.local_variables_initializer(),
                         tf.tables_initializer(),
                         *tf.get_collection('iterator_init')))

        evaluation_cost = 0.
        evaluation_mouth_cost = 0.
        evaluation_eyelids_cost = 0.
        results, occlusion_gt, image_names, face_bbox, pred_conf = [], [], [], [], []

        progress_list = range(self._steps)

        for _ in progress_list:
            if self.mode == 'kpi_testing':
                batch_prediction, batch_prediction_conf, batch_gt, batch_evaluation_cost, \
                    batch_mouth_cost, batch_eyelids_cost, \
                    batch_gt_occ, batch_face_bbox, \
                    batch_image_names = sess.run(self.evaluation_tensors)
            else:
                batch_prediction, batch_prediction_conf, batch_gt, batch_evaluation_cost, \
                         batch_mouth_cost, batch_eyelids_cost = \
                         sess.run(self.evaluation_tensors)

            evaluation_cost += batch_evaluation_cost / self._steps
            evaluation_mouth_cost += batch_mouth_cost / self._steps
            evaluation_eyelids_cost += batch_eyelids_cost / self._steps

            # Write ground truth and prediction together for each data point.
            for i in range(self.batch_size):
                result = np.stack([batch_gt[i], batch_prediction[i]], axis=-1)
                results.append(result)
                pred_conf.append(batch_prediction_conf[i])

                if self.mode == 'kpi_testing':
                    occlusion_gt.append(batch_gt_occ[i])
                    face_bbox.append(batch_face_bbox[i])
                    image_names.append(batch_image_names[i])

        if self.mode != 'validation' and results == []:
            raise ValueError(
                "Need valid 'test_file_name' in experiment spec for evaluation!"
            )

        if self.mode == 'validation' and self.save_dir is not None:

            final_errors, _ = compute_error_keypoints(results)
            epoch = int(global_step / self._steps_per_epoch)
            save_best_model(
                self.save_dir, epoch, evaluation_cost, epoch_based_checkpoint=True,
                extension="hdf5"
            )

            [(mean_err_x, std_err_x), (mean_err_y, std_err_y),
             (mean_err_xy, std_err_xy)] = final_errors
            with open(os.path.join(self.save_dir, 'validation.log'),
                      'a+') as f:
                cur_time = strftime("%Y-%m-%d %H:%M:%S UTC", gmtime())
                f.write(
                    '{} - global_step {} : {} (total), {} (mouth), {} (eyelids) '
                    '| Mean errors in px: '.format(
                        cur_time, global_step, evaluation_cost,
                        evaluation_mouth_cost, evaluation_eyelids_cost))
                f.write(
                    'err_x: {:.2f} (+/-{:.2f}), err_y: {:.2f} (+/-{:.2f}), err_xy: {:.2f} '
                    '(+/-{:.2f})\n'.format(mean_err_x, std_err_x, mean_err_y,
                                           std_err_y, mean_err_xy, std_err_xy))

        elif self.save_dir is not None:
            # Write predictions and gt to json files only in kpi_testing phase
            output_filename = os.path.join(self.save_dir, self.mode + '_all_data.json')
            print('writing predictions to {}'.format(output_filename))

            # Write predictions by masking occluded points for KPI data
            write_errors_per_region(results,
                                    pred_conf,
                                    occlusion_gt,
                                    face_bbox,
                                    image_names,
                                    self.save_dir,
                                    self.crop_size,
                                    self.mode)

        else:
            raise ValueError("Evaluation mode not supported or checkpoint_dir missing.")

        logger.info('Validation #{}: {}'.format(global_step, evaluation_cost))
        kpi_data = {
            "evaluation_cost ": evaluation_cost
        }
        s_logger = status_logging.get_status_logger()
        if isinstance(s_logger, status_logging.StatusLogger):
            s_logger.kpi = kpi_data
            s_logger.write(
                status_level=status_logging.Status.RUNNING,
                message="Evaluation metrics generated."
            )

        return evaluation_cost


def compute_error_keypoints(results):
    """Compute the final keypoints error using ground truth and prediction results.

    Args:
        result (list): List of ground truth and prediction for each data point.
    Returns:
        final_errors (list): mean and std error for x, y, xy.
        num_samples (int): Number of samples in results.
    """
    num_samples = len(results)
    results = np.asarray(results)

    # Get ground truth and prediction lists.
    gt_x = results[:, :, 0, 0]
    gt_y = results[:, :, 1, 0]
    pred_x = results[:, :, 0, 1]
    pred_y = results[:, :, 1, 1]

    # Calculate the error.
    error_x = np.absolute([a - b for a, b in zip(gt_x, pred_x)])
    error_y = np.absolute([a - b for a, b in zip(gt_y, pred_y)])

    mean_err_x = np.mean(error_x)
    mean_err_y = np.mean(error_y)

    std_err_x = np.std(error_x)
    std_err_y = np.std(error_y)

    error_xy = np.sqrt(np.power(error_x, 2) + np.power(error_y, 2))
    mean_err_xy = np.mean(error_xy)
    std_err_xy = np.std(error_xy)

    final_errors = [(mean_err_x, std_err_x), (mean_err_y, std_err_y)]

    final_errors.append((mean_err_xy, std_err_xy))

    return final_errors, num_samples


def dump_json(save_dir, mode, image_names, results, pred_conf, occlusion_gt, face_bbox):
    """
    Utility function to dump all data into a json.

    Args:
        save_dir (str): Path to save results.
        mode (str): run mode used as post script for file name.
        image_names (list): List of image names with paths.
        results (list): List of ground truth and prediction for each data point.
        pred_conf (list): List of predicted softargmax confidence values.
        occlusion_gt (list): List of ground truth occlusion flags for each data point.
        face_bbox (list): Ground truth face bounding box in format (x, y, h, w).

    Returns:
        None
    """
    num_samples = len(image_names)
    data = []
    for i in range(num_samples):
        sample = {}
        sample['image_path'] = str(image_names[i], 'utf-8')
        sample['face_box'] = face_bbox[i].tolist()
        results = np.asarray(results)
        sample['gt_keypoints'] = results[i, :, :, 0].tolist()
        sample['pred_keypoints'] = results[i, :, :, 1].tolist()
        sample['gt_occlusions'] = occlusion_gt[i].tolist()
        sample['pred_conf'] = pred_conf[i].tolist()
        data.append(sample)

    with open(os.path.join(save_dir, mode + '_all_data.json'), 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)
    print('KPI results saved to: {}'.format(os.path.join(save_dir, mode + '_all_data.json')))


def write_errors_per_region(results,
                            pred_conf,
                            occlusion_gt,
                            face_bbox,
                            image_names,
                            save_dir,
                            crop_size,
                            mode):
    """Compute the errors per facial region and write them to a file.

    Args:
        results (list): List of ground truth and prediction for each data point.
        pred_conf (list): List of predicted softargmax confidence values.
        occlusion_gt (list): List of ground truth occlusion flags for each data point.
        face_bbox (list): Ground truth face bounding box in format (x, y, h, w).
        image_names (list): List of image names with paths.
        save_dir (string): Path to save results.
        crop_size (int): Face crop size
        mode (string): Run mode used as post script for file name.
    """
    # dump results as json for offline processing/analysis
    dump_json(save_dir, mode, image_names, results, pred_conf, occlusion_gt, face_bbox)

    num_images = len(results)

    results = np.asarray(results)
    occlusion_gt = np.asarray(occlusion_gt)
    face_bbox = np.asarray(face_bbox)

    nkeypoints = results.shape[1]

    # face crop size for error normalization
    face_bbox_crop_size = float(crop_size)

    # Various face region keypoints for region based error.
    # ordering of points listed here-
    # https://docs.google.com/document/d/13q8NciZtGyx5TgIgELkCbXGfE7PstKZpI3cENBGWkVw/edit#
    if nkeypoints in [68, 80]:
        region_idx = {'All': range(0, nkeypoints),
                      'Eyes': range(36, 48),
                      'Nose': range(27, 36),
                      'Mouth': range(48, 68),
                      'Eyebrows': range(17, 27),
                      'Chin': range(0, 17),
                      'HP': [8, 17, 21, 22, 26, 31, 35, 36, 39, 42, 45, 48, 54, 57],
                      'Pupil': range(68, 76),
                      'Ears': range(76, 80)}
        regions = ['All', 'Eyes', 'Nose', 'Mouth', 'Eyebrows', 'Chin', 'HP', 'Pupil', 'Ears']
        if nkeypoints == 68:
            regions = regions[:-2]
    else:
        region_idx = {'All': range(0, nkeypoints)}
        regions = ['All']

    # normalize GT and predictions with face bbox size
    results = results*(face_bbox[:, 2].reshape(-1, 1, 1, 1)/face_bbox_crop_size)

    # mask all GT occluded points
    results_occ_masked = np.multiply(results, occlusion_gt.reshape([num_images, nkeypoints, 1, 1]))

    # compute error per point
    points_error = []
    points_error_all = []
    for point in range(nkeypoints):

        # only get the samples for a point which are GT not occluded
        results_non_occ = [results_occ_masked[x, point:point+1, :, :] for x in range(num_images)
                           if occlusion_gt[x, point] != 0.0]
        results_all = [results[x, point:point+1, :, :] for x in range(num_images)]

        # get the point error non occluded
        if len(results_non_occ) > 0:
            point_error = compute_error_keypoints(results_non_occ)
        else:
            point_error = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], -1
        points_error.append(point_error)

        # get the point error all points
        point_error_all = compute_error_keypoints(results_all)
        points_error_all.append(point_error_all)

    # save error for all points
    output_filename = os.path.join(save_dir, mode + '_error_per_point.csv')

    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Error per point',
                         'mean_pxl_err_xy_all',
                         'mean_pxl_err_xy_non_occ',
                         'num_samples_non_occluded'])

        for point in range(nkeypoints):
            # get error per point
            results_point = points_error[point][0][-1][0]
            results_point_all = points_error_all[point][0][-1][0]
            writer.writerow([str(point),
                             str(results_point_all),
                             str(results_point),
                             str(points_error[point][-1])])

    # save points for all regions
    output_filename = os.path.join(save_dir, mode + '_error_per_region.csv')
    with open(output_filename, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Error per region',
                         'mean_region_err_xy_all',
                         'mean_region_err_xy_non_occ'])

        for region in regions:
            # get error per region
            results_region = [points_error[x][0][-1][0] for x in region_idx[region]]
            region_error = sum(results_region)/len(results_region)

            results_region_all = [points_error_all[x][0][-1][0] for x in region_idx[region]]
            region_error_all = sum(results_region_all)/len(results_region_all)

            writer.writerow([region,
                            str(region_error_all),
                            str(region_error)])
