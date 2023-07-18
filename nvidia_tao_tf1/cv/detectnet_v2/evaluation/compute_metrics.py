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

"""A class to compute detection metrics on the gridbox model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import sys

import numpy as np

from tqdm import trange

logger = logging.getLogger(__name__)


def iou(boxes1, boxes2, border_pixels='half'):
    '''
    numpy version of element-wise iou.

    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates
            for one box in the format specified by `coords` or a 2D Numpy array of shape
            `(m, 4)` containing the coordinates for `m` boxes. If `mode` is set to 'element_wise',
            the shape must be broadcast-compatible with `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for
            one box in the format specified by `coords` or a 2D Numpy array of shape `(n, 4)`
            containing the coordinates for `n` boxes. If `mode` is set to 'element_wise', the
            shape must be broadcast-compatible with `boxes1`.`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float
        containing values in [0,1], the Jaccard similarity of the boxes in `boxes1` and
        `boxes2`. 0 means there is no overlap between two given boxes, 1 means their
        coordinates are identical.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}."
                         .format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}."
                         .format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("Boxes list last dim should be 4 but got shape {} and {}, respectively."
                         .format(boxes1.shape, boxes2.shape))

    # Set the correct coordinate indices for the respective formats.
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    # Compute the union areas.
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    # Compute the IoU.

    min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
    max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

    # Compute the side lengths of the intersection rectangles.
    side_lengths = np.maximum(0, max_xy - min_xy + d)

    intersection_areas = side_lengths[:, 0] * side_lengths[:, 1]

    boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * \
        (boxes1[:, ymax] - boxes1[:, ymin] + d)
    boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * \
        (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas


class ComputeMetrics(object):
    '''
    Simple class to compute metrics for a detection model.

    Returns VOC 2009 metrics: mAP, precision, recall.
    '''

    def __init__(self,
                 clustered_detections, ground_truth_labels,
                 image_width, image_height, target_class_names,
                 evaluation_config):
        '''
        Init function.

        Arguments:
            clustered_detections: postprocessed final detection tensor
            ground_truth_labels: tensor containing ground truth
            target_class_names: mapping of target class names
            evaluation config: for getting iou thresholds, min/max box config for evaluations
        '''
        self.target_class_names = target_class_names
        self.n_classes = len(self.target_class_names)
        gt_format = {'class_id': 0, 'xmin': 1,
                     'ymin': 2, 'xmax': 3, 'ymax': 4}
        self.gt_format = gt_format
        self.prediction_results = None
        self.num_gt_per_class = None
        self.true_positives = None
        self.false_positives = None
        self.cumulative_true_positives = None
        self.cumulative_false_positives = None
        self.valid_evaluation_modes = {0: "sample",
                                       1: "integrate"}
        # "Cumulative" means that the i-th element in each list represents the precision for the
        # first i highest condidence predictions for that class.
        self.cumulative_precisions = None
        # "Cumulative" means that the i-th element in each list represents the recall for the first
        # i highest condidence predictions for that class.
        self.cumulative_recalls = None
        self.average_precisions = None
        self.mean_average_precision = None
        self.image_ids = []
        self.image_labels = {}
        self.image_height = image_height
        self.image_width = image_width
        # specifications from evaluation config proto
        self.min_iou_thresholds = \
            evaluation_config.minimum_detection_ground_truth_overlap
        ap_key = evaluation_config.average_precision_mode
        self.average_precision_mode = self.valid_evaluation_modes[ap_key]
        self.detection_spec_for_gtruth_matching = \
            evaluation_config.evaluation_box_configs
        # call preparation of gtruth and detection config
        self._prepare_internal_structures(
            clustered_detections, ground_truth_labels)

    def __call__(self,
                 round_confidences=False,
                 border_pixels='include',
                 sorting_algorithm='quicksort',
                 num_recall_points=11,
                 ignore_neutral_boxes=True,
                 verbose=True):
        '''
        Computes the mean average precision of the given Keras SSD model on the given dataset.

        Optionally also returns the averages precisions, precisions, and recalls.

        All the individual steps of the overall evaluation algorithm can also be called separately
        (check out the other methods of this class) but this runs the overall algorithm all at once.

        Arguments:
            img_height (int): The input image height for the model.
            img_width (int): The input image width for the model.
            batch_size (int): The batch size for the evaluation.
            data_generator_mode (str, optional): Either of 'resize' and 'pad'. If 'resize', the
                input images will be resized (i.e. warped) to `(img_height, img_width)`. This mode
                does not preserve the aspect ratios of the images. If 'pad', the input images will
                be first padded so that they have the aspect ratio defined by `img_height` and
                `img_width` and then resized to `(img_height, img_width)`. This mode preserves the
                aspect ratios of the images.
            round_confidences (int, optional): `False` or an integer that is the number of decimals
                that the prediction confidences will be rounded to. If `False`, the confidences will
                not be rounded.
            matching_iou_threshold (float, optional): A prediction will be considered a true
                positive if it has a Jaccard overlap of at least `matching_iou_threshold` with any
                ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should
                use. This argument accepts any valid sorting algorithm for Numpy's `argsort()`
                function. You will usually want to choose between 'quicksort' (fastest and most
                memory efficient, but not stable) and 'mergesort' (slight slower and less memory
                efficient, but stable). The official Matlab evaluation algorithm uses a stable
                sorting algorithm, so this algorithm is only guaranteed to behave identically if you
                choose 'mergesort' as the sorting algorithm, but it will almost always behave
                identically even if you choose 'quicksort' (but no guarantees).
            average_precision_mode (str, optional): Can be either 'sample' or 'integrate'. In the
                case of 'sample', the average precision will be computed according to the Pascal VOC
                formula that was used up until VOC 2009, where the precision will be sampled for
                `num_recall_points` recall values. In the case of 'integrate', the average precision
                will be computed according to the Pascal VOC formula that was used from VOC 2010
                onward, where the average precision will be computed by numerically integrating
                over the whole preciscion-recall curve instead of sampling individual points from
                it. 'integrate' mode is basically just the limit case of 'sample' mode as the number
                of sample points increases.
            num_recall_points (int, optional): The number of points to sample from the
                precision-recall-curve to compute the average precisions. In other words, this is
                the number of equidistant recall values for which the resulting precision will be
                computed. 11 points is the value used in the official Pascal VOC 2007 detection
                evaluation algorithm.
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations
                indicating whether a ground truth bounding box is supposed to either count or be
                neutral for the evaluation, this argument decides what to do with these annotations.
                If `False`, even boxes that are annotated as neutral will be counted into the
                evaluation. If `True`, neutral boxes will be ignored for the evaluation. An example
                for evaluation-neutrality are the ground truth boxes annotated as "difficult" in the
                Pascal VOC datasets, which are usually treated as neutral for the evaluation.
            return_precisions (bool, optional): If `True`, returns a nested list containing the
                cumulative precisions for each class.
            return_recalls (bool, optional): If `True`, returns a nested list containing the
                cumulative recalls for each class.
            return_average_precisions (bool, optional): If `True`, returns a list containing the
                average precision for each class.
            verbose (bool, optional): If `True`, will print out the progress during runtime.

        Returns:
            A float, the mean average precision, plus any optional returns specified in the
                arguments.
        '''
        ############################################################################################
        # Get the total number of ground truth boxes for each class.
        ############################################################################################

        self.get_num_gt_per_class(ignore_neutral_boxes=ignore_neutral_boxes,
                                  verbose=False,
                                  ret=False)

        ############################################################################################
        # Match predictions to ground truth boxes for all classes.
        ############################################################################################

        self.match_predictions(ignore_neutral_boxes=ignore_neutral_boxes,
                               matching_iou_threshold=self.min_iou_thresholds,
                               border_pixels=border_pixels,
                               sorting_algorithm=sorting_algorithm,
                               verbose=verbose,
                               ret=False)

        ############################################################################################
        # Compute the cumulative precision and recall for all classes.
        ############################################################################################

        self.compute_precision_recall(verbose=verbose, ret=False)

        ############################################################################################
        # Compute the average precision for this class.
        ############################################################################################

        self.compute_average_precisions(mode=self.average_precision_mode,
                                        num_recall_points=num_recall_points,
                                        verbose=verbose,
                                        ret=False)

        ############################################################################################
        # Compute the mean average precision.
        ############################################################################################

        mean_average_precision = self.compute_mean_average_precision(ret=True)

        ############################################################################################
        # Compile the returns.
        dict_average_precisions = {}
        for idx, t in enumerate(self.target_class_names):
            dict_average_precisions[t] = self.average_precisions[idx]
        composite_metrics = {}
        composite_metrics['mAP'] = mean_average_precision
        composite_metrics['average_precisions'] = dict_average_precisions
        return composite_metrics

    def _check_if_bbox_is_valid(self, bbox, class_name):
        '''
        Checks if a box is valid based on evaluation config.

        Arguments:
           bbox  - [x1, y1, x2, y2]
           class name - class name (string)
           return: returns "True" is the box meets spec standards
        '''
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if h < self.detection_spec_for_gtruth_matching[class_name].minimum_height:
            return False
        if w < self.detection_spec_for_gtruth_matching[class_name].minimum_width:
            return False
        if h > self.detection_spec_for_gtruth_matching[class_name].maximum_height:
            return False
        if w > self.detection_spec_for_gtruth_matching[class_name].maximum_width:
            return False
        return True

    def _prepare_internal_structures(self, clustered_detections,
                                     ground_truth_labels):
        '''
        configures detections and ground truth tensor to right format.

        Also fills up the gtruth tensor
        Arguments:
           clustered_detections: tensor containing postprocessed detections
           ground_truth_labels: tensor containing ground truth
        '''
        logger.debug("Preparing internal datastructures")
        target_class_names = self.target_class_names
        results = [list() for _ in range(len(target_class_names))]
        self.image_ids = []
        for frame_index, frame_ground_truths in enumerate(ground_truth_labels):
            gtruth_per_frame = []
            self.image_ids.append(frame_index)
            for target_class_id, target_class in enumerate(target_class_names):
                for box_struct in clustered_detections[target_class][frame_index]:
                    # for box_struct in frame_ground_truths:
                    bbox = box_struct.bbox
                    if (self._check_if_bbox_is_valid(bbox, target_class)):
                        confidence = box_struct.confidence
                        prediction = (int(frame_index), confidence,
                                      round(bbox[0]), round(bbox[1]),
                                      round(bbox[2]), round(bbox[3]))
                        results[target_class_id].append(prediction)
            # build gtruth
            for box_struct in frame_ground_truths:
                bbox = box_struct.bbox
                cid = box_struct.class_name
                if cid == '-1':
                    continue
                if not (self._check_if_bbox_is_valid(bbox, cid)):
                    continue
                cid = target_class_names.index(cid)
                gtruth = (int(cid),
                          round(bbox[0]), round(bbox[1]),
                          round(bbox[2]), round(bbox[3]))
                gtruth_per_frame.append(gtruth)
            self.image_labels[frame_index] = gtruth_per_frame
        self.prediction_results = results
        logger.debug("Internal datastructure prepared.")

    def write_predictions_to_txt(self,
                                 classes=None,
                                 out_file_prefix='comp3_det_test_',
                                 verbose=True):
        '''
        Writes the predictions for all classes to txt according to the Pascal VOC results format.

        Arguments:
            classes (list, optional): `None` or a list of strings containing the class names of all
                classes in the dataset, including some arbitrary name for the background class. This
                list will be used to name the output text files. The ordering of the names in the
                list represents the ordering of the classes as they are predicted by the model,
                i.e. the element with index 3 in this list should correspond to the class with class
                ID 3 in the model's predictions. If `None`, the output text files will be named by
                their class IDs.
            out_file_prefix (str, optional): A prefix for the output text file names. The suffix to
                each output text file name will be the respective class name followed by the `.txt`
                file extension. This string is also how you specify the directory in which the
                results are to be saved.
            verbose (bool, optional): If `True`, will print out the progress during runtime.

        Returns:
            None.
        '''

        if self.prediction_results is None:
            raise ValueError("There are no prediction results. You must run `predict_on_dataset()` \
            before calling this method.")

        # We generate a separate results file for each class.
        for class_id in range(self.n_classes):
            logger.debug("Writing results file for class {}/{}.".format(class_id+1,
                                                                        self.n_classes+1))

            if classes is None:
                class_suffix = '{:04d}'.format(class_id)
            else:
                class_suffix = classes[class_id]

            results_file = open('{}{}.txt'.format(
                out_file_prefix, class_suffix), 'w')
            logger.debug("Print out the file of results path: {}".format(results_file))
            for prediction in self.prediction_results[class_id]:
                prediction_list = list(prediction)
                prediction_list[0] = '{:06d}'.format(int(prediction_list[0]))
                prediction_list[1] = round(prediction_list[1], 4)
                prediction_txt = ' '.join(map(str, prediction_list)) + '\n'
                results_file.write(prediction_txt)

            results_file.close()

        logger.debug("All results files saved.")

    def get_num_gt_per_class(self,
                             ignore_neutral_boxes=True,
                             verbose=True,
                             ret=False):
        '''
        Counts the number of ground truth boxes for each class across the dataset.

        Arguments:
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations
                indicating whether a ground truth bounding box is supposed to either count or be
                neutral for the evaluation, this argument decides what to do with these annotations.
                If `True`, only non-neutral ground truth boxes will be counted, otherwise all ground
                truth boxes will be counted.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the list of counts.

        Returns:
            None by default. Optionally, a list containing a count of the number of ground truth
                boxes for each class across the entire dataset.
        '''

        if self.image_labels is None:
            raise ValueError("Computing the number of ground truth boxes per class not possible, \
            no ground truth given.")

        num_gt_per_class = np.zeros(shape=(self.n_classes), dtype=np.int)
        class_id_index = self.gt_format['class_id']
        ground_truth = self.image_labels
        if verbose:
            logger.debug('Computing the number of positive ground truth boxes per class.')
            tr = trange(len(ground_truth), file=sys.stdout)
        else:
            tr = range(len(ground_truth))

        # Iterate over the ground truth for all images in the dataset.
        for i in tr:
            boxes = np.asarray(ground_truth[i])
            # Iterate over all ground truth boxes for the current image.
            for j in range(boxes.shape[0]):
                # If there is no such thing as evaluation-neutral boxes for
                # our dataset, always increment the counter for the respective
                # class ID.
                class_id = int(boxes[j, class_id_index])
                num_gt_per_class[class_id] += 1

        self.num_gt_per_class = num_gt_per_class

        if ret:
            return num_gt_per_class
        return None

    def match_predictions(self,
                          matching_iou_threshold,
                          ignore_neutral_boxes=True,
                          border_pixels='include',
                          sorting_algorithm='quicksort',
                          verbose=True,
                          ret=False):
        '''
        Matches predictions to ground truth boxes.

        Note that `predict_on_dataset()` must be called before calling this method.

        Arguments:
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations
                indicating whether a ground truth bounding box is supposed to either count or be
                neutral for the evaluation, this argument decides what to do with these annotations.
                If `False`, even boxes that are annotated as neutral will be counted into the
                evaluation. If `True`, neutral boxes will be ignored for the evaluation. An example
                for evaluation-neutrality are the ground truth boxes annotated as "difficult" in the
                Pascal VOC datasets, which are usually treated as neutral for the evaluation.
            matching_iou_threshold (per class threshold dict): A prediction will be considered true
                positive if it has a Jaccard overlap of at least `matching_iou_threshold` with any
                ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should
                use. This argument accepts any valid sorting algorithm for Numpy's `argsort()`
                function. You will usually want to choose between 'quicksort' (fastest and most
                memory efficient, but not stable) and 'mergesort' (slight slower and less memory
                efficient, but stable). The official Matlab evaluation algorithm uses a stable
                sorting algorithm, so this algorithm is only guaranteed to behave identically if you
                choose 'mergesort' as the sorting algorithm, but it will almost always behave
                identically even if you choose 'quicksort' (but no guarantees).
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the true and false positives.

        Returns:
            None by default. Optionally, four nested lists containing the true positives, false
            positives, cumulative true positives, and cumulative false positives for each class.
        '''
        if self.image_labels is None:
            raise ValueError("Matching predictions to ground truth boxes not possible, no ground \
            truth given.")

        if self.prediction_results is None:
            raise ValueError("There are no prediction results. You must run `predict_on_dataset()` \
            before calling this method.")

        class_id_gt = self.gt_format['class_id']
        xmin_gt = self.gt_format['xmin']
        ymin_gt = self.gt_format['ymin']
        xmax_gt = self.gt_format['xmax']
        ymax_gt = self.gt_format['ymax']

        # Convert the ground truth to a more efficient format for what we need
        # to do, which is access ground truth by image ID repeatedly.
        ground_truth = {}
        # Whether or not we have annotations to decide whether ground truth boxes should be neutral
        # or not.
        eval_neutral_available = False
        for i in range(len(self.image_ids)):
            image_id = self.image_ids[i]
            labels = self.image_labels[i]
            ground_truth[image_id] = np.asarray(labels)

        # The false positives for each class, sorted by descending confidence.
        true_positives = []
        # The true positives for each class, sorted by descending confidence.
        false_positives = []
        cumulative_true_positives = []
        cumulative_false_positives = []

        # Iterate over all classes.
        for class_id in range(self.n_classes):
            if not self.target_class_names[class_id] in self.min_iou_thresholds:
                raise ValueError("class {}, not in spec file for minimum overlap thresh"
                                 .format(self.target_class_names[class_id]))
            matching_iou_threshold = self.min_iou_thresholds[self.target_class_names[class_id]]
            predictions = self.prediction_results[class_id]
            # Store the matching results in these lists:

            # 1 for every prediction that is a true positive, 0 otherwise
            true_pos = np.zeros(len(predictions), dtype=np.int)
            # 1 for every prediction that is a false positive, 0 otherwise
            false_pos = np.zeros(len(predictions), dtype=np.int)
            # In case there are no predictions at all for this class, we're done here.
            if len(predictions) == 0:
                logger.debug("No predictions for class {}/{}".format(class_id + 1,
                                                                     self.n_classes))
                true_positives.append(true_pos)
                false_positives.append(false_pos)
                cumulative_true_positives.append(np.cumsum(true_pos))
                cumulative_false_positives.append(np.cumsum(false_pos))
                continue

            # Convert the predictions list for this class into a structured array so that we can
            # sort it by confidence.

            # Get the number of characters needed to store the image ID strings in the structured
            # array.

            # Create the data type for the structured array.
            preds_data_type = np.dtype([('image_id', 'int'),
                                        ('confidence', 'f4'),
                                        ('xmin', 'f4'),
                                        ('ymin', 'f4'),
                                        ('xmax', 'f4'),
                                        ('ymax', 'f4')])
            # Create the structured array
            predictions = np.array(predictions, dtype=preds_data_type)

            # Sort the detections by decreasing confidence.
            descending_indices = np.argsort(-predictions['confidence'], kind=sorting_algorithm)
            predictions_sorted = predictions[descending_indices]

            if verbose:
                tr = trange(len(predictions), file=sys.stdout)
                tr.set_description("Matching predictions to ground truth, class {}/{}."
                                   .format(class_id + 1, self.n_classes))
            else:
                tr = range(len(predictions.shape))

            # Keep track of which ground truth boxes were already matched to a detection.
            gt_matched = {}

            # Iterate over all predictions.
            for i in tr:

                prediction = predictions_sorted[i]
                image_id = prediction['image_id']
                # Convert the structured array element to a regular array.
                pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']]))
                # Get the relevant ground truth boxes for this prediction,
                # i.e. all ground truth boxes that match the prediction's
                # image ID and class ID.

                # The ground truth could either be a tuple with
                # `(ground_truth_boxes, eval_neutral_boxes)` or only `ground_truth_boxes`.
                if ignore_neutral_boxes and eval_neutral_available:
                    gt, eval_neutral = ground_truth[image_id]
                else:
                    gt = ground_truth[image_id]
                gt = np.asarray(gt)
                if gt.size == 0:
                    # If the image doesn't contain any objects of this class,
                    # the prediction becomes a false positive.
                    false_pos[i] = 1
                    continue
                else:
                    class_mask = gt[:, class_id_gt] == class_id
                    gt = gt[class_mask]
                if gt.size == 0:
                    # If the image doesn't contain any objects of this class,
                    # the prediction becomes a false positive.
                    false_pos[i] = 1
                    continue
                if ignore_neutral_boxes and eval_neutral_available:
                    eval_neutral = eval_neutral[class_mask]
                # Compute the IoU of this prediction with all ground truth boxes of the same class.
                overlaps = iou(boxes1=gt[:, [xmin_gt, ymin_gt, xmax_gt, ymax_gt]],
                               boxes2=pred_box,
                               border_pixels=border_pixels)
                # For each detection, match the ground truth box with the highest overlap.
                # It's possible that the same ground truth box will be matched to multiple
                # detections.
                gt_match_index = np.argmax(overlaps)
                gt_match_overlap = overlaps[gt_match_index]

                if gt_match_overlap < matching_iou_threshold:
                    # False positive, IoU threshold violated:
                    # Those predictions whose matched overlap is below the threshold become
                    # false positives.
                    false_pos[i] = 1
                else:
                    if (not (ignore_neutral_boxes and eval_neutral_available) or
                            (eval_neutral[gt_match_index] is False)):
                        # If this is not a ground truth that is supposed to be evaluation-neutral
                        # (i.e. should be skipped for the evaluation) or if we don't even have the
                        # concept of neutral boxes.
                        if not (image_id in gt_matched):
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been
                            # matched to a different prediction already, we have a true positive.
                            true_pos[i] = 1
                            gt_matched[image_id] = np.zeros(
                                shape=(gt.shape[0]), dtype=np.bool)
                            gt_matched[image_id][gt_match_index] = True
                        elif not gt_matched[image_id][gt_match_index]:
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been
                            # matched to a different prediction already, we have a true positive.
                            true_pos[i] = 1
                            gt_matched[image_id][gt_match_index] = True
                        else:
                            # False positive, duplicate detection:
                            # If the matched ground truth box for this prediction has already been
                            # matched to a different prediction previously, it is a duplicate
                            # detection for an already detected object, which counts as a false
                            # positive.
                            false_pos[i] = 1

            true_positives.append(true_pos)
            false_positives.append(false_pos)

            # Cumulative sums of the true positives
            cumulative_true_pos = np.cumsum(true_pos)
            # Cumulative sums of the false positives
            cumulative_false_pos = np.cumsum(false_pos)

            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)

        self.true_positives = true_positives
        self.false_positives = false_positives
        self.cumulative_true_positives = cumulative_true_positives
        self.cumulative_false_positives = cumulative_false_positives

        if ret:
            return true_positives, false_positives, cumulative_true_positives, \
                cumulative_false_positives
        return None

    def compute_precision_recall(self, verbose=True, ret=False):
        '''
        Computes the precisions and recalls for all classes.

        Note that `match_predictions()` must be called before calling this method.

        Arguments:
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the precisions and recalls.

        Returns:
            None by default. Optionally, two nested lists containing the cumulative precisions and
                recalls for each class.
        '''

        if (self.cumulative_true_positives is None) or (self.cumulative_false_positives is None):
            raise ValueError("True and false positives not available. You must run \
            `match_predictions()` before you call this method.")

        if (self.num_gt_per_class is None):
            raise ValueError("Number of ground truth boxes per class not available. You must run \
            `get_num_gt_per_class()` before you call this method.")

        cumulative_precisions = []
        cumulative_recalls = []

        # Iterate over all classes.
        for class_id in range(self.n_classes):
            logger.debug("Computing precisions and recalls, class {}/{}".format(class_id + 1,
                                                                                self.n_classes))

            tp = self.cumulative_true_positives[class_id]
            fp = self.cumulative_false_positives[class_id]

            # 1D array with shape `(num_predictions,)`
            cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
            # 1D array with shape `(num_predictions,)`
            cumulative_recall = tp / self.num_gt_per_class[class_id]

            cumulative_precisions.append(cumulative_precision)
            cumulative_recalls.append(cumulative_recall)

        self.cumulative_precisions = cumulative_precisions
        self.cumulative_recalls = cumulative_recalls

        if ret:
            return cumulative_precisions, cumulative_recalls
        return None

    def compute_average_precisions(self, mode='sample', num_recall_points=11,
                                   verbose=True, ret=False):
        '''
        Computes the average precision for each class.

        Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
        and post-2010 (integration) algorithm versions.

        Note that `compute_precision_recall()` must be called before calling this method.

        Arguments:
            mode (str, optional): Can be either 'sample' or 'integrate'. In the case of 'sample',
                the average precision will be computed according to the Pascal VOC formula that was
                used up until VOC 2009, where the precision will be sampled for `num_recall_points`
                recall values. In the case of 'integrate', the average precision will be computed
                according to the Pascal VOC formula that was used from VOC 2010 onward, where the
                average precision will be computed by numerically integrating over the whole
                preciscion-recall curve instead of sampling individual points from it. 'integrate'
                mode is basically just the limit case of 'sample' mode as the number of sample
                points increases. For details, see the references below.
            num_recall_points (int, optional): Only relevant if mode is 'sample'. The number of
                points to sample from the precision-recall-curve to compute the average precisions.
                In other words, this is the number of equidistant recall values for which the
                resulting precision will be computed. 11 points is the value used in the official
                Pascal VOC pre-2010 detection evaluation algorithm.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the average precisions.

        Returns:
            None by default. Optionally, a list containing average precision for each class.

        References:
            http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap
        '''

        if (self.cumulative_precisions is None) or (self.cumulative_recalls is None):
            raise ValueError("Precisions and recalls not available. You must run \
            `compute_precision_recall()` before you call this method.")

        if not (mode in {'sample', 'integrate'}):
            raise ValueError("`mode` can be either 'sample' or 'integrate', but received '{}'"
                             .format(mode))

        average_precisions = []

        # Iterate over all classes.
        for class_id in range(self.n_classes):

            logger.debug("Computing average precision, class {}/{}".format(class_id + 1,
                                                                           self.n_classes))
            cumulative_precision = self.cumulative_precisions[class_id]
            cumulative_recall = self.cumulative_recalls[class_id]
            average_precision = 0.0

            if mode == 'sample':

                for t in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):

                    cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]

                    if cum_prec_recall_greater_t.size == 0:
                        precision = 0.0
                    else:
                        precision = np.amax(cum_prec_recall_greater_t)

                    average_precision += precision

                average_precision /= num_recall_points

            elif mode == 'integrate':

                # We will compute the precision at all unique recall values.
                unique_recalls, unique_recall_indices, _ \
                    = np.unique(cumulative_recall, return_index=True,
                                return_counts=True)

                # Store the maximal precision for each recall value and the absolute difference
                # between any two unique recal values in the lists below. The products of these
                # two nummbers constitute the rectangular areas whose sum will be our numerical
                # integral.
                maximal_precisions = np.zeros(unique_recalls.shape, unique_recalls.dtype)
                recall_deltas = np.zeros(unique_recalls.shape, unique_recalls.dtype)

                # Iterate over all unique recall values in reverse order. This saves a lot of
                # computation: For each unique recall value `r`, we want to get the maximal
                # precision value obtained for any recall value `r* >= r`. Once we know the maximal
                # precision for the last `k` recall values after a given iteration, then in the next
                # iteration, in order compute the maximal precisions for the last `l > k` recall
                # values, we only need to compute the maximal precision for `l - k` recall values
                # and then take the maximum between that and the previously computed maximum instead
                # of computing the maximum over all `l` values. We skip the very last recall value,
                # since the precision after between the last recall value recall 1.0 is defined to
                # be zero.
                for i in range(len(unique_recalls)-2, -1, -1):
                    begin = unique_recall_indices[i]
                    end = unique_recall_indices[i + 1]
                    # When computing the maximal precisions, use the maximum of the previous
                    # iteration to avoid unnecessary repeated computation over the same precision
                    # values. The maximal precisions are the heights of the rectangle areas of our
                    # integral under the precision-recall curve.
                    maximal_precisions[i] = np.maximum(np.amax(cumulative_precision[begin:end]),
                                                       maximal_precisions[i + 1])
                    # The differences between two adjacent recall values are the widths of our
                    # rectangle areas.
                    recall_deltas[i] = unique_recalls[i + 1] - \
                        unique_recalls[i]

                average_precision = np.sum(maximal_precisions * recall_deltas)

            average_precisions.append(average_precision)

        self.average_precisions = [val if not math.isnan(val) else 0 for val in average_precisions]

        if ret:
            return average_precisions
        return None

    def compute_mean_average_precision(self, ret=True):
        '''
        Computes the mean average precision over all classes.

        Note that `compute_average_precisions()` must be called before calling this method.

        Arguments:
            ret (bool, optional): If `True`, returns the mean average precision.

        Returns:
            A float, the mean average precision, by default. Optionally, None.
        '''

        if self.average_precisions is None:
            raise ValueError("Average precisions not available. You must run \
            `compute_average_precisions()` before you call this method.")

        # The first element is for the background class, so skip it.
        mean_average_precision = np.average(self.average_precisions)
        self.mean_average_precision = mean_average_precision

        if ret:
            return mean_average_precision
        return None
