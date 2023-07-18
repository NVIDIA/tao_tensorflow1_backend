# Copyright (c) 2017 - 2019, NVIDIA CORPORATION.  All rights reserved.
"""Post processing handler for TLT DetectNet_v2 models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import logging
from multiprocessing import Pool
import operator
import os
from time import time

from addict import Dict

import numpy as np

from PIL import ImageDraw
from six.moves import range
import wandb

from nvidia_tao_tf1.cv.common.mlops.wandb import is_wandb_initialized
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.utilities import cluster_bboxes
from nvidia_tao_tf1.cv.detectnet_v2.utilities.constants import criterion, scales

logger = logging.getLogger(__name__)

CLUSTERING_ALGORITHM = {0: "dbscan",
                        1: "nms",
                        2: "hybrid"}


@contextmanager
def pool_context(*args, **kwargs):
    """Simple wrapper to get pool context with close function."""
    pool = Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.terminate()


def render_single_image_output(input_tuple, target_classes,
                               image_overlay, save_kitti,
                               output_image_root, output_label_root,
                               class_wise_detections,
                               linewidth, resized_size,
                               confidence_model,
                               box_color,
                               output_map,
                               frame_height,
                               frame_width):
    """Rendering for a single image.

    Args:
        input_tuple (tuple): Tuple of rendering inputs.
        target_classes (list): List of classes to be post-processed.
        image_overlay (bool): Flag to set images to overlay.
        save_kitti (bool): Flag to dump kitti label files.
        output_image_root (str): Path to the directory where rendered images are to be saved.
        output_label_root (str): Path to the directory where kitti labels are to be saved.
        class_wise_detections (dict): Dictionary of class-wise detections.
        linewidth (int): Thickness of bbox pixels.
        resized_size (tuple): Size of resized images.
        confidence_model (dict): Dictionary of confidence models per class.
        box_color (dict): Dictionary of rendering colors for boxes.
        output_map (dict): Dictionary of output map classes.
        frame_height (int): Inference frame height.
        frame_width (int): Inference frame width.

    Returns:
        No explicit returns.
    """
    idx = input_tuple[0]
    pil_input = input_tuple[1]
    image_name = input_tuple[2]
    scaling_factor = tuple(map(operator.truediv, pil_input.size, resized_size))
    processed_image = deepcopy(pil_input)
    label_list = []
    image_file = os.path.join(output_image_root, image_name)
    label_file = os.path.join(output_label_root, os.path.splitext(image_name)[0] + '.txt')
    draw = ImageDraw.Draw(processed_image)

    for keys in target_classes:
        key = str(keys)
        cluster_key = key
        if key not in list(output_map.keys()):
            cluster_key = "default"

        bbox_list, confidence_list = _get_bbox_and_confs(class_wise_detections[key][idx],
                                                         scaling_factor,
                                                         cluster_key,
                                                         confidence_model,
                                                         frame_height,
                                                         frame_width)
        num_boxes = len(bbox_list)
        if num_boxes != 0:
            for box in range(len(bbox_list)):
                edgecolor = box_color[cluster_key]
                x1 = float(bbox_list[box][0])
                y1 = float(bbox_list[box][1])
                x2 = float(bbox_list[box][2])
                y2 = float(bbox_list[box][3])
                if cluster_key == "default":
                    class_name = key
                else:
                    class_name = output_map[key] \
                        if key in list(output_map.keys()) else key
                if image_overlay:
                    if (x2 - x1) >= 0.0 and (y2 - y1) >= 0.0:
                        draw.rectangle(((x1, y1), (x2, y2)), outline=edgecolor)
                    for i in range(linewidth):
                        draw.rectangle(((x1 - i, y1 - i), (x2 + i, y2 + i)), outline=edgecolor)
                    draw.text((x1, y1), f"{class_name}:{confidence_list[box]:.3f}")
                if save_kitti:
                    label_tail = " 0.00 0.00 0.00 "\
                                 "0.00 0.00 0.00 0.00 {:.3f}\n".format(confidence_list[box])

                    label_head = class_name + " 0.00 0 0.00 "
                    bbox_string = "{:.3f} {:.3f} {:.3f} {:.3f}".format(x1, y1,
                                                                       x2, y2)
                    label_string = label_head + bbox_string + label_tail
                    label_list.append(label_string)

    if image_overlay:
        processed_image.save(image_file)
        if is_wandb_initialized():
            wandb_image = wandb.Image(processed_image, os.path.basename(os.path.splitext(image_file)[0]))
            wandb.log({"Rendered images": wandb_image})


    if save_kitti:
        with open(label_file, 'w') as f:
            if label_list:
                for line in label_list:
                    f.write(line)
        f.closed


def _get_bbox_and_confs(classwise_detections, scaling_factor,
                        key, confidence_model, frame_height,
                        frame_width):
    """Simple function to get bbox and confidence formatted list."""
    bbox_list = []
    confidence_list = []
    for i in range(len(classwise_detections)):
        bbox_object = classwise_detections[i]
        coords_scaled = _scale_bbox(bbox_object.bbox, scaling_factor,
                                    frame_height, frame_width)
        if confidence_model[key] == 'mlp':
            confidence = bbox_object.confidence[0]
        else:
            confidence = bbox_object.confidence
        bbox_list.append(coords_scaled)
        confidence_list.append(confidence)
    return bbox_list, confidence_list


def _scale_bbox(bbox, scaling_factor, frame_height, frame_width):
    '''
    Scale bbox coordinates back to original image dimensions.

    Args:
        bbox (list): bbox coordinates ltrb
        scaling factor (float): input_image size/model inference size

    Returns:
        bbox_scaled (list): list of scaled coordinates
    '''
    # Clipping and clamping coordinates.
    x1 = min(max(0.0, float(bbox[0])), frame_width)
    y1 = min(max(0.0, float(bbox[1])), frame_height)
    x2 = max(min(float(bbox[2]), frame_width), x1)
    y2 = max(min(float(bbox[3]), frame_height), y1)

    # Rescaling center.
    hx, hy = x2 - x1, y2 - y1
    cx = x1 + hx/2
    cy = y1 + hy/2

    # Rescaling height, width
    nx, ny = cx * scaling_factor[0], cy * scaling_factor[1]
    nhx, nhy = hx * scaling_factor[0], hy * scaling_factor[1]

    # Final bbox coordinates.
    nx1, nx2 = nx - nhx/2, nx + nhx/2
    ny1, ny2 = ny - nhy/2, ny + nhy/2

    # Stacked coordinates.
    bbox_scaled = np.asarray([nx1, ny1, nx2, ny2])
    return bbox_scaled


class BboxHandler(object):
    """Class to handle bbox output from the inference script."""

    def __init__(self, spec=None, **kwargs):
        """Setting up Bbox handler."""
        self.spec = spec
        self.cluster_params = Dict()
        self.frame_height = kwargs.get('frame_height', 544)
        self.frame_width = kwargs.get('frame_width', 960)
        self.bbox_normalizer = kwargs.get('bbox_normalizer', 35)
        self.bbox = kwargs.get('bbox', 'ltrb')
        self.cluster_params = kwargs.get('cluster_params', None)
        self.classwise_cluster_params = kwargs.get("classwise_cluster_params", None)
        self.bbox_norm = (self.bbox_normalizer, )*2
        self.stride = kwargs.get("stride", 16)
        self.train_img_size = kwargs.get('train_img_size', None)
        self.save_kitti = kwargs.get('save_kitti', True)
        self.image_overlay = kwargs.get('image_overlay', True)
        self.extract_crops = kwargs.get('extract_crops', True)
        self.target_classes = kwargs.get('target_classes', None)
        self.bbox_offset = kwargs.get("bbox_offset", 0.5)
        self.clustering_criterion = kwargs.get("criterion", "IOU")
        self.postproc_classes = kwargs.get('postproc_classes', self.target_classes)
        confidence_threshold = {}
        nms_confidence_threshold = {}

        for key, value in list(self.classwise_cluster_params.items()):
            confidence_threshold[key] = value.clustering_config.dbscan_confidence_threshold
            if value.clustering_config.nms_confidence_threshold:
                nms_confidence_threshold[key] = value.clustering_config.nms_confidence_threshold

        self.state = Dict({
            'scales': scales,
            'display_classes': self.target_classes,
            'min_height': 0,
            'criterion': criterion,
            'confidence_th': {'car': 0.9, 'person': 0.1, 'truck': 0.1},
            'nms_confidence_th': {'car': 0.9, 'person': 0.1, 'truck': 0.1},
            'cluster_weights': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        })
        self.framework = kwargs.get("framework", "tlt")
        self.state.confidence_th = confidence_threshold
        self.state.nms_confidence_th = nms_confidence_threshold

    def bbox_preprocessing(self, input_cluster):
        """Function to perform inplace manipulation of prediction dicts before clustering.

        Args:
            input_cluster (Dict): prediction dictionary of output cov and bbox per class.

        Returns:
            input_cluster (Dict): shape manipulated prediction dictionary.
        """
        for classes in self.target_classes:
            input_cluster[classes]['bbox'] = self.abs_bbox_converter(input_cluster[classes]
                                                                     ['bbox'])
            # Stack predictions
            for keys in list(input_cluster[classes].keys()):
                if 'bbox' in keys:
                    input_cluster[classes][keys] = \
                        input_cluster[classes][keys][np.newaxis, :, :, :, :]
                    input_cluster[classes][keys] = \
                        np.asarray(input_cluster[classes][keys]).transpose((1, 2, 3, 4, 0))
                elif 'cov' in keys:
                    input_cluster[classes][keys] = input_cluster[classes][keys][np.newaxis,
                                                                                np.newaxis,
                                                                                :, :, :]
                    input_cluster[classes][keys] = \
                        np.asarray(input_cluster[classes][keys]).transpose((2, 1, 3, 4, 0))

        return input_cluster

    def abs_bbox_converter(self, bbox):
        '''Convert the raw grid cell corrdinates to image space coordinates.

        Args:
            bbox (np.array): BBox coordinates blob per batch with shape [n, 4, h, w].
        Returns:
            bbox (np.array): BBox coordinates reconstructed from grid cell based coordinates
              with the same dimensions.
        '''
        target_shape = bbox.shape[-2:]

        # Define grid cell centers
        gc_centers = [(np.arange(s) * self.stride + self.bbox_offset) for s in target_shape]
        gc_centers = [s / n for s, n in zip(gc_centers, self.bbox_norm)]

        # Mapping cluster output
        if self.bbox == 'arxy':
            assert not self.train_img_size, \
                "ARXY bbox format needs same train and inference image shapes."
            # reverse mapping of abs bbox to arxy
            area = (bbox[:, 0, :, :] / 10.) ** 2.
            width = np.sqrt(area * bbox[:, 1, :, :])
            height = np.sqrt(area / bbox[:, 1, :, :])
            cen_x = width * bbox[:, 2, :, :] + gc_centers[0][:, np.newaxis]
            cen_y = height * bbox[:, 3, :, :] + gc_centers[1]
            bbox[:, 0, :, :] = cen_x - width / 2.
            bbox[:, 1, :, :] = cen_y - height / 2.
            bbox[:, 2, :, :] = cen_x + width / 2.
            bbox[:, 3, :, :] = cen_y + height / 2.
            bbox[:, 0, :, :] *= self.bbox_norm[0]
            bbox[:, 1, :, :] *= self.bbox_norm[1]
            bbox[:, 2, :, :] *= self.bbox_norm[0]
            bbox[:, 3, :, :] *= self.bbox_norm[1]
        elif self.bbox == 'ltrb':
            # Convert relative LTRB bboxes to absolute bboxes inplace.
            # Input bbox in format (image, bbox_value,
            # grid_cell_x, grid_cell_y).
            # Ouput bboxes given in pixel coordinates in the source resolution.
            if not self.train_img_size:
                self.train_img_size = self.bbox_norm
            # Compute scalers that allow using different resolution in
            # inference and training
            scale_w = self.bbox_norm[0] / self.train_img_size[0]
            scale_h = self.bbox_norm[1] / self.train_img_size[1]
            bbox[:, 0, :, :] -= gc_centers[0][:, np.newaxis] * scale_w
            bbox[:, 1, :, :] -= gc_centers[1] * scale_h
            bbox[:, 2, :, :] += gc_centers[0][:, np.newaxis] * scale_w
            bbox[:, 3, :, :] += gc_centers[1] * scale_h
            bbox[:, 0, :, :] *= -self.train_img_size[0]
            bbox[:, 1, :, :] *= -self.train_img_size[1]
            bbox[:, 2, :, :] *= self.train_img_size[0]
            bbox[:, 3, :, :] *= self.train_img_size[1]
        return bbox

    def cluster_detections(self, preds):
        """
        Cluster detections and filter based on confidence.

        Also determines false positives and missed detections based on the
        clustered detections.

        Args:
            - spec: The experiment spec
            - preds: Raw predictions, a Dict of Dicts
            - state: The DetectNet_v2 viz state

        Returns:
            - classwise_detections (NamedTuple): DBSCan clustered detections.
        """
        # Cluster
        classwise_detections = Dict()
        clustering_time = 0.
        for object_type in preds:
            start_time = time()
            if object_type not in list(self.classwise_cluster_params.keys()):
                logger.info("Object type {} not defined in cluster file. Falling back to default"
                            "values".format(object_type))
                buffer_type = "default"
                if buffer_type not in list(self.classwise_cluster_params.keys()):
                    raise ValueError("If the class-wise cluster params for an object isn't "
                                     "there then please mention a default class.")
            else:
                buffer_type = object_type
            logger.debug("Clustering bboxes {}".format(buffer_type))
            classwise_params = self.classwise_cluster_params[buffer_type]
            clustering_config = classwise_params.clustering_config
            clustering_algorithm = CLUSTERING_ALGORITHM[clustering_config.clustering_algorithm]
            nms_iou_threshold = 0.3
            if clustering_config.nms_iou_threshold:
                nms_iou_threshold = clustering_config.nms_iou_threshold
            confidence_threshold = self.state.confidence_th.get(buffer_type, 0.1)
            nms_confidence_threshold = self.state.nms_confidence_th.get(buffer_type, 0.1)
            detections = cluster_bboxes(preds[object_type],
                                        criterion=self.clustering_criterion,
                                        eps=classwise_params.clustering_config.dbscan_eps + 1e-12,
                                        min_samples=clustering_config.dbscan_min_samples,
                                        min_weight=clustering_config.coverage_threshold,
                                        min_height=clustering_config.minimum_bounding_box_height,
                                        confidence_model=classwise_params.confidence_model,
                                        cluster_weights=self.state.cluster_weights,
                                        image_size=(self.frame_width, self.frame_height),
                                        framework=self.framework,
                                        confidence_threshold=confidence_threshold,
                                        clustering_algorithm=clustering_algorithm,
                                        nms_iou_threshold=nms_iou_threshold,
                                        nms_confidence_threshold=nms_confidence_threshold)

            clustering_time += (time() - start_time) / len(preds)
            classwise_detections[object_type] = detections

        return classwise_detections

    def render_outputs(self, _classwise_detections, pil_list,
                       output_image_root, output_label_root,
                       chunk_list, resized_size, linewidth=2):
        """Overlay primary detections on original image.

        Args:
            class_wise_detections (list): classwise detections outputs from network
              handler
            pil_input (PIL object): PIL object (image) on which detector was inferenced
            scaling factor (float): input/models image size ratio to reconstruct labels
              back to image coordinates
            output_image_root (str): Output image directory where the images are
              saved after rendering
            output_label_root (str): Path to the output directory where the labels
              are saved after rendering.
            image_name (str): Name of the current image.
            idx (int): batchwise inferencing image id in the batch
            linewidth (int): thickness of bbox lines in pixels

        Returns:
            processed_image (pil_object): Detections overlain pil object
        """
        if self.image_overlay:
            if not os.path.exists(output_image_root):
                os.makedirs(output_image_root)
        if self.save_kitti:
            if not os.path.exists(output_label_root):
                os.makedirs(output_label_root)

        if len(pil_list) != len(chunk_list):
            raise ValueError("Cannot render a chunk with unequal number of images and image_names.")

        # Setting up picklable arguments.
        input_tuples = [(i, pil_list[i], chunk_list[i]) for i in range(len(pil_list))]

        # Unpacking cluster params.
        box_color = {}
        output_map = {}
        confidence_model = {}
        for key in list(self.classwise_cluster_params.keys()):
            confidence_model[key] = None
            if self.classwise_cluster_params[key].confidence_model:
                confidence_model[key] = self.classwise_cluster_params[key].confidence_model
            output_map[key] = None
            if self.classwise_cluster_params[key].output_map:
                output_map[key] = self.classwise_cluster_params[key].output_map
            box_color[key] = (0, 255, 0)
            if self.classwise_cluster_params[key].bbox_color:
                color = self.classwise_cluster_params[key].bbox_color
                box_color[key] = (color.R, color.G, color.B)

        # Running rendering across mulitple threads
        with pool_context() as pool:
            pool.map(partial(render_single_image_output,
                             target_classes=list(self.postproc_classes),
                             image_overlay=self.image_overlay,
                             save_kitti=self.save_kitti,
                             output_image_root=output_image_root,
                             output_label_root=output_label_root,
                             class_wise_detections=_classwise_detections,
                             linewidth=linewidth,
                             resized_size=resized_size,
                             confidence_model=confidence_model,
                             box_color=box_color,
                             output_map=output_map,
                             frame_height=self.frame_height,
                             frame_width=self.frame_width), input_tuples)

    def extract_bboxes(self, class_wise_detections, pil_input, scaling_factor, idx=0):
        '''Extract sub images of primary detections from primary image.

        Args:
            class_wise_detections (list): classwise detections outputs from network
              handler.
            pil_input (Pillow object): PIL object for input image from which crops are extracted.
            scaling factor (float): input/models image size ratio to reconstruct labels
              back to image coordinates
            idx (int): batchwise inferencing image id in the batch

        Returns:
            crop_list (list): list of pil objects corresponding to crops of primary
                       detections
        '''
        crops = {}
        for keys in self.postproc_classes:
            key = str(keys)
            bbox_list = []
            for i in range(len(class_wise_detections[key][idx])):
                bbox_list.append(_scale_bbox(class_wise_detections[key][idx][i].bbox,
                                             scaling_factor,
                                             self.frame_height,
                                             self.frame_width))

            crop_list = []
            if bbox_list:
                for box in range(len(bbox_list)):
                    x1 = float(bbox_list[box][0])
                    y1 = float(bbox_list[box][1])
                    x2 = float(bbox_list[box][2])
                    y2 = float(bbox_list[box][3])
                    crop = pil_input.crop((x1, y1, x2, y2))
                    crop_list.append(crop)

            crops[key] = crop_list
        return crops
