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

"""Command line interface for converting COCO json to VOC images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import sys

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from skimage.measure import label, regionprops
import tensorflow as tf
import nvidia_tao_tf1.cv.common.logging.logging as status_logging


logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    """Build command line parser for dataset_convert."""

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='dataset_converter',
            description='Convert COCO json to VOC images.'
        )
    parser.add_argument(
        '-f',
        '--coco_file',
        required=True,
        help='Path to COCO json file.')
    parser.add_argument(
        '-n',
        '--num_files',
        type=int,
        default=None,
        required=False,
        help='Number of images to convert from COCO json to VOC.'
             'These will be first N COCO records ')
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help="Path to the results directory where the VOC images are saved."
    )
    return parser


def anns_to_seg(anns, coco_instance, category_ids, s_logger, skipped_annotations,
                log_list):
    """
    Converts COCO-format annotations of a given image to a PASCAL-VOC segmentation style label.

    Args:
        anns (dict): COCO annotations as returned by 'coco.loadAnns'
        coco_instance (class): coco class instance 
        category_ids (List): label ids for different classes
        s_logger (class): logger class
        skipped_annotations (int): Total number of skipped annotations
        log_list (list): List of logging info
    Returns:
        Three 2D numpy arrays where the value of each pixel is the class id,
        instance number, and instance id.
    """
    image_details = coco_instance.loadImgs(anns[0]['image_id'])[0]
    h = image_details['height']
    w = image_details['width']
    class_seg = np.zeros((h, w))
    instance_seg = np.zeros((h, w))
    id_seg = np.zeros((h, w))
    masks, anns, skipped_annotations, log_list = anns_to_mask(anns, h, w, category_ids,
                                                              s_logger,
                                                              skipped_annotations,
                                                              log_list)
    for i, mask in enumerate(masks):
        class_seg = np.where(class_seg > 0, class_seg, mask*anns[i]['category_id'])
        instance_seg = np.where(instance_seg > 0, instance_seg, mask*(i+1))
        id_seg = np.where(id_seg > 0, id_seg, mask * anns[i]['id'])

    return class_seg, image_details["file_name"], skipped_annotations, log_list


def ann_to_RLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.

    Args:
        ann (Dict): Dictionary with annotation details
        h (int): height of input image
        w (int): width of input image
    Returns: 
        Binary mask (numpy 2D array)
    """

    image_id = ann["image_id"]
    assert('segmentation' in ann.keys()), "Segmentation field is absent in the" \
        "COCO json file for image id: {}.".format(image_id)
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif 'counts' in segm.keys():
        if type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            rle = ann['segmentation']
    else:
        raise ValueError('Please check the segmentation format.')

    return rle


def anns_to_mask(anns, h, w, category_ids, s_logger, skipped_annotations, log_list):
    """
    Convert annotations which can be polygons, uncompressed RLE, or RLE to binary masks.

    Returns:
        masks(list): A list of binary masks (each a numpy 2D array) of all the annotations in anns
        anns(list): List of annotations
    """

    masks = []
    for ann in anns:
        ann_error = {}
        if len(ann) > 0:
            if not ('image_id' in ann.keys()):
                logging.warning("image_id field is absent in the COCO json file.")
                s_logger.write(
                    message="image_id field is absent in the COCO json file.",
                    status_level=status_logging.Verbosity.WARNING)
                skipped_annotations += 1
                ann_error["error"] = "image_id field is absent in the COCO json file."
                ann_error["image_id"] = ann["image_id"]
                ann_error["annotation_id"] = ann["id"]
                log_list.append(ann_error)
                continue
            image_id = ann["image_id"]
            if not ('segmentation' in ann.keys()):
                logging.warning("Segmentation field is absent in the COCO"
                                "json file for image id: {}.".format(image_id))
                s_logger.write(
                    message="Segmentation field is absent in the COCO"
                    "json file for image id: {}.".format(image_id),
                    status_level=status_logging.Verbosity.WARNING)
                ann_error["error"] = "Segmentation field is absent in the COCO file."
                ann_error["image_id"] = ann["image_id"]
                ann_error["annotation_id"] = ann["id"]
                log_list.append(ann_error)
                skipped_annotations += 1
                continue
            # Check if the assigned category id is in the accepted list of category ids
            if not (ann["category_id"] in category_ids):
                logging.warning("Skipping annotation_id:{} in image_id:{}, as the category_id:{}"
                                "is not in supported category_ids:{}".format(ann["id"],
                                                                             ann["image_id"],
                                                                             ann["category_id"],
                                                                             category_ids))
                s_logger.write(
                    message="Skipping annotation_id:{} in image_id:{}, as the category_id:{}"
                    "is not in supported category_ids:{}".format(ann["id"],
                                                                 ann["image_id"],
                                                                 ann["category_id"],
                                                                 category_ids),
                    status_level=status_logging.Verbosity.WARNING)
                ann_error["error"] = "The category id provided is not in supported" \
                    "category id's: {}.".format(category_ids)
                ann_error["image_id"] = ann["image_id"]
                ann_error["category_id"] = ann["category_id"]
                ann_error["annotation_id"] = ann["id"]
                log_list.append(ann_error)
                skipped_annotations += 1
                continue

            rle = ann_to_RLE(ann, h, w)
            m = maskUtils.decode(rle)
            label_tmp = label(m)
            props = regionprops(label_tmp)
            for prop in props:
                # Get the tightest bounding box of the binary mask
                x1, y1, x2, y2 = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
                # Check the boundary conditions for the segmentation tight box
                if not (x1 < x2 <= w and y1 < y2 <= h):
                    logging.warning("Skipping annotation_id:{} in image_id:{},"
                                    "as the segmentation map of "
                                    "is out of bounds or faulty.".format(ann["image_id"],
                                                                         ann["id"]))
                    s_logger.write(
                        message="Skipping annotation_id:{} in image_id:{},"
                        "as the segmentation map of "
                        "is out of bounds or faulty.".format(ann["image_id"],
                                                             ann["id"]),
                        status_level=status_logging.Verbosity.WARNING)
                    ann_error["error"] = "The segmentation map is out of bounds or faulty."
                    ann_error["image_id"] = ann["image_id"]
                    ann_error["annotation_id"] = ann["id"]
                    log_list.append(ann_error)
                    skipped_annotations += 1
                    continue

            masks.append(m)

    return masks, anns, skipped_annotations, log_list


def coco2voc(anns_file, target_folder, n=None, s_logger=None):
    """Function to convert COCO json file to VOC images.
    Args:
        anns_file (str):
        target_folder (str):
        n (int):
        s_logger (logger class):
    
    """

    skipped_annotations, log_list = 0, []
    skipped_images = 0
    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs
    super_categories = coco_instance.cats
    category_ids = []
    for sc in super_categories:
        category_ids.append(sc)
    if n is None:
        n = len(coco_imgs)
    else:
        if not isinstance(n, int):
            s_logger.write(message="N must be int.", status_level=status_logging.Status.FAILURE)
            raise TypeError("N must be set as an int.")
        if n <= 0:
            s_logger.write(message="N must be greater than 0.",
                           status_level=status_logging.Status.FAILURE)
            raise ValueError("N must be greater than 0.")
        if n > len(coco_imgs):
            s_logger.write(
                message="N must be less than or equal to total number of images"
                "in the COCO json file."
                "Setting the N to total number of images in the coco json.",
                status_level=status_logging.Verbosity.WARNING)
        n = min(n, len(coco_imgs))

    logger.info("Number of images that are going to be converted {}".format(n))
    s_logger.write(message="Number of images that are going to be converted {}".format(n))
    # Some images may not have coco object, hence we need to account
    # for that to count total images saved. So counter to count the total
    # number of images actually saved.
    img_cntr = 0
    for _, img in enumerate(coco_imgs):
        img_error = {}
        anns_ids = coco_instance.getAnnIds(img)
        anns = coco_instance.loadAnns(anns_ids)
        if not anns:
            logging.warning("Skipping image {} that does not have"
                            " coco annotation".format(img))
            s_logger.write(
                message="Skipping image {} that does not have"
                        " coco annotation".format(img),
                status_level=status_logging.Verbosity.WARNING)
            skipped_images += 1
            img_error["error"] = "Image does not have annotation field defined."
            img_error["image_id"] = img
            log_list.append(img_error)
            continue
        class_seg, fn, skipped_annotations, log_list = anns_to_seg(
            anns, coco_instance, category_ids, s_logger, skipped_annotations,
            log_list)
        img_name = fn.split("/")[-1]
        img_name = img_name.split(".")[0]
        save_img_name = os.path.join(target_folder, img_name+".png")
        img_cntr += 1
        Image.fromarray(class_seg).convert("L").save(save_img_name)
        if img_cntr >= n:
            break

    # Logging the buggy anotations and images
    logging.info("The total number of skipped annotations are {}".format(skipped_annotations))
    logging.info("The total number of skipped images are {}".format(skipped_images))

    log_file = os.path.join(target_folder, "skipped_annotations_log.json")
    try:
        with open(log_file, "w") as final:
            json.dump(log_list, final)
    finally:
        logging.info("The details of faulty annotations  and images that were skipped"
                     " are logged in {}".format(log_file))


def parse_command_line_args(cl_args=None):
    """Parse sys.argv arguments from commandline.

    Args:
        cl_args: List of command line arguments.

    Returns:
        args: list of parsed arguments.
    """
    parser = build_command_line_parser()
    args = parser.parse_args(cl_args)
    return args


def main(args=None):
    """
    Convert a COCO json to VOC dataset format.

    Args:
        args(list): list of arguments to be parsed if called from another module.
    """
    args = parse_command_line_args(cl_args=args)

    verbosity = 'INFO'
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=verbosity)

    # Defining the results directory.
    results_dir = args.results_dir
    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=logger.getEffectiveLevel(),
            append=False
        )
    )
    s_logger = status_logging.get_status_logger()
    s_logger.write(
        data=None,
        message="Starting Semantic Segmentation Dataset to VOC Convert.",
        status_level=status_logging.Status.STARTED
    )
    try:
        coco2voc(args.coco_file, args.results_dir, args.num_files, s_logger)
        s_logger.write(
            status_level=status_logging.Status.SUCCESS,
            message="Conversion finished successfully."
        )
    except Exception as e:
        s_logger.write(
            status_level=status_logging.Status.FAILURE,
            message="Conversion failed with following error: {}.".format(e)
        )
        raise e


if __name__ == '__main__':
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Dataset convert finished successfully."
        )
    except Exception as e:
        if type(e) == tf.errors.ResourceExhaustedError:
            logger = logging.getLogger(__name__)
            logger.error(
                "Ran out of GPU memory, please lower the batch size, use a smaller input "
                "resolution, or use a smaller backbone."
            )
            status_logging.get_status_logger().write(
                message="Ran out of GPU memory, please lower the batch size, use a smaller input "
                        "resolution, or use a smaller backbone.",
                verbosity_level=status_logging.Verbosity.INFO,
                status_level=status_logging.Status.FAILURE
            )
            sys.exit(1)
        else:
            # throw out the error as-is if they are not OOM error
            raise e
