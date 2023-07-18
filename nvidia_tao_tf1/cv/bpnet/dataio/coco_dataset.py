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

"""COCO pose estimation dataset."""

import json
import logging
import multiprocessing
import os
import cv2
import matplotlib
# Use Agg to avoid installing an additional backend package like python-tk.
# This call (mpl.use('Agg') needs to happen before importing pyplot.
matplotlib.use('Agg')  # noqa # pylint-disable: To stop static test complains.

import numpy as np
import pandas
from pycocotools.coco import COCO
# TODO: Use custom script for COCO eval to support different combination
from pycocotools.cocoeval import COCOeval
from scipy.spatial.distance import cdist
import tqdm

from nvidia_tao_tf1.cv.bpnet.inferencer.bpnet_inferencer import BpNetInferencer
from nvidia_tao_tf1.cv.bpnet.utils import dataio_utils

logger = logging.getLogger(__name__)


class COCODataset(object):
    """COCO Dataset helper class."""

    def __init__(self,
                 dataset_spec,
                 parse_annotations=False):
        """Init function.

        Args:
            dataset_spec (dict): Specifications and parameters to be used for
                the dataset export
        """
        self.dataset_name = dataset_spec["dataset"]
        self.pose_config = self._get_category(dataset_spec, 'person')
        self.num_joints = self.pose_config["num_joints"]
        self.parts = self.pose_config["keypoints"]
        self.parts2idx = dict(zip(self.parts, range(self.num_joints)))
        assert self.num_joints == len(self.parts), "Assertion error: num_joints and \
            the number of keypoints are not matching! Please check dataset_spec config"

        self.root_dir = dataset_spec['root_directory_path']
        self.train_anno_path = os.path.join(
            self.root_dir, dataset_spec["train_data"]['annotation_root_dir_path'])
        self.test_anno_path = os.path.join(
            self.root_dir, dataset_spec["test_data"]['annotation_root_dir_path'])
        self.train_images_root_dir_path = dataset_spec["train_data"]["images_root_dir_path"]
        self.train_masks_root_dir_path = dataset_spec["train_data"]["mask_root_dir_path"]
        self.test_images_root_dir_path = dataset_spec["test_data"]["images_root_dir_path"]
        self.test_masks_root_dir_path = dataset_spec["test_data"]["mask_root_dir_path"]

        # Data filtering parameters
        self.min_acceptable_kpts = dataset_spec["data_filtering_params"]["min_acceptable_kpts"]
        self.min_acceptable_width = dataset_spec["data_filtering_params"]["min_acceptable_width"]
        self.min_acceptable_height = dataset_spec["data_filtering_params"]["min_acceptable_height"]
        self.min_acceptable_area = self.min_acceptable_width * self.min_acceptable_height
        self.min_acceptable_interperson_dist_ratio = \
            dataset_spec["data_filtering_params"]["min_acceptable_interperson_dist_ratio"]

        # Load train and test data
        self.train_data, self.train_images, self.train_coco, self.train_image_ids = \
            self.load_dataset(self.train_anno_path, parse_annotations)
        self.test_data, self.test_images, self.test_coco, self.test_image_ids = \
            self.load_dataset(self.test_anno_path, parse_annotations)

    def load_dataset(self, annotation_path, parse_annotations):
        """Function to load the dataset.

        Args:
            annotation_path (str): Path to the annotations json
            parse_annotations (bool): If enabled, it would parse
                through the annotations and extract individual annos.

        Returns:
            data (dict): Dictionary containing parsed image and anno info
            images (dict): Dictionary containing image info
            coco (COCO): Object of type COCO initialized with the anno file
            image_ids (list): List of image ids in the annotations
        """
        coco = COCO(annotation_path)
        image_ids = list(coco.imgs.keys())

        images = []
        for _, image_id in enumerate(image_ids):
            data_point = {}
            data_point['image_id'] = image_id
            data_point['image_meta'] = coco.imgs[image_id]
            data_point['full_image_path'] = data_point['image_meta']['file_name']
            # append only image related information
            images.append(data_point)

        data = []
        # Parse annotations of all the images in coco dataset
        if parse_annotations:
            for _, image_id in enumerate(tqdm.tqdm(image_ids)):
                data_point = {}
                data_point['image_id'] = image_id
                data_point['image_meta'] = coco.imgs[image_id]
                annotation_ids = coco.getAnnIds(imgIds=image_id)
                image_annotation = coco.loadAnns(annotation_ids)
                all_persons, main_persons = self._parse_annotation(image_annotation)
                # If no keypoint labeling in this image, skip it
                if not len(all_persons):
                    continue
                data_point['all_persons'] = all_persons
                data_point['main_persons'] = main_persons

                data.append(data_point)
        return data, images, coco, image_ids

    @staticmethod
    def _get_category(data, cat_name):
        """Get the configuration corresponding to the given category name.

        TODO: Move to utils

        Args:
            cat_name (str): category name

        Return:
            (dict): meta information about the category
        """
        return [c for c in data['categories'] if c['name'] == cat_name][0]

    @staticmethod
    def get_image_name(coco, image_id):
        """Get the image path.

        Args:
            coco (COCO): Object of type COCO
            image_id (int): id of the image to retrieve filepath

        Returns:
            (str): filepath
        """
        return coco.imgs[image_id]['file_name']

    def _parse_annotation(self, img_anns):
        """Parse the given annotations in the image and compile the info.

        Args:
            img_anns (list): list of annotations associated with the current image.

        Returns:
            all_persons (list): list consisting of all the annotated people in the image
            main_persons (list): filtered list of annotated people in the image
                based on certain criteria.
        """

        num_people = len(img_anns)
        all_persons = []

        for p in range(num_people):

            pers = dict()
            person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,
                             img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]

            pers["objpos"] = person_center
            pers["bbox"] = img_anns[p]["bbox"]
            pers["iscrowd"] = img_anns[p]["iscrowd"]
            pers["segment_area"] = img_anns[p]["area"]
            pers["segmentation"] = img_anns[p]["segmentation"]
            pers["num_keypoints"] = img_anns[p]["num_keypoints"]

            kpts = img_anns[p]["keypoints"]

            pers["joint"] = np.zeros((self.num_joints, 3))
            for part in range(self.num_joints):
                # The convention for visbility flags used in COCO is as follows:
                #   0: not labeled (in which case x=y=0)
                #   1: labeled but not visible,
                #   2: labeled and visible.
                pers["joint"][part, 0] = kpts[part * 3]
                pers["joint"][part, 1] = kpts[part * 3 + 1]
                pers["joint"][part, 2] = kpts[part * 3 + 2]
            pers["scale_provided"] = img_anns[p]["bbox"][3]

            all_persons.append(pers)

        main_persons = []
        prev_center = []

        # Filter out the "main people" based on following creteria
        #   1. Number of keypoints less than `min_acceptable_kpts`.
        #   2. Pixel Area is less than `min_acceptable_area`
        # This is used later during training to augment the data around the main persons
        for pers in all_persons:

            # Filter the persons with few visible/annotated keypoints parts or
            # little occupied area (relating to the scale of the person).
            if pers["num_keypoints"] < self.min_acceptable_kpts:
                continue

            if pers["segment_area"] < self.min_acceptable_area:
                continue

            person_center = pers["objpos"]

            # Filter the persons very close to the existing list of persons in the
            # `main_persons` already.
            flag = 0
            for pc in prev_center:
                a = np.expand_dims(pc[:2], axis=0)
                b = np.expand_dims(person_center, axis=0)
                dist = cdist(a, b)[0]
                if dist < pc[2] * self.min_acceptable_interperson_dist_ratio:
                    flag = 1
                    continue

            if flag == 1:
                continue

            main_persons.append(pers)
            prev_center.append(
                np.append(person_center, max(img_anns[p]["bbox"][2], img_anns[p]["bbox"][3]))
            )

        return all_persons, main_persons

    def process_segmentation_masks(self, data, mask_root_dir):
        """Generate and save binary mask to disk.

        Args:
            data (list): List of groundtruth annotations
                with corresponding meta data.
            mask_root_dir (str): Root directory path to save the
                generated binary masks.
        """
        pool = multiprocessing.Pool()
        total_samples = len(data)
        for idx, _ in enumerate(
            pool.imap(
                COCODataset._pool_process_segmentation_mask,
                [(data_point, mask_root_dir) for data_point in data])):
            if idx % 1000 == 0:
                logger.info('Mask Generation: {}/{}'.format(idx, total_samples))

    @staticmethod
    def _pool_process_segmentation_mask(args):
        """Wrapper for process_single_segmentation_mask for multiprocessing.

        Args:
            args (list): List of all args that needs to be forwarded
                to process_single_segmentation_mask fn.
        """
        COCODataset.process_single_segmentation_mask(*args)

    @staticmethod
    def process_single_segmentation_mask(data_point, mask_root_dir):
        """Generate and save binary mask to disk.

        Args:
            data_point (dict): Dictionary containing the groundtruth
                annotation for one image with corresponding meta data.
            mask_root_dir (str): Root directory path to save the
                generated binary masks.
        """
        image_meta = data_point['image_meta']
        height = image_meta['height']
        width = image_meta['width']
        annotations = data_point['all_persons']

        mask_out, _, _, _ = COCODataset.get_segmentation_masks(
            height, width, annotations)

        mask_out = mask_out.astype(np.uint8)
        mask_out *= 255

        # TODO: How to handle IX dataset? Maybe add keys for replacement
        # of path? Provide mask path as pattern?
        filename = image_meta['file_name']
        mask_path = os.path.join(mask_root_dir, filename)
        # Create directory if it doesn't already exist
        if not os.path.exists(os.path.dirname(mask_path)):
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, mask_out)

    @staticmethod
    def get_segmentation_masks(height, width, annotations):
        """Generate and save binary mask to disk.

        Args:
            height (int): Height of the image.
            width (int): Width of the image.
            annotations (dict): Dictionary containing the groundtruth
                annotation for one image.

        Returns:
            mask_out (np.ndarray): Binary mask of regions to mask
                out during training.
            crowd_region (np.ndarray): Binary mask of regions labeled
                as `crowd`.
            unlabeled_region (np.ndarray): Binary mask of regions without
                keypoint labeling for persons.
            labeled_region (np.ndarray): Binary mask of regions with
                keypoint labeling for persons.
        """
        labeled_region = np.zeros((height, width), dtype=np.uint8)
        unlabeled_region = np.zeros((height, width), dtype=np.uint8)
        crowd_region = np.zeros((height, width), dtype=np.uint8)
        mask_out = np.zeros((height, width), dtype=np.uint8)
        # Iterate through the annaotations
        for annotation in annotations:
            # Convert the segemtation to RLE to binary mask
            binary_mask = dataio_utils.annotation_to_mask(
                annotation['segmentation'], height, width)

            if annotation["iscrowd"]:
                crowd_region = np.bitwise_or(crowd_region, binary_mask)
            elif annotation["num_keypoints"] <= 0:
                unlabeled_region = np.bitwise_or(unlabeled_region, binary_mask)
            else:
                labeled_region = np.bitwise_or(labeled_region, binary_mask)

        # Remove the overlap region from `crowd_region` to ensure the
        # labeled person is not masked out.
        # TODO: Should we be doing the same for `unlabeled_region_mask`?
        overlap_region = np.bitwise_and(labeled_region, crowd_region)
        mask_out = crowd_region - overlap_region
        # Union of crowd region and unlabeled region should be masked out
        mask_out = np.bitwise_or(mask_out, unlabeled_region)
        # Invert the mask to ensure valid regions are 1s.
        mask_out = np.logical_not(mask_out)

        return mask_out, crowd_region, unlabeled_region, labeled_region

    # TODO: Move to utils?
    @staticmethod
    def convert_kpts_format(kpts, target_parts2idx, source_parts2idx):
        """Convert keypoints from source to target format.

        Args:
            kpts (np.ndarray): source keypoints that needs to be converted
                to target ordering.
            target_parts2idx (dict): Dict with mapping from target keypoint
                names to keypoint index
            source_parts2idx (dict): Dict with mapping from source keypoint
                names to keypoint index

        Returns:
            converted_kpts (np.ndarray): converted keypoints
        """

        converted_kpts = np.zeros(
            (kpts.shape[0], len(target_parts2idx.keys()), kpts.shape[-1]), dtype=np.float32)
        for part in source_parts2idx:
            source_part_id = source_parts2idx[part]
            if part in target_parts2idx:
                target_part_id = target_parts2idx[part]
                converted_kpts[:, target_part_id, :] = kpts[:, source_part_id, :]
        return converted_kpts

    def dump_detection_in_coco_format(self, results, detections_parts2idx, results_dir):
        """Function to dump results as expected by COCO AP evaluation.

        Args:
            results (dict): Keypoint results in BpNetInferencer format.
            detections_parts2idx (dict): Meta data about the pose configuration
            results_dir (dict): Directory to save the formatted results.

        Returns:
            output_path (str): Path to the final saved results json.
        """
        detections = []
        for result_dict in results['images']:
            keypoints_list = result_dict['keypoints']
            scores = result_dict['scores']
            if len(keypoints_list) == 0:
                continue
            converted_kpts_list = self.convert_kpts_format(
                    np.array(keypoints_list), self.parts2idx, detections_parts2idx).tolist()
            for keypoints, score in zip(converted_kpts_list, scores):
                format_keypoint_list = []
                for x, y in keypoints:
                    # Append visibility index (0 if kpt was not detected)
                    for fkpts in [int(x), int(y), 1 if x > 0 or y > 0 else 0]:
                        format_keypoint_list.append(fkpts)

                detections.append({
                    "image_id": result_dict['image_id'],
                    "category_id": 1,
                    "keypoints": format_keypoint_list,
                    # TODO: derive score from the model
                    "score": score,
                })
        # Dump the formatted detection dict for COCO evaluation
        output_path = os.path.join(results_dir, 'detections_coco_format.json')
        if os.path.exists(output_path):
            logger.warning(
                "File already exists: {}. Overwritting into same file.".format(output_path))
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)
        return output_path

    @staticmethod
    def dump_coco_results_summary(results, results_dir):
        """Function to dump evaluation results summary into a csv.

        Args:
            results (COCOeval): evaluation results
            results_dir (str): Directory to save the formatted results.
        """
        columns = [
            'metric', 'IoU', 'area', 'maxDets', 'score'
        ]
        stats = results.stats
        num_rows = len(stats)
        df = pandas.DataFrame(columns=columns, index=np.arange(0, num_rows))
        df.loc[0:4, 'metric'] = 'AP'
        df.loc[5:10, 'metric'] = 'AR'
        df.loc[[0, 3, 4, 5, 8, 9], 'IoU'] = '0.50:0.95'
        df.loc[[1, 6], 'IoU'] = '0.50'
        df.loc[[2, 7], 'IoU'] = '0.75'
        df.loc[[0, 1, 2, 5, 6, 7], 'area'] = 'all'
        df.loc[[3, 8], 'area'] = 'medium'
        df.loc[[4, 9], 'area'] = 'large'
        df.loc[:, 'maxDets'] = 20
        for i in range(num_rows):
            df.loc[i, 'score'] = stats[i]
        # dump as csv
        results_file = os.path.join(results_dir, 'results.csv')
        df.to_csv(results_file)

    def infer(self,
              model_path,
              inference_spec,
              experiment_spec,
              results_dir,
              key=None,
              visualize=False):
        """Run inference on the validation set and save results.

        Args:
            model_path (str): Path to model.
            inference_spec (dict): Inference specification.
            experiment_spec (dict): Training experiment specification.
            results_dir (str): Directory to save results.
            visualize (bool): Option to enable visualization

        Returns:
            output_path (str): Path to the formatted detections json
        """
        # init BpNet Inferencer
        inferencer = BpNetInferencer(
            model_path,
            inference_spec,
            experiment_spec,
            key=key
        )
        # Run inference
        data = dict(images=self.test_images)
        image_root_path = os.path.join(self.root_dir, self.test_images_root_dir_path)
        results = inferencer.run(
            data,
            results_dir=results_dir,
            image_root_path=image_root_path,
            visualize=visualize)
        # Also save in format expected by COCO eval
        output_path = self.dump_detection_in_coco_format(
            results,
            inferencer.bpnet_pose_config.parts2idx,
            results_dir)
        return output_path

    @staticmethod
    def evaluate(coco_gt, detections_path, results_dir):
        """Run evaluation on the detections and save results.

        Args:
            coco_gt (COCO): COCO object initialized with annotations
                of the test/val set.
            detections_path (str): Path to the formatted detections json
            results_dir (str): Directory to save results.

        Returns:
            results (COCOeval): evaluation results
        """
        annotation_type = 'keypoints'
        print('Running test for {} results.'.format(annotation_type))

        coco_dt = coco_gt.loadRes(detections_path)

        results = COCOeval(coco_gt, coco_dt, annotation_type)
        results.evaluate()
        results.accumulate()
        results.summarize()
        # Save results
        COCODataset.dump_coco_results_summary(results, results_dir)
        return results

    def visualize(self):
        """Visualize annotations."""
        pass
