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

"""Converts a COCO detection dataset to TFRecords."""


from __future__ import absolute_import
from __future__ import print_function
from collections import Counter
import logging
import os
from pycocotools.coco import COCO

import six
import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _bytes_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _float_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _int64_feature

from nvidia_tao_tf1.cv.detectnet_v2.dataio.dataset_converter_lib import DatasetConverter


logger = logging.getLogger(__name__)


class COCOConverter(DatasetConverter):
    """Converts a COCO detection dataset to TFRecords."""

    def __init__(self, root_directory_path, num_partitions, num_shards,
                 output_filename,
                 sample_modifier,
                 image_dir_names=None,
                 annotation_files=None,
                 use_dali=False,
                 class2idx=None):
        """Initialize the converter.

        Args:
            root_directory_path (string): Dataset directory path relative to data root.
            num_partitions (int): Number of partitions (folds).
            num_shards (list): Number of shards for each partition.
            output_filename (str): Path for the output file.
            sample_modifier(SampleModifier): An instance of sample modifier
                that does e.g. duplication and filtering of samples.
            image_dir_names (list): List of image directories for each partition.
            annotation_files (list): List of annotation files for each partition
        """
        super(COCOConverter, self).__init__(
            root_directory_path=root_directory_path,
            num_partitions=num_partitions,
            num_shards=num_shards,
            output_filename=output_filename,
            sample_modifier=sample_modifier)

        # COCO defaults.
        self.coco = []
        self.cat_idx = {}
        self.img_dir_names = image_dir_names
        self.annotation_files = annotation_files
        self.use_dali = use_dali
        self.class2idx = class2idx
        self.idx2class = None

    def _partition(self):
        """Load COCO annotations."""
        logger.debug("Generating partitions")
        s_logger = status_logging.get_status_logger()
        s_logger.write(message="Generating partitions")
        partitions = []
        cat_idx = {}
        for ann_file in self.annotation_files:
            ann_file = os.path.join(self.root_dir, ann_file)
            if not os.path.exists(ann_file):
                raise FileNotFoundError(f"Failed to load annotation from {ann_file}")
            logger.debug("Loadding annotations from {}".format(ann_file))
            c = COCO(ann_file)

            # Error checking on the annotation file
            if len(c.anns) == 0:
                raise ValueError(f"\"annotations\" field is missing in the JSON file {ann_file}")
            if len(c.imgs) == 0:
                raise ValueError(f"\"images\" field is missing in the JSON file {ann_file}")
            if len(c.cats) == 0:
                raise ValueError(f"\"categories\" field is missing in the JSON file {ann_file}")

            cats = c.loadCats(c.getCatIds())
            if len(cat_idx) and sorted(cat_idx.keys()) != sorted([cat['id'] for cat in cats]):
                raise ValueError("The categories in your partitions don't match. "
                                 "Please check your labels again")

            for cat in cats:
                # Remove any white spaces
                cat_idx[cat['id']] = cat['name'].replace(" ", "")

            self.coco.append(c)
            partitions.append(c.getImgIds())
        self.idx2class = cat_idx

        if self.class2idx is None:
            self.class2idx = {v: k for k, v in self.idx2class.items()}

        return partitions

    def _write_shard(self, shard, partition_number, shard_number):
        """Write a single shard into the tfrecords file.

        Note that the dataset-specific part is captured in function
        create_example_proto() which needs to be overridden for each
        specific dataset.

        Args:
            shard (list): A list of frame IDs for this shard.
            partition_number (int): Current partition (fold) index.
            shard_number (int): Current shard index.

        Returns:
            object_count (Counter): The number of written objects per target class.
        """
        logger.info('Writing partition {}, shard {}'.format(partition_number, shard_number))
        status_logging.get_status_logger().write(
            message='Writing partition {}, shard {}'.format(partition_number, shard_number)
        )
        output = self.output_filename

        if self.output_partitions != 0:
            output = '{}-fold-{:03d}-of-{:03d}'.format(output, partition_number,
                                                       self.output_partitions)
        if self.output_shards[partition_number] != 0:
            output = '{}-shard-{:05d}-of-{:05d}'.format(output, shard_number,
                                                        self.output_shards[partition_number])

        object_count = Counter()

        # Store all the data for the shard.
        writer = tf.python_io.TFRecordWriter(output)
        for frame_id in shard:

            # Create the Example with all labels for this frame_id.
            example = self._create_example_proto(frame_id, partition_number)

            # The example might be skipped e.g. due to missing labels.
            if example is not None:
                # Apply modifications to the current sample such as filtering and duplication.
                # Only samples in the training set are modified.
                modified_examples = self.sample_modifier.modify_sample(example, partition_number)

                # Write the list of (possibly) modified samples.
                frame_object_count = Counter()
                for modified_example in modified_examples:
                    # Serialize the example.
                    writer.write(modified_example.SerializeToString())

                    # Count objects that got written per target class.
                    frame_object_count += self._count_targets(modified_example)

                object_count += frame_object_count

        writer.close()

        return object_count

    def _shard(self, partitions):
        """Shard each partition."""
        shards = []
        for partition, num_shards in zip(partitions, self.output_shards):
            num_shards = max(num_shards, 1)  # 0 means 1 shard.

            result = []
            if len(partition) == 0:
                continue
            shard_size = len(partition) // num_shards
            for i in range(num_shards):
                begin = i * shard_size
                end = (i + 1) * shard_size if i + 1 < num_shards else len(partition)
                result.append(partition[begin:end])
            shards.append(result)
        return shards

    def _write_partitions(self, partitions):
        """Shard and write partitions into tfrecords.

        Args:
            partitions (list): A list of list of frame IDs.

        Returns:
            object_count (Counter): The total number of objects per target class.
        """
        # Divide partitions into shards.
        sharded_partitions = self._shard(partitions)

        # Write .tfrecords to disk for each partition and shard.
        # Also count the target objects per partition and over the whole dataset.
        object_count = Counter()
        for p, partition in enumerate(sharded_partitions):
            partition_object_count = Counter()
            for s, shard in enumerate(partition):
                shard_object_count = self._write_shard(shard, p, s)
                partition_object_count += shard_object_count

            # Log the count in this partition and increase total
            # object count.
            self._log_object_count(partition_object_count)
            object_count += partition_object_count

        return object_count

    def convert(self):
        """Do the dataset conversion."""
        # Load coco annotations for each partition.
        partitions = self._partition()

        # Shard and write the partitions to tfrecords.
        object_count = self._write_partitions(partitions)

        # Log how many objects per class got written in total.
        logger.info("Cumulative object statistics")
        cumulative_count_dict = {
            target_class.decode("ascii"): object_count.get(target_class)
            for target_class in object_count.keys()
        }
        s_logger = status_logging.get_status_logger()
        s_logger.categorical = {"num_objects": cumulative_count_dict}
        s_logger.write(
            message="Cumulative object statistics"
        )
        self._log_object_count(object_count)

        # Print out the class map
        log_str = "Class map. \nLabel in GT: Label in tfrecords file "
        for key, value in six.iteritems(self.class_map):
            log_str += "\n{}: {}".format(key, value)
        logger.info(log_str)
        s_logger.write(message=log_str)
        note_string = (
            "For the dataset_config in the experiment_spec, "
            "please use labels in the tfrecords file, while writing the classmap.\n"
        )
        print(note_string)
        s_logger.write(message=note_string)

        logger.info("Tfrecords generation complete.")
        s_logger.write(
            status_level=status_logging.Status.SUCCESS,
            message="TFRecords generation complete."
            )

        # Save labels with error to a JSON file
        self._save_log_warnings()

    def _create_example_proto(self, img_id, partition_number):
        """Generate the example proto for this img.

        Args:
            img_id (int): The img id.
            partition_number (string): The partition number.

        Returns:
            example (tf.train.Example): An Example containing all labels for the frame.
        """
        # Create proto for the training example.

        # Load neccesary dict for img and annotations
        img_dict = self.coco[partition_number].loadImgs(img_id)[0]
        annIds = self.coco[partition_number].getAnnIds(imgIds=img_dict['id'])
        ann_dict = self.coco[partition_number].loadAnns(annIds)

        orig_filename = self.coco[partition_number].loadImgs(img_id)[0]['file_name']

        # Need to remove the file extensions to meet KITTI format
        # Prepend the image directory name of the current partition
        img_id = os.path.join(self.img_dir_names[partition_number], orig_filename.rsplit(".", 1)[0])
        example = self._example_proto(img_id, img_dict)

        if self.use_dali:
            width, height = img_dict['width'], img_dict['height']
            img_full_path = os.path.join(self.root_dir, self.img_dir_names[partition_number])
            self._add_image(example, img_dict, img_dir=img_full_path)
            self._add_targets(example, img_dict, ann_dict, width, height)
        else:
            self._add_targets(example, img_dict, ann_dict)

        return example

    def _example_proto(self, img_id, img_dict):
        """Generate a base Example protobuf to which COCO-specific features are added."""
        width, height = img_dict['width'], img_dict['height']

        example = tf.train.Example(features=tf.train.Features(feature={
            'frame/id': _bytes_feature(img_id.encode('utf-8')),
            'frame/height': _int64_feature(height),
            'frame/width': _int64_feature(width),
        }))

        return example

    def _add_image(self, example, img_dict, img_dir):
        """Add encoded image to example."""
        image_file = os.path.join(img_dir, img_dict['file_name'])
        image_string = open(image_file, "rb").read()
        f = example.features.feature
        f['frame/encoded'].MergeFrom(_bytes_feature(image_string))

    def _add_targets(self, example, img_dict, ann_dict, width=None, height=None):
        """Add COCO target features such as bbox to the Example protobuf.

        Args:
            example (tf.train.Example): The Example protobuf for this frame.
            frame_id (string): Frame id.
        """
        object_classes = []
        truncation = []
        occlusion = []
        observation_angle = []
        coordinates_x1 = []
        coordinates_y1 = []
        coordinates_x2 = []
        coordinates_y2 = []
        world_bbox_h = []
        world_bbox_w = []
        world_bbox_l = []
        world_bbox_x = []
        world_bbox_y = []
        world_bbox_z = []
        world_bbox_rot_y = []
        object_class_ids = []

        # reads the labels as a list of tuples
        labels = ann_dict
        if isinstance(labels, tuple):
            labels = [labels]

        for label in labels:
            # Convert x,y,w,h to x1,y1,x2,y2 format
            bbox = label['bbox']
            bbox = [bbox[0], bbox[1], (bbox[2] + bbox[0]), (bbox[3] + bbox[1])]

            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            # Check to make sure the coordinates are 'ltrb' format.
            error_string = "Top left coordinate must be less than bottom right."\
                           f"Error in Img Id {img_dict['id']} and Ann Id {label['id']}. \n"\
                           f"Coordinates: x1 = {x1}, x2 = {x2}, y1: {y1}, y2: {y2}"
            if not (x1 < x2 and y1 < y2):
                logger.debug(error_string)
                logger.debug("Skipping this object")
                self.log_warning[label['id']] = [x1, y1, x2, y2]
                continue

            # Convert category id to actual category
            cat = self.idx2class[label['category_id']]

            # Map object classes as they are in the dataset to target classes of the model
            self.class_map[cat] = cat.lower()
            object_class = cat.lower()
            if self.use_dali:
                if (str(object_class) not in self.class2idx):
                    logger.debug("Skipping the class {} in dataset".format(object_class))
                    continue

            object_classes.append(object_class)

            truncation.append(0)
            occlusion.append(0)
            observation_angle.append(0)
            if self.use_dali:
                # @tylerz: DALI requires relative coordinates and integer
                coordinates_x1.append(float(bbox[0]) / width)
                coordinates_y1.append(float(bbox[1]) / height)
                coordinates_x2.append(float(bbox[2]) / width)
                coordinates_y2.append(float(bbox[3]) / height)
                object_class_id = self.class2idx[str(object_class)]
                object_class_ids.append(object_class_id)
            else:
                coordinates_x1.append(bbox[0])
                coordinates_y1.append(bbox[1])
                coordinates_x2.append(bbox[2])
                coordinates_y2.append(bbox[3])
            world_bbox_h.append(0)
            world_bbox_w.append(0)
            world_bbox_l.append(0)
            world_bbox_x.append(0)
            world_bbox_y.append(0)
            world_bbox_z.append(0)
            world_bbox_rot_y.append(0)

        f = example.features.feature
        if self.use_dali:
            f['target/object_class_id'].MergeFrom(_float_feature(*object_class_ids))
        else:
            f['target/object_class'].MergeFrom(_bytes_feature(*object_classes))
        f['target/truncation'].MergeFrom(_float_feature(*truncation))
        f['target/occlusion'].MergeFrom(_int64_feature(*occlusion))
        f['target/observation_angle'].MergeFrom(_float_feature(*observation_angle))
        f['target/coordinates_x1'].MergeFrom(_float_feature(*coordinates_x1))
        f['target/coordinates_y1'].MergeFrom(_float_feature(*coordinates_y1))
        f['target/coordinates_x2'].MergeFrom(_float_feature(*coordinates_x2))
        f['target/coordinates_y2'].MergeFrom(_float_feature(*coordinates_y2))
        f['target/world_bbox_h'].MergeFrom(_float_feature(*world_bbox_h))
        f['target/world_bbox_w'].MergeFrom(_float_feature(*world_bbox_w))
        f['target/world_bbox_l'].MergeFrom(_float_feature(*world_bbox_l))
        f['target/world_bbox_x'].MergeFrom(_float_feature(*world_bbox_x))
        f['target/world_bbox_y'].MergeFrom(_float_feature(*world_bbox_y))
        f['target/world_bbox_z'].MergeFrom(_float_feature(*world_bbox_z))
        f['target/world_bbox_rot_y'].MergeFrom(_float_feature(*world_bbox_rot_y))

    def _count_targets(self, example):
        """Count the target objects in the given example protobuf.

        Args:
            example (tf.train.Example): Example protobuf containing the labels for a frame.

        Returns:
            object_count (Counter): Number of objects per target class.
        """
        target_classes = example.features.feature['target/object_class'].bytes_list.value
        if len(target_classes) == 0:
            target_classes_id = example.features.feature['target/object_class_id'].float_list.value
            if len(target_classes_id) != 0:
                if self.idx2class is None:
                    self.idx2class = {self.class2idx[k] : k for k in self.class2idx}
                target_classes = []
                for idx in target_classes_id:
                    target_classes.append(self.idx2class[idx].encode("ascii"))

        object_count = Counter(target_classes)
        return object_count
