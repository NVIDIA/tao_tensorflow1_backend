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

"""Converts a KITTI detection dataset to TFRecords."""


from __future__ import absolute_import
from __future__ import print_function
from collections import Counter
import json
import logging
import os
import random
import numpy as np
from PIL import Image
from six.moves import range
import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _bytes_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _float_feature
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _int64_feature

from nvidia_tao_tf1.cv.detectnet_v2.dataio.dataset_converter_lib import DatasetConverter


logger = logging.getLogger(__name__)


class KITTIConverter(DatasetConverter):
    """Converts a KITTI detection dataset to TFRecords."""

    def __init__(self, root_directory_path, num_partitions, num_shards,
                 output_filename,
                 sample_modifier,
                 image_dir_name=None,
                 label_dir_name=None,
                 kitti_sequence_to_frames_file=None,
                 point_clouds_dir=None,
                 calibrations_dir=None,
                 extension='.png',
                 partition_mode='sequence',
                 val_split=None,
                 use_dali=False,
                 class2idx=None):
        """Initialize the converter.

        Args:
            root_directory_path (string): Dataset directory path relative to data root.
            num_partitions (int): Number of partitions (folds).
            num_shards (int): Number of shards.
            output_filename (str): Path for the output file.
            sample_modifier(SampleModifier): An instance of sample modifier
                that does e.g. duplication and filtering of samples.
            image_dir_name (str): Name of the subdirectory containing images.
            label_dir_name (str): Name of the subdirectory containing the label files for the
                respective images in image_dir_name
            kitti_sequence_to_frames_file (str): name of the kitti sequence to frames map file in
                root directory path. This file contains a mapping of the sequences to images in
                image_dir_name.
            point_clouds_dir (str): Path to the point cloud data within root_dirctory_path.
            calibrations_dir (str): Path to the calibration data within root_dirctory_path.
            extension (str): Extension of the images in the dataset directory.
            partition_mode (str): Mode to partitition the dataset. We only support sequence or
                random split mode. In the sequence mode, it is mandatory to instantiate the
                kitti sequence to frames file. Also, any arbitrary number of partitions maybe
                used. However, for random split, the sequence map file is ignored and only 2
                partitions can every be used. Here, the data is divided into two folds
                    1. validation fold
                    2. training fold
                Validation fold (defaults to fold=0) contains val_split% of data, while train
                fold contains (100-val_split)% of data.
            val_split (int): Percentage split for validation. This is used with the random
                partition mode only.
        """
        super(KITTIConverter, self).__init__(
            root_directory_path=root_directory_path,
            num_partitions=num_partitions,
            num_shards=num_shards,
            output_filename=output_filename,
            sample_modifier=sample_modifier)

        # KITTI defaults.
        self.images_dir = image_dir_name
        self.labels_dir = label_dir_name
        self.point_clouds_dir = point_clouds_dir
        self.calibrations_dir = calibrations_dir
        self.extension = extension
        self.partition_mode = partition_mode
        self.sequence_to_frames_file = kitti_sequence_to_frames_file
        self.val_split = val_split / 100.
        self.use_dali = use_dali
        self.class2idx = class2idx
        self.idx2class = None

    def _partition(self):
        """Partition KITTI dataset to self.output_partitions partitions based on sequences.

        The following code is a modified version of the KITTISplitter class in Rumpy.

        Returns:
            partitions (list): A list of lists of frame ids, one list per partition.
        """
        logger.debug("Generating partitions")
        s_logger = status_logging.get_status_logger()
        s_logger.write(message="Generating partitions")
        partitions = [[] for _ in range(self.output_partitions)]
        # Sequence wise parition to n partitions.
        if self.partition_mode == 'sequence':
            if not self.sequence_to_frames_file:
                raise ValueError("Kitti sequence to frames file is required for "
                                 "sequence wise paritioning. Please set this as the relative "
                                 "path to the file from `root_directory_path`")
            # Create sequence to frames mapping.
            self.sequence_to_frames_map = self._read_sequence_to_frames_file()
            if self.output_partitions > 1:
                sorted_sequences = sorted(iter(self.sequence_to_frames_map.items()),
                                          key=lambda k_v: (-len(k_v[1]), k_v[0]))
                total_frames = 0

                for counter, (_, frame_ids) in enumerate(sorted_sequences):
                    total_frames += len(frame_ids)
                    partition_idx = counter % self.output_partitions
                    partitions[partition_idx].extend(frame_ids)

                logger.debug("Total number of frames: {}".format(total_frames))
                s_logger.kpi = {
                    "num_images": total_frames
                }
                s_logger.write(
                    message=f"Total number of images: {total_frames}"
                )
                # in Rumpy with 5 folds, the first validation bucket contains the fifth sequence.
                # Similarly, the second validation bucket contains samples from the fourth sequence,
                # and so on. Thus, the partition order needs to be reversed to match the Rumpy
                # validation buckets.
                partitions = partitions[::-1]
            else:
                partitions = [[frame for frames in list(self.sequence_to_frames_map.values())
                               for frame in frames]]
                s_logger.kpi = {
                    "num_images": len(partitions[0])
                }
                s_logger.write(
                    message=f"Total number of images: {len(partitions[0])}"
                )
        # Paritioning data in random to train and val split.
        elif self.partition_mode == 'random':
            assert self.output_partitions == 2, "Invalid number of partitions ({}) "\
                   "for random split mode.".format(self.output_partitions)
            assert 0 <= self.val_split < 1, (
                "Validation split must satisfy the criteria, 0 <= val_split < 100. "
            )
            images_root = os.path.join(self.root_dir, self.images_dir)
            images_list = [os.path.splitext(imfile)[0] for imfile in
                           sorted(os.listdir(images_root)) if
                           imfile.endswith(self.extension)]
            total_num_images = len(images_list)
            num_val_images = (int)(self.val_split * total_num_images)
            logger.debug("Validation percentage: {}".format(self.val_split))
            partitions[0].extend(images_list[:num_val_images])
            partitions[1].extend(images_list[num_val_images:])
            for part in partitions:
                random.shuffle(part)
            logger.info("Num images in\nTrain: {}\tVal: {}".format(len(partitions[1]),
                                                                   len(partitions[0])))

            s_logger.kpi = {
                "num_images": total_num_images
            }
            s_logger.write(
                message="Num images in\nTrain: {}\tVal: {}".format(
                    len(partitions[1]),
                    len(partitions[0])
                )
            )

            if self.val_split == 0:
                logger.info("Skipped validation data...")
                s_logger.write(message="Skipped validation data.")
            else:
                validation_note = (
                    "Validation data in partition 0. Hence, while choosing the validation"
                    "set during training choose validation_fold 0."
                )
                logger.info(validation_note)
                s_logger.write(message=validation_note)
        else:
            raise NotImplementedError("Unknown partition mode. Please stick to either "
                                      "random or sequence")

        return partitions

    def _create_example_proto(self, frame_id):
        """Generate the example proto for this frame.

        Args:
            frame_id (string): The frame id.

        Returns:
            example (tf.train.Example): An Example containing all labels for the frame.
        """
        # Create proto for the training example. Populate with frame attributes.
        example = self._example_proto(frame_id)

        if self.use_dali:
            width, height = self._get_image_size(frame_id)
            self._add_image(example, frame_id)
            self._add_targets(example, frame_id, width, height)
        # Add labels.
        else:
            self._add_targets(example, frame_id)
            self._add_point_cloud(example, frame_id)
            self._add_calibrations(example, frame_id)

        return example

    def _add_image(self, example, frame_id):
        """Add encoded image to example."""
        image_file = os.path.join(self.root_dir, self.images_dir, frame_id + self.extension)
        image_string = open(image_file, "rb").read()
        f = example.features.feature
        f['frame/encoded'].MergeFrom(_bytes_feature(image_string))

    def _add_point_cloud(self, example, frame_id):
        """Add path to the point cloud file in the Example protobuf."""
        if self.point_clouds_dir is not None:
            frame_id = os.path.join(self.point_clouds_dir, frame_id)
            f = example.features.feature
            f['point_cloud/id'].MergeFrom(_bytes_feature(frame_id.encode('utf-8')))
            f['point_cloud/num_input_channels'].MergeFrom(_int64_feature(4))

    def _add_calibrations(self, example, frame_id):
        """Add calibration matrices in the Example protobuf."""
        if self.calibrations_dir is not None:
            calibration_file = os.path.join(self.root_dir,
                                            self.calibrations_dir, '{}.txt'.format(frame_id))
            self._add_calibration_matrices(example, calibration_file)

    def _read_sequence_to_frames_file(self):
        with open(os.path.join(self.root_dir, self.sequence_to_frames_file), 'r') as f:
            sequence_to_frames_map = json.load(f)

        return sequence_to_frames_map

    def _get_image_size(self, frame_id):
        """Read image size from the image file, image sizes vary in KITTI."""
        image_file = os.path.join(self.root_dir, self.images_dir, frame_id + self.extension)
        width, height = Image.open(image_file).size

        return width, height

    def _example_proto(self, frame_id):
        """Generate a base Example protobuf to which KITTI-specific features are added."""
        width, height = self._get_image_size(frame_id)

        # Add the image directory name to the frame id so that images and
        # point clouds can be easily stored in separate folders.
        frame_id = os.path.join(self.images_dir, frame_id)

        example = tf.train.Example(features=tf.train.Features(feature={
            'frame/id': _bytes_feature(frame_id.encode('utf-8')),
            'frame/height': _int64_feature(height),
            'frame/width': _int64_feature(width),
        }))

        return example

    def _add_targets(self, example, frame_id, width=None, height=None):
        """Add KITTI target features such as bbox to the Example protobuf.

        Reads labels from KITTI txt files with following fields:
        (From Kitti devkit's README)
        1    type         Describes the type of object: 'Car', 'Van',
                          'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist',
                          'Tram', 'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated),
                          where truncated refers to the object leaving image
                          boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                          0 = fully visible, 1 = partly occluded
                          2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based
                          index): contains left, top, right, bottom pixel
                          coordinates
        3    dimensions   3D object dimensions: height, width, length (in
                          meters)
        3    location     3D object location x,y,z in camera coordinates (in
                          meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates
                          [-pi..pi]

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
        label_file = os.path.join(self.root_dir, self.labels_dir, '{}.txt'.format(frame_id))
        # np.genfromtxt will fail if the class name is integer literal: '1', etc
        with open(label_file) as lf:
            labels = lf.readlines()
        labels = [l.strip() for l in labels if l.strip()]
        labels = [l.split() for l in labels]
        # Following steps require the class names to be bytes
        labels = [[l[0].encode("utf-8")] + [float(x) for x in l[1:]] for l in labels]

        for label in labels:
            assert len(label) == 15, 'Ground truth kitti labels should have only 15 fields.'
            x1 = int(label[4])
            y1 = int(label[5])
            x2 = int(label[6])
            y2 = int(label[7])

            # Check to make sure the coordinates are 'ltrb' format.
            error_string = "Top left coordinate must be less than bottom right."\
                           "Error in object {} of label_file {}. \nCoordinates: "\
                           "x1 = {}, x2 = {}, y1: {}, y2: {}".format(labels.index(label),
                                                                     label_file,
                                                                     x1, x2, y1, y2)
            if not (x1 < x2 and y1 < y2):
                logger.debug(error_string)
                logger.debug("Skipping this object")
                # @scha: KITTI does not have annotation id
                self.log_warning[f"{label_file}_{labels.index(label)}"] = [x1, y1, x2, y2]
                continue

            # Map object classes as they are in the dataset to target classes of the model
            self.class_map[label[0]] = label[0].lower()
            object_class = label[0].lower()
            if self.use_dali:
                if (object_class.decode() not in self.class2idx):
                    logger.debug("Skipping the class {} in dataset".format(object_class))
                    continue

            object_classes.append(object_class)

            truncation.append(label[1])
            occlusion.append(int(label[2]))
            observation_angle.append(label[3])
            if self.use_dali:
                # @tylerz: DALI requires relative coordinates and integer
                coordinates_x1.append(float(label[4]) / width)
                coordinates_y1.append(float(label[5]) / height)
                coordinates_x2.append(float(label[6]) / width)
                coordinates_y2.append(float(label[7]) / height)
                object_class_id = self.class2idx[object_class.decode()]
                object_class_ids.append(object_class_id)
            else:
                coordinates_x1.append(label[4])
                coordinates_y1.append(label[5])
                coordinates_x2.append(label[6])
                coordinates_y2.append(label[7])
            world_bbox_h.append(label[8])
            world_bbox_w.append(label[9])
            world_bbox_l.append(label[10])
            world_bbox_x.append(label[11])
            world_bbox_y.append(label[12])
            world_bbox_z.append(label[13])
            world_bbox_rot_y.append(label[14])

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

    def _add_calibration_matrices(self, example, filename):
        """Add KITTI calibration matrices to the Example protobuf.

        Adds the following matrices to the Example protobuf:
            - 4x4 transformation matrix from Lidar coordinates to camera coordinates.
            - 3x4 projection matrix from Lidar coordinates to image plane.

        Args:
            example: Protobuf to which the matrices are added.
            filename: Absolute path to the calibration file.
        """
        # KITTI calibration file has the following format (each matrix is given on a separate
        # line in the file in the following order):
        # P0: 3x4 projection matrix after rectification for camera 0 (12 floats)
        # P1: 3x4 projection matrix after rectification for camera 1 (12 floats)
        # P2: 3x4 projection matrix after rectification for camera 2 (12 floats)
        # P3: 3x4 projection matrix after rectification for camera 3 (12 floats)
        # R0_rect: 3x3 rectifying rotation matrix (9 floats)
        # Tr_velo_to_cam: 3x4 transformation matrix from Lidar to reference camera (12 floats)
        # Tr_imu_to_velo: 3x4 transformation matrix from GPS/IMU to Lidar (12 floats)
        if os.path.isfile(filename):
            # Camera projection matrix after rectification. Projects a 3D point X = (x, y, z, 1)^T
            # in rectified (rotated) camera coordinates to a point Y = (u, v, 1)^T in the camera
            # image with Y = P2*X. P2 corresponds to the left color image camera.
            P2 = np.genfromtxt(filename, dtype=np.float32, skip_header=2,
                               skip_footer=4, usecols=tuple(range(1, 13)))
            # Rectifying rotation matrix
            R0_rect = np.genfromtxt(filename, dtype=np.float32, skip_header=4,
                                    skip_footer=2, usecols=tuple(range(1, 10)))
            # Rigid body transformation matrix from Lidar coordinates to camera coordinates
            Tr_velo_to_cam = np.genfromtxt(filename, dtype=np.float32, skip_header=5,
                                           skip_footer=1, usecols=tuple(range(1, 13)))
        else:
            raise IOError("Calibration file %s not found." % filename)

        P2 = P2.reshape((3, 4))

        # Expand R0_rect by appending 4th row and column of zeros, and setting R0[3, 3] = 1
        R0_rect = R0_rect.reshape((3, 3))
        R0 = np.eye(4)
        R0[:3, :3] = R0_rect

        # Expand Tr_velo_to_cam by appending 4th row of zeros, and setting Tr[3, 3] = 1
        Tr_velo_to_cam = Tr_velo_to_cam.reshape((3, 4))
        Tr = np.eye(4)
        Tr[:3, :4] = Tr_velo_to_cam

        # Transformation matrix T_lidar_to_camera = R0*Tr_velo_to_cam from Lidar coordinates
        # (x, y, z, 1)^T to reference camera coordinates (u, v, w, q)^T
        T_lidar_to_camera = np.dot(R0, Tr)

        # Projection matrix P_lidar_to_image = P2*T_lidar_to_camera from Lidar coordinates
        # (x, y, z, 1)^T to image coordinates (u, v, w)^T
        P_lidar_to_image = np.dot(P2, T_lidar_to_camera)

        f = example.features.feature
        f['calibration/T_lidar_to_camera'].MergeFrom(_float_feature(*T_lidar_to_camera.flatten()))
        f['calibration/P_lidar_to_image'].MergeFrom(_float_feature(*P_lidar_to_image.flatten()))

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
